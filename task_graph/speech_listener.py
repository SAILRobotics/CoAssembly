"""speech_listener.py — Background speech capture + Parakeet ASR.

Usage:
    listener = SpeechListener(device="bluez_source...", wake_word="hey robot")
    listener.start()

    # each UI frame:
    events = listener.poll()   # updates internal state; returns important events
    print(listener.current_status, listener.current_rms, listener.remaining_time)
    for kind, payload in events:
        if kind == "transcript": ...

    listener.close()
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections import deque

import numpy as np


class SpeechListener:
    RATE              = 16000
    CHANNELS          = 1
    SAMPLES_PER_BLOCK = 1024
    BYTES_PER_BLOCK   = SAMPLES_PER_BLOCK * 2   # int16

    # current_status values
    STATUS_LOADING    = "loading"
    STATUS_IDLE       = "idle"
    STATUS_SPEECH     = "speech"
    STATUS_QUEUED     = "queued"
    STATUS_TRANSCRIBE = "transcribing"
    STATUS_LISTENING  = "listening"
    STATUS_ERROR      = "error"

    def __init__(
        self,
        device:           str   = "bluez_source.50_C2_ED_43_95_C8.handsfree_head_unit",
        wake_word:        str   = "hey robot",
        listen_timeout:   float = 30.0,
        rms_threshold:    float = 0.015,
        end_silence:      float = 1.5,
        min_utterance:    float = 0.5,
        max_utterance:    float = 15.0,
        pre_roll:         float = 0.3,
        model_name:       str   = "nvidia/parakeet-tdt-0.6b-v2",
        max_transcripts:  int   = 8,
    ) -> None:
        self._device          = device
        self._wake_word       = wake_word.lower()
        self._listen_timeout  = listen_timeout
        self._rms_threshold   = rms_threshold
        self._end_silence     = end_silence
        self._min_utterance   = min_utterance
        self._max_utterance   = max_utterance
        self._pre_roll        = pre_roll
        self._model_name      = model_name

        self._audio_queue = queue.Queue(maxsize=3)
        self._event_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._model_ready = threading.Event()
        self._running     = False
        self._process: "subprocess.Popen | None" = None

        # Public state — read after poll()
        self.current_status:   str   = self.STATUS_LOADING
        self.current_rms:      float = 0.0
        self.listening_active: bool  = False
        self.last_speech_time: float = 0.0
        self.transcript_history: deque[str] = deque(maxlen=max_transcripts)
        self.wake_word = wake_word   # original case for display

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._process = subprocess.Popen(
            ["parec", f"--device={self._device}", "--format=s16le",
             f"--rate={self.RATE}", f"--channels={self.CHANNELS}"],
            stdout=subprocess.PIPE,
        )
        self._running = True
        threading.Thread(target=self._asr_worker, daemon=True).start()
        threading.Thread(target=self._vad_loop,   daemon=True).start()

    def close(self) -> None:
        self._running = False
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._process:
            self._process.terminate()
            self._process.wait()

    # ── Background threads ────────────────────────────────────────────────────

    def _asr_worker(self) -> None:
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            self._event_queue.put(("error", "NeMo not installed — pip install nemo-toolkit[asr]"))
            return

        self._event_queue.put(("_status", self.STATUS_LOADING))
        try:
            model = nemo_asr.models.ASRModel.from_pretrained(self._model_name)
            model.eval()
        except Exception as e:
            self._event_queue.put(("error", f"Model load failed: {e}"))
            return

        self._model_ready.set()
        self._event_queue.put(("_status", self.STATUS_IDLE))

        while self._running:
            try:
                utterance = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if utterance is None:
                break
            self._event_queue.put(("_status", self.STATUS_TRANSCRIBE))
            try:
                output = model.transcribe([utterance], batch_size=1)[0]
                text = (output.text if hasattr(output, "text") else str(output)).strip()
                if text:
                    self._event_queue.put(("transcript", text))
                else:
                    self._event_queue.put(("_status", self.STATUS_LISTENING
                                           if self.listening_active else self.STATUS_IDLE))
            except Exception as e:
                self._event_queue.put(("error", f"Transcription error: {e}"))

    def _vad_loop(self) -> None:
        RATE = self.RATE
        pre_roll_n    = max(1, int(np.ceil(self._pre_roll * RATE / self.SAMPLES_PER_BLOCK)))
        pre_roll      = deque(maxlen=pre_roll_n)
        utt_blocks:   list = []
        utt_samples   = 0
        sil_samples   = 0
        speech_active = False

        while self._running:
            raw = self._process.stdout.read(self.BYTES_PER_BLOCK)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))
            self._event_queue.put(("rms", rms))

            if rms >= self._rms_threshold:
                if not speech_active:
                    speech_active = True
                    utt_blocks    = list(pre_roll)
                    utt_samples   = sum(len(b) for b in utt_blocks)
                    self._event_queue.put(("_status", self.STATUS_SPEECH))
                utt_blocks.append(samples.copy())
                utt_samples += len(samples)
                sil_samples  = 0
            elif speech_active:
                utt_blocks.append(samples.copy())
                utt_samples += len(samples)
                sil_samples  += len(samples)

            pre_roll.append(samples.copy())

            finished = speech_active and sil_samples  >= int(self._end_silence  * RATE)
            too_long = speech_active and utt_samples  >= int(self._max_utterance * RATE)
            if finished or too_long:
                if self._model_ready.is_set():
                    utterance = np.concatenate(utt_blocks).astype(np.float32, copy=False)
                    if len(utterance) >= int(self._min_utterance * RATE):
                        try:
                            self._audio_queue.put_nowait(utterance)
                            self._event_queue.put(("_status", self.STATUS_QUEUED))
                        except queue.Full:
                            print("[SpeechListener] queue full — dropping utterance")
                utt_blocks  = []
                utt_samples = 0
                sil_samples = 0
                speech_active = False

    # ── Poll (call each UI frame) ─────────────────────────────────────────────

    def poll(self) -> list[tuple[str, str]]:
        """Drain pending events, update internal state. Returns notable events:
        ("wake_word", text), ("transcript", text), ("ready", ""), ("error", msg)."""
        now      = time.time()
        notable: list[tuple[str, str]] = []

        if self.listening_active and now - self.last_speech_time > self._listen_timeout:
            self.listening_active = False
            self.current_status   = self.STATUS_IDLE
            notable.append(("timeout", ""))

        while True:
            try:
                kind, payload = self._event_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "rms":
                self.current_rms = payload
            elif kind == "_status":
                self.current_status = payload
                if payload == self.STATUS_IDLE:
                    notable.append(("ready", ""))
            elif kind == "transcript":
                text = str(payload)
                if not self.listening_active:
                    if self._wake_word in text.lower():
                        self.listening_active = True
                        self.last_speech_time = now
                        self.current_status   = self.STATUS_LISTENING
                        notable.append(("wake_word", text))
                    # else silently dropped (idle, not wake word)
                else:
                    self.last_speech_time = now
                    self.transcript_history.append(text)
                    self.current_status = self.STATUS_LISTENING
                    notable.append(("transcript", text))
            elif kind == "error":
                self.current_status = self.STATUS_ERROR
                notable.append(("error", str(payload)))

        return notable

    @property
    def remaining_time(self) -> float:
        if not self.listening_active:
            return 0.0
        return max(0.0, self._listen_timeout - (time.time() - self.last_speech_time))
