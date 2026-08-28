"""speech_listener.py — Background speech capture + Parakeet ASR.

Usage:
    listener = SpeechListener(device="bluez_source...")
    listener.start()

    # each UI frame:
    events = listener.poll()   # updates internal state; returns important events
    print(listener.current_status, listener.current_rms, listener.remaining_time)
    for kind, payload in events:
        if kind == "transcript": ...

    listener.close()
"""

from __future__ import annotations

import os

# On Windows/conda, PyTorch and MKL/LLVM can each load a separate OpenMP
# runtime, which aborts with "OMP: Error #15". Allow the duplicate. Harmless
# on Linux. Must be set before torch/nemo import, so keep it at module top.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import queue
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    from .inference_lock import GPU_INFERENCE_LOCK
except ImportError:  # Script execution: task_graph is placed directly on sys.path.
    from inference_lock import GPU_INFERENCE_LOCK

# ── Audio-capture backend ─────────────────────────────────────────────────────
# "auto" selects sounddevice on Windows and PulseAudio (parec) elsewhere.
# Force a specific backend by setting this to "sounddevice" or "pulseaudio".
AUDIO_BACKEND = "auto"

# Default input device per backend, used when no device is passed (or when a
# PulseAudio source name is passed to the sounddevice backend, which can't use
# it). For sounddevice: an int index or case-insensitive name substring, or
# None for the system default input.
# AOC ACW4212 headset used by microphone_test.py. Its microphone exists only
# while the Bluetooth card is in the hands-free profile.
DEFAULT_PULSEAUDIO_DEVICE  = "bluez_source.41_42_D1_99_84_1D.handsfree_head_unit"
DEFAULT_SOUNDDEVICE_DEVICE = "Microphone Array"   # laptop built-in mic array


def _resolve_backend(backend: str | None) -> str:
    backend = backend or AUDIO_BACKEND
    if backend != "auto":
        return backend
    return "sounddevice" if sys.platform == "win32" else "pulseaudio"


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
    STATUS_DISABLED   = "disabled"
    STATUS_ERROR      = "error"

    def __init__(
        self,
        device:           str | None = None,
        wake_word:        str | None = None,
        listen_timeout:   float = 30.0,
        rms_threshold:    float = 0.015,
        end_silence:      float = 1.5,
        min_utterance:    float = 0.5,
        max_utterance:    float = 15.0,
        pre_roll:         float = 0.3,
        model_name:       str   = "nvidia/parakeet-tdt-0.6b-v2",
        max_transcripts:  int   = 8,
        backend:          str | None = None,
    ) -> None:
        self._backend         = _resolve_backend(backend)
        self._device          = self._resolve_device(device)
        self._wake_word       = (wake_word or "").strip().lower()
        self.always_listening = not bool(self._wake_word)
        self._listen_timeout  = listen_timeout
        self._rms_threshold   = rms_threshold
        self._end_silence     = end_silence
        self._min_utterance   = min_utterance
        self._max_utterance   = max_utterance
        self._pre_roll        = pre_roll
        self._model_name      = model_name

        self._audio_queue = queue.Queue(maxsize=3)
        self._event_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        # Raw float32 mono blocks flow from whichever capture backend is active
        # into this queue; the VAD loop consumes it (replaces reading parec's
        # stdout pipe directly, so both backends share the same downstream path).
        self._raw_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._model_ready = threading.Event()
        self._input_enabled = threading.Event()
        self._input_enabled.set()
        self._running     = False
        self._process: "subprocess.Popen | None" = None
        self._sd_stream = None   # sounddevice.InputStream when backend == sounddevice

        # Public state — read after poll()
        self.current_status:   str   = self.STATUS_LOADING
        self.current_rms:      float = 0.0
        self.listening_active: bool  = self.always_listening
        self.last_speech_time: float = 0.0
        self.transcript_history: deque[str] = deque(maxlen=max_transcripts)
        self.wake_word = (wake_word or "").strip()  # original case for display

    def _resolve_device(self, device):
        """Pick a sensible device for the active backend. A PulseAudio source
        name is meaningless to sounddevice, so fall back to the Windows default
        mic in that case."""
        if self._backend == "sounddevice":
            if device is None or "bluez" in str(device).lower():
                return DEFAULT_SOUNDDEVICE_DEVICE
            return device
        return device or DEFAULT_PULSEAUDIO_DEVICE

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        if self._backend == "sounddevice":
            self._start_sounddevice()
        else:
            self._start_pulseaudio()
        threading.Thread(target=self._asr_worker, daemon=True).start()
        threading.Thread(target=self._vad_loop,   daemon=True).start()

    def close(self) -> None:
        self._running = False
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None

    @property
    def input_enabled(self) -> bool:
        return self._input_enabled.is_set()

    def set_input_enabled(self, enabled: bool) -> None:
        """Pause/resume recognition without unloading the ASR model."""
        if enabled:
            self._input_enabled.set()
            self.listening_active = self.always_listening
            self.current_status = (self.STATUS_LISTENING if self.always_listening
                                   else self.STATUS_IDLE)
            return
        self._input_enabled.clear()
        self.listening_active = False
        self.current_rms = 0.0
        self.current_status = self.STATUS_DISABLED
        for pending in (self._audio_queue, self._raw_queue):
            while True:
                try:
                    pending.get_nowait()
                except queue.Empty:
                    break

    # ── Audio capture backends ────────────────────────────────────────────────

    @staticmethod
    def _bluetooth_card_from_source(source: str | None) -> str | None:
        """Convert ``bluez_source.<address>.<profile>`` to its card name."""
        source = str(source or "")
        prefix = "bluez_source."
        profile_suffixes = (
            ".handsfree_head_unit",
            ".headset_head_unit",
        )
        if not source.startswith(prefix):
            return None
        for suffix in profile_suffixes:
            if source.endswith(suffix):
                address = source[len(prefix):-len(suffix)]
                return f"bluez_card.{address}" if address else None
        return None

    def _start_pulseaudio(self) -> None:
        """Linux: capture from PulseAudio via a `parec` subprocess and feed the
        shared raw queue from a reader thread."""
        # Match microphone_test.py: Bluetooth headset sources are unavailable
        # in A2DP playback mode, so activate the microphone-capable profile
        # before starting parec.
        bluetooth_card = self._bluetooth_card_from_source(self._device)
        if bluetooth_card is not None:
            subprocess.run(
                ["pactl", "set-card-profile", bluetooth_card,
                 "handsfree_head_unit"],
                check=True,
            )
            print(f"[SpeechListener] Bluetooth hands-free profile: {bluetooth_card}")
        print(f"[SpeechListener] PulseAudio source: {self._device}")
        self._process = subprocess.Popen(
            ["parec", f"--device={self._device}", "--format=s16le",
             f"--rate={self.RATE}", f"--channels={self.CHANNELS}"],
            stdout=subprocess.PIPE,
        )
        threading.Thread(target=self._pulse_reader, daemon=True).start()

    def _pulse_reader(self) -> None:
        while self._running:
            raw = self._process.stdout.read(self.BYTES_PER_BLOCK)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            try:
                self._raw_queue.put_nowait(samples)
            except queue.Full:
                pass  # drop a block rather than stall capture

    def _start_sounddevice(self) -> None:
        """Windows (and any host with PortAudio): capture via sounddevice and
        feed the shared raw queue from the audio callback."""
        import sounddevice as sd

        dev = self._resolve_sd_index(self._device)

        def _callback(indata, frames, time_info, status):
            # indata is (frames, channels) int16; take mono channel 0.
            samples = indata[:, 0].astype(np.float32) / 32768.0
            try:
                self._raw_queue.put_nowait(samples.copy())
            except queue.Full:
                pass  # drop a block rather than block the audio thread

        self._sd_stream = sd.InputStream(
            samplerate=self.RATE,
            channels=self.CHANNELS,
            dtype="int16",
            blocksize=self.SAMPLES_PER_BLOCK,
            device=dev,
            callback=_callback,
        )
        self._sd_stream.start()

    @staticmethod
    def _resolve_sd_index(spec):
        """Resolve a sounddevice device from an int index, a name substring, or
        None (system default). Returns None if no input match is found, so voice
        degrades gracefully instead of crashing."""
        import sounddevice as sd
        if spec is None or (isinstance(spec, str) and spec.strip() == ""):
            return None
        try:
            return int(spec)
        except (TypeError, ValueError):
            pass
        spec_lower = spec.lower()
        for index, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and spec_lower in dev["name"].lower():
                return index
        return None

    # ── Background threads ────────────────────────────────────────────────────

    @staticmethod
    def _is_fatal_cuda_error(error: BaseException) -> bool:
        """Return whether CUDA cannot safely be reused in this process."""
        message = str(error).casefold()
        fatal_markers = (
            "illegal memory access",
            "illegal address",
            "device-side assert",
            "unspecified launch failure",
        )
        return "cuda" in message and any(
            marker in message for marker in fatal_markers)

    def _stop_after_fatal_asr_error(self) -> None:
        """Stop capture after CUDA context corruption; retries cannot recover."""
        self._running = False
        self._model_ready.clear()
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
            except Exception:
                pass
        if self._process is not None:
            try:
                self._process.terminate()
            except Exception:
                pass

    def _asr_worker(self) -> None:
        try:
            import nemo.collections.asr as nemo_asr
        except Exception as e:
            self._event_queue.put(("error", f"NeMo import failed: {e}"))
            return

        self._event_queue.put(("_status", self.STATUS_LOADING))
        try:
            import torch
            with GPU_INFERENCE_LOCK:
                model = nemo_asr.models.ASRModel.from_pretrained(self._model_name)
                # Place on GPU when available rather than trusting NeMo's default.
                if torch.cuda.is_available():
                    model = model.to("cuda")
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
                with GPU_INFERENCE_LOCK:
                    output = model.transcribe(
                        [utterance], batch_size=1, verbose=False)[0]
                text = (output.text if hasattr(output, "text") else str(output)).strip()
                if text:
                    self._event_queue.put(("transcript", text))
                else:
                    self._event_queue.put(("_status", self.STATUS_LISTENING
                                           if self.listening_active else self.STATUS_IDLE))
            except Exception as e:
                if self._is_fatal_cuda_error(e):
                    self._event_queue.put((
                        "error",
                        "Fatal CUDA ASR error. Voice input has stopped because "
                        "retrying a corrupted CUDA context cannot recover it. "
                        "Restart the application to restore Parakeet. "
                        f"Details: {e}",
                    ))
                    self._stop_after_fatal_asr_error()
                    break
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
            try:
                samples = self._raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self.input_enabled:
                pre_roll.clear()
                utt_blocks = []
                utt_samples = 0
                sil_samples = 0
                speech_active = False
                continue
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

        if not self.input_enabled:
            self.current_status = self.STATUS_DISABLED
            self.current_rms = 0.0
            while True:
                try:
                    kind, payload = self._event_queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "error":
                    notable.append(("error", str(payload)))
            return notable

        if (not self.always_listening and self.listening_active
                and now - self.last_speech_time > self._listen_timeout):
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
                self.current_status = (self.STATUS_LISTENING
                                       if self.always_listening
                                       and payload == self.STATUS_IDLE else payload)
                if payload == self.STATUS_IDLE:
                    notable.append(("ready", ""))
            elif kind == "transcript":
                text = str(payload)
                if self.always_listening:
                    self.last_speech_time = now
                    self.transcript_history.append(text)
                    self.current_status = self.STATUS_LISTENING
                    notable.append(("transcript", text))
                elif not self.listening_active:
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
        if self.always_listening or not self.listening_active:
            return 0.0
        return max(0.0, self._listen_timeout - (time.time() - self.last_speech_time))
