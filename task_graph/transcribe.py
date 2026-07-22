"""
transcribe.py — Live speech-to-text from Jabra Bluetooth mic via Parakeet ASR.
Prints transcripts to terminal; no GUI required.

Usage:
    python transcribe.py
"""

import queue
import subprocess
import threading
from collections import deque

import numpy as np

RATE                 = 16000
CHANNELS             = 1
SAMPLES_PER_BLOCK    = 1024
BYTES_PER_BLOCK      = SAMPLES_PER_BLOCK * 2   # int16 = 2 bytes

MODEL_NAME           = "nvidia/parakeet-tdt-0.6b-v2"
SPEECH_RMS_THRESHOLD = 0.015
END_SILENCE_SECONDS  = 0.8
MIN_UTTERANCE_SECONDS = 0.5
MAX_UTTERANCE_SECONDS = 15.0
PRE_ROLL_SECONDS     = 0.3

JABRA_SOURCE = "bluez_source.50_C2_ED_43_95_C8.handsfree_head_unit"


def transcription_worker(audio_queue: queue.Queue, result_queue: queue.Queue):
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        result_queue.put(("error", "NeMo not installed. Run: pip install nemo-toolkit[asr]"))
        return

    result_queue.put(("status", f"Loading {MODEL_NAME} ..."))
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        model.eval()
    except Exception as e:
        result_queue.put(("error", f"Model load failed: {e}"))
        return

    result_queue.put(("status", "Parakeet ready — listening."))

    while True:
        utterance = audio_queue.get()
        if utterance is None:
            break
        result_queue.put(("status", "Transcribing ..."))
        try:
            output = model.transcribe([utterance], batch_size=1)[0]
            text = (output.text if hasattr(output, "text") else str(output)).strip()
            if text:
                result_queue.put(("transcript", text))
            else:
                result_queue.put(("status", "No speech — listening."))
        except Exception as e:
            result_queue.put(("error", f"Transcription error: {e}"))


def enqueue_utterance(blocks, audio_queue: queue.Queue):
    if not blocks:
        return
    utterance = np.concatenate(blocks).astype(np.float32, copy=False)
    if len(utterance) < int(MIN_UTTERANCE_SECONDS * RATE):
        return
    try:
        audio_queue.put_nowait(utterance)
    except queue.Full:
        print("[warn] Transcription behind — dropping utterance.")


def main():
    process = subprocess.Popen(
        ["parec", f"--device={JABRA_SOURCE}", "--format=s16le",
         f"--rate={RATE}", f"--channels={CHANNELS}"],
        stdout=subprocess.PIPE,
    )

    audio_queue  = queue.Queue(maxsize=3)
    result_queue = queue.Queue()
    worker = threading.Thread(target=transcription_worker,
                              args=(audio_queue, result_queue), daemon=True)
    worker.start()

    pre_roll_blocks = max(1, int(np.ceil(PRE_ROLL_SECONDS * RATE / SAMPLES_PER_BLOCK)))
    pre_roll        = deque(maxlen=pre_roll_blocks)
    utterance_blocks  = []
    utterance_samples = 0
    silence_samples   = 0
    speech_active     = False

    print(f"Listening on Jabra ({JABRA_SOURCE}). Press Ctrl+C to stop.\n")

    try:
        while True:
            raw = process.stdout.read(BYTES_PER_BLOCK)
            if not raw:
                print("Audio stream ended.")
                break

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))

            if rms >= SPEECH_RMS_THRESHOLD:
                if not speech_active:
                    speech_active    = True
                    utterance_blocks = list(pre_roll)
                    utterance_samples = sum(len(b) for b in utterance_blocks)
                    print(f"[speech]  rms={rms:.4f}")
                utterance_blocks.append(samples.copy())
                utterance_samples += len(samples)
                silence_samples = 0
            elif speech_active:
                utterance_blocks.append(samples.copy())
                utterance_samples += len(samples)
                silence_samples   += len(samples)

            pre_roll.append(samples.copy())

            finished  = speech_active and silence_samples   >= int(END_SILENCE_SECONDS  * RATE)
            too_long  = speech_active and utterance_samples >= int(MAX_UTTERANCE_SECONDS * RATE)
            if finished or too_long:
                enqueue_utterance(utterance_blocks, audio_queue)
                utterance_blocks  = []
                utterance_samples = 0
                silence_samples   = 0
                speech_active     = False

            while True:
                try:
                    kind, msg = result_queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "transcript":
                    print(f"\n>>> {msg}\n")
                else:
                    print(f"[{kind}] {msg}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if speech_active:
            enqueue_utterance(utterance_blocks, audio_queue)
        try:
            audio_queue.put_nowait(None)
        except queue.Full:
            pass
        process.terminate()
        process.wait()


if __name__ == "__main__":
    main()
