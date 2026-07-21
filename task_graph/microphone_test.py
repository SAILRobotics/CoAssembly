import queue
import subprocess
import threading
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

RATE = 16000
CHANNELS = 1
SAMPLES_PER_BLOCK = 1024
BYTES_PER_BLOCK = SAMPLES_PER_BLOCK * 2  # int16 = 2 bytes

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
DISPLAY_SECONDS = 1.0
SPEECH_RMS_THRESHOLD = 0.015
END_SILENCE_SECONDS = 0.8
MIN_UTTERANCE_SECONDS = 0.5
MAX_UTTERANCE_SECONDS = 15.0
PRE_ROLL_SECONDS = 0.3


def transcription_worker(audio_queue, result_queue):
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        result_queue.put(
            (
                "error",
                "NeMo ASR is not installed. Install 'nemo-toolkit[asr]' in "
                "the active environment.",
            )
        )
        return

    try:
        result_queue.put(("status", f"Loading {MODEL_NAME} ..."))
        model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        model.eval()
        result_queue.put(("status", "Parakeet ready; listening for speech."))

        while True:
            utterance = audio_queue.get()
            if utterance is None:
                break

            result_queue.put(("status", "Transcribing ..."))
            output = model.transcribe([utterance], batch_size=1)[0]
            transcript = output.text if hasattr(output, "text") else str(output)
            transcript = transcript.strip()

            if transcript:
                result_queue.put(("transcript", transcript))
            else:
                result_queue.put(("status", "No speech recognized; listening."))
    except Exception as error:
        result_queue.put(("error", f"Parakeet error: {error}"))


def enqueue_utterance(blocks, audio_queue):
    if not blocks:
        return

    utterance = np.concatenate(blocks).astype(np.float32, copy=False)
    if len(utterance) < int(MIN_UTTERANCE_SECONDS * RATE):
        return

    try:
        audio_queue.put_nowait(utterance)
    except queue.Full:
        print("Transcription is behind; dropping one utterance.")


command = [
    "parec",
    "--device=audiorelay-virtual-mic-sink.monitor",
    "--format=s16le",
    f"--rate={RATE}",
    f"--channels={CHANNELS}",
]

process = subprocess.Popen(command, stdout=subprocess.PIPE)

audio_queue = queue.Queue(maxsize=3)
result_queue = queue.Queue()
worker = threading.Thread(
    target=transcription_worker,
    args=(audio_queue, result_queue),
    daemon=True,
)
worker.start()

display_samples = int(RATE * DISPLAY_SECONDS)
waveform = np.zeros(display_samples, dtype=np.float32)
time_axis_ms = np.linspace(
    -DISPLAY_SECONDS * 1000,
    0,
    display_samples,
    endpoint=False,
)

plt.ion()
figure, axis = plt.subplots(figsize=(11, 5))
line, = axis.plot(time_axis_ms, waveform, linewidth=1)
level_text = axis.text(0.02, 0.92, "RMS: 0.0000", transform=axis.transAxes)
status_text = axis.text(0.02, 0.84, "Loading Parakeet ...", transform=axis.transAxes)
transcript_text = figure.text(
    0.5,
    0.02,
    "Transcript will appear here",
    horizontalalignment="center",
    wrap=True,
)
axis.set_title("AudioRelay microphone and NVIDIA Parakeet transcription")
axis.set_xlabel("Time (ms)")
axis.set_ylabel("Amplitude")
axis.set_xlim(time_axis_ms[0], time_axis_ms[-1])
axis.set_ylim(-1.0, 1.0)
axis.grid(True, alpha=0.3)
figure.subplots_adjust(bottom=0.15)
figure.show()

pre_roll_blocks = max(1, int(np.ceil(PRE_ROLL_SECONDS * RATE / SAMPLES_PER_BLOCK)))
pre_roll = deque(maxlen=pre_roll_blocks)
utterance_blocks = []
utterance_samples = 0
silence_samples = 0
speech_active = False

print("Receiving Android microphone audio and loading NVIDIA Parakeet.")
print("Close the waveform window or press Ctrl+C to stop.")

try:
    while plt.fignum_exists(figure.number):
        raw = process.stdout.read(BYTES_PER_BLOCK)

        if not raw:
            print("Audio stream ended.")
            break

        audio = np.frombuffer(raw, dtype=np.int16)
        samples = audio.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples**2)))

        sample_count = min(len(samples), display_samples)
        waveform = np.roll(waveform, -sample_count)
        waveform[-sample_count:] = samples[-sample_count:]
        line.set_ydata(waveform)
        level_text.set_text(f"RMS: {rms:.4f}")

        if rms >= SPEECH_RMS_THRESHOLD:
            if not speech_active:
                speech_active = True
                utterance_blocks = list(pre_roll)
                utterance_samples = sum(len(block) for block in utterance_blocks)
            utterance_blocks.append(samples.copy())
            utterance_samples += len(samples)
            silence_samples = 0
            status_text.set_text("Speech detected ...")
        elif speech_active:
            utterance_blocks.append(samples.copy())
            utterance_samples += len(samples)
            silence_samples += len(samples)

        pre_roll.append(samples.copy())

        utterance_is_finished = (
            speech_active and silence_samples >= int(END_SILENCE_SECONDS * RATE)
        )
        utterance_is_too_long = (
            speech_active and utterance_samples >= int(MAX_UTTERANCE_SECONDS * RATE)
        )
        if utterance_is_finished or utterance_is_too_long:
            enqueue_utterance(utterance_blocks, audio_queue)
            utterance_blocks = []
            utterance_samples = 0
            silence_samples = 0
            speech_active = False
            status_text.set_text("Utterance queued for transcription.")

        while True:
            try:
                result_type, message = result_queue.get_nowait()
            except queue.Empty:
                break

            if result_type == "transcript":
                print(f"Transcript: {message}")
                transcript_text.set_text(message)
                status_text.set_text("Listening for speech.")
            else:
                print(message)
                status_text.set_text(message)

        figure.canvas.draw_idle()
        figure.canvas.flush_events()

except KeyboardInterrupt:
    print("\nStopping.")

finally:
    if speech_active:
        enqueue_utterance(utterance_blocks, audio_queue)
    try:
        audio_queue.put_nowait(None)
    except queue.Full:
        pass
    process.terminate()
    process.wait()
    plt.close(figure)
