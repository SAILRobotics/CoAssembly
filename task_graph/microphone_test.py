import queue
import subprocess
import threading
import time
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
END_SILENCE_SECONDS = 1.5
MIN_UTTERANCE_SECONDS = 0.5
MAX_UTTERANCE_SECONDS = 15.0
PRE_ROLL_SECONDS = 0.3

WAKE_WORD              = "hey robot"  # say this to start listening
LISTEN_TIMEOUT_SECONDS = 30.0         # seconds of silence before returning to idle
MAX_SHOWN_TRANSCRIPTS  = 8            # transcript lines kept in the log panel


def transcription_worker(audio_queue, result_queue, model_ready: threading.Event):
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
        model_ready.set()
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
    "--device=bluez_source.50_C2_ED_43_95_C8.handsfree_head_unit",
    "--format=s16le",
    f"--rate={RATE}",
    f"--channels={CHANNELS}",
]

process = subprocess.Popen(command, stdout=subprocess.PIPE)

audio_queue  = queue.Queue(maxsize=3)
result_queue = queue.Queue()
model_ready  = threading.Event()
worker = threading.Thread(
    target=transcription_worker,
    args=(audio_queue, result_queue, model_ready),
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
figure, (axis, ax_log) = plt.subplots(
    2, 1, figsize=(11, 8),
    gridspec_kw={"height_ratios": [3, 2]},
)
figure.subplots_adjust(hspace=0.35)

# ── Waveform axis ──────────────────────────────────────────────────────────
line, = axis.plot(time_axis_ms, waveform, linewidth=1)
level_text  = axis.text(0.02, 0.95, "RMS: 0.0000",        transform=axis.transAxes, va="top", fontsize=9)
status_text = axis.text(0.02, 0.87, "Loading Parakeet ...", transform=axis.transAxes, va="top", fontsize=9, color="yellow")
timer_text  = axis.text(0.98, 0.95, "",                    transform=axis.transAxes, va="top", ha="right", fontsize=9, color="cyan")
axis.set_title("Jabra Bluetooth microphone — NVIDIA Parakeet")
axis.set_xlabel("Time (ms)")
axis.set_ylabel("Amplitude")
axis.set_xlim(time_axis_ms[0], time_axis_ms[-1])
axis.set_ylim(-1.0, 1.0)
axis.grid(True, alpha=0.3)
axis.set_facecolor("#111111")

# ── Transcript log axis ────────────────────────────────────────────────────
ax_log.set_facecolor("#0a0a0a")
ax_log.set_xlim(0, 1)
ax_log.set_ylim(0, 1)
ax_log.axis("off")
ax_log.set_title("Transcript log", fontsize=9, loc="left")
log_text = ax_log.text(
    0.01, 0.97, f"Say \"{WAKE_WORD}\" to start...",
    transform=ax_log.transAxes,
    va="top", ha="left",
    fontsize=9, family="monospace",
    color="#cccccc",
    wrap=False,
)

figure.show()

pre_roll_blocks = max(1, int(np.ceil(PRE_ROLL_SECONDS * RATE / SAMPLES_PER_BLOCK)))
pre_roll = deque(maxlen=pre_roll_blocks)
utterance_blocks = []
utterance_samples = 0
silence_samples = 0
speech_active = False

listening_active   = False
last_speech_time   = 0.0
transcript_history = deque(maxlen=MAX_SHOWN_TRANSCRIPTS)

def _refresh_log():
    if transcript_history:
        log_text.set_text("\n".join(f"› {t}" for t in transcript_history))
    else:
        log_text.set_text(f'Say "{WAKE_WORD}" to start...')

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
            if model_ready.is_set():
                enqueue_utterance(utterance_blocks, audio_queue)
            utterance_blocks = []
            utterance_samples = 0
            silence_samples = 0
            speech_active = False
            status_text.set_text("Utterance queued for transcription.")

        # Timer display + idle timeout
        now = time.time()
        if listening_active:
            remaining = LISTEN_TIMEOUT_SECONDS - (now - last_speech_time)
            if remaining <= 0:
                listening_active = False
                timer_text.set_text("")
                status_text.set_text(f"Idle — say \"{WAKE_WORD}\" to start.")
                status_text.set_color("yellow")
            else:
                timer_text.set_text(f"timeout in {remaining:.0f}s")
        else:
            timer_text.set_text("")

        while True:
            try:
                result_type, message = result_queue.get_nowait()
            except queue.Empty:
                break

            if result_type == "transcript":
                if not listening_active:
                    if WAKE_WORD in message.lower():
                        listening_active = True
                        last_speech_time = time.time()
                        status_text.set_text("Listening ...")
                        status_text.set_color("lime")
                        print("[Wake word detected]")
                        _refresh_log()
                else:
                    last_speech_time = time.time()
                    print(f"Transcript: {message}")
                    transcript_history.append(message)
                    _refresh_log()
                    status_text.set_text("Listening for speech.")
            else:
                if result_type == "status" and "ready" in message:
                    status_text.set_text(f"Idle — say \"{WAKE_WORD}\" to start.")
                    status_text.set_color("yellow")
                    print(message)
                else:
                    print(message)
                    if not listening_active:
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
