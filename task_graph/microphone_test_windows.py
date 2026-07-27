"""microphone_test_windows.py — Windows port of microphone_test.py.

The original script captures audio on Linux via the PulseAudio `parec`
subprocess. On Windows there is no `parec`, so this version uses the
cross-platform `sounddevice` (PortAudio) library instead. Everything
downstream — VAD, waveform plot, transcript log, and the NeMo/Parakeet
transcription worker — is unchanged.

Usage:
    # See available input devices and exit:
    python task_graph/microphone_test_windows.py --list

    # Test the mic only (waveform + RMS, no model download):
    python task_graph/microphone_test_windows.py --device "USB" --no-asr

    # Full run (loads NVIDIA Parakeet; needs nemo-toolkit[asr] + torch):
    python task_graph/microphone_test_windows.py --device 1

    --device accepts either an integer index (from --list) or a case-
    insensitive name substring (e.g. "Jabra", "USB"). If omitted, the
    system default input device is used.
"""

import os

# Windows/conda ships two OpenMP runtimes (PyTorch's libiomp5md.dll and
# LLVM/MKL's libomp.dll). When both load into one process, OpenMP aborts with
# "OMP: Error #15". Allow the duplicate so the program can run. Must be set
# BEFORE numpy / torch / nemo import, so keep this at the very top.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import queue
import sys
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

RATE = 16000
CHANNELS = 1
SAMPLES_PER_BLOCK = 1024

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

# Default input device (name substring or index). The laptop's built-in mic
# array shows up as "Microphone Array on SoundWire Device (Cirrus Logic XU)".
# Override at runtime with --device; use --device "" to fall back to the
# system default input.
DEFAULT_DEVICE = "Microphone Array"


# ── Device selection ────────────────────────────────────────────────────────

def resolve_device(spec):
    """Return a sounddevice input-device index from an int, a name substring,
    or None (system default)."""
    if spec is None or (isinstance(spec, str) and spec.strip() == ""):
        return None

    # Integer index (either a real int or a numeric string).
    try:
        return int(spec)
    except (TypeError, ValueError):
        pass

    # Name substring — match the first input-capable device.
    spec_lower = spec.lower()
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and spec_lower in dev["name"].lower():
            return index
    raise SystemExit(
        f'No input device matching "{spec}". Run with --list to see options.'
    )


def print_devices():
    print(sd.query_devices())
    try:
        default_in, _ = sd.default.device
        if default_in is not None and default_in >= 0:
            print(f"\nDefault input device: [{default_in}] "
                  f"{sd.query_devices(default_in)['name']}")
    except Exception:
        pass


# ── Transcription worker (unchanged from the Linux version) ──────────────────

def transcription_worker(audio_queue, result_queue, model_ready: threading.Event):
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        result_queue.put(
            (
                "error",
                "NeMo ASR is not installed. Install 'nemo-toolkit[asr]' in "
                "the active environment, or run with --no-asr to test the mic only.",
            )
        )
        return

    try:
        import torch
        result_queue.put(("status", f"Loading {MODEL_NAME} ..."))
        model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        # Explicitly place the model on the GPU when available, rather than
        # relying on NeMo's default device selection.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        model_ready.set()
        result_queue.put(("status", f"Parakeet ready on {device.upper()}; listening for speech."))

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="List audio input devices and exit.")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help=f'Input device index or name substring (default: "{DEFAULT_DEVICE}"). '
                             'Pass an empty string to use the system default input.')
    parser.add_argument("--no-asr", action="store_true",
                        help="Skip NeMo/Parakeet; show waveform + RMS only.")
    args = parser.parse_args()

    if args.list:
        print_devices()
        return

    device = resolve_device(args.device)
    dev_name = sd.query_devices(device)["name"] if device is not None \
        else sd.query_devices(sd.default.device[0])["name"]
    print(f"Using input device: {dev_name}")

    # Raw int16 audio blocks flow from the PortAudio callback into this queue,
    # replacing the parec subprocess stdout pipe.
    raw_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        # indata is (frames, channels) int16; flatten to mono 1-D.
        try:
            raw_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # drop a block rather than block the audio thread

    audio_queue  = queue.Queue(maxsize=3)
    result_queue = queue.Queue()
    model_ready  = threading.Event()

    if not args.no_asr:
        worker = threading.Thread(
            target=transcription_worker,
            args=(audio_queue, result_queue, model_ready),
            daemon=True,
        )
        worker.start()

    display_samples = int(RATE * DISPLAY_SECONDS)
    waveform = np.zeros(display_samples, dtype=np.float32)
    time_axis_ms = np.linspace(
        -DISPLAY_SECONDS * 1000, 0, display_samples, endpoint=False,
    )

    plt.ion()
    figure, (axis, ax_log) = plt.subplots(
        2, 1, figsize=(11, 8),
        gridspec_kw={"height_ratios": [3, 2]},
    )
    figure.subplots_adjust(hspace=0.35)

    # ── Waveform axis ──────────────────────────────────────────────────────
    line, = axis.plot(time_axis_ms, waveform, linewidth=1)
    level_text  = axis.text(0.02, 0.95, "RMS: 0.0000",        transform=axis.transAxes, va="top", fontsize=9)
    initial_status = "Mic test — no ASR" if args.no_asr else "Loading Parakeet ..."
    status_text = axis.text(0.02, 0.87, initial_status, transform=axis.transAxes, va="top", fontsize=9, color="yellow")
    timer_text  = axis.text(0.98, 0.95, "",                    transform=axis.transAxes, va="top", ha="right", fontsize=9, color="cyan")
    axis.set_title(f"{dev_name} — NVIDIA Parakeet")
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Amplitude")
    axis.set_xlim(time_axis_ms[0], time_axis_ms[-1])
    axis.set_ylim(-1.0, 1.0)
    axis.grid(True, alpha=0.3)
    axis.set_facecolor("#111111")

    # ── Transcript log axis ────────────────────────────────────────────────
    ax_log.set_facecolor("#0a0a0a")
    ax_log.set_xlim(0, 1)
    ax_log.set_ylim(0, 1)
    ax_log.axis("off")
    ax_log.set_title("Transcript log", fontsize=9, loc="left")
    log_text = ax_log.text(
        0.01, 0.97, f'Say "{WAKE_WORD}" to start...',
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

    if args.no_asr:
        print("Mic-only test: speak and watch the waveform / RMS. Ctrl+C or close window to stop.")
    else:
        print("Capturing microphone audio and loading NVIDIA Parakeet.")
        print("Close the waveform window or press Ctrl+C to stop.")

    stream = sd.InputStream(
        samplerate=RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=SAMPLES_PER_BLOCK,
        device=device,
        callback=audio_callback,
    )

    try:
        with stream:
            while plt.fignum_exists(figure.number):
                # Drain EVERY audio block that has arrived, running VAD on each,
                # so audio processing keeps pace with real time regardless of how
                # long the (relatively slow) matplotlib redraw below takes. The
                # plot is refreshed once per outer iteration, not once per block,
                # which is what previously let the display drift behind live audio.
                blocks_this_tick = 0
                try:
                    samples_i16 = raw_queue.get(timeout=0.5)
                except queue.Empty:
                    samples_i16 = None

                while samples_i16 is not None:
                    blocks_this_tick += 1
                    samples = samples_i16.astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(samples**2)))

                    sample_count = min(len(samples), display_samples)
                    waveform = np.roll(waveform, -sample_count)
                    waveform[-sample_count:] = samples[-sample_count:]
                    level_text.set_text(f"RMS: {rms:.4f}")

                    if rms >= SPEECH_RMS_THRESHOLD:
                        if not speech_active:
                            speech_active = True
                            utterance_blocks = list(pre_roll)
                            utterance_samples = sum(len(block) for block in utterance_blocks)
                        utterance_blocks.append(samples.copy())
                        utterance_samples += len(samples)
                        silence_samples = 0
                        if not args.no_asr:
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
                        if not args.no_asr and model_ready.is_set():
                            enqueue_utterance(utterance_blocks, audio_queue)
                            status_text.set_text("Utterance queued for transcription.")
                        utterance_blocks = []
                        utterance_samples = 0
                        silence_samples = 0
                        speech_active = False

                    try:
                        samples_i16 = raw_queue.get_nowait()
                    except queue.Empty:
                        samples_i16 = None

                if blocks_this_tick:
                    line.set_ydata(waveform)

                # Timer display + idle timeout
                now = time.time()
                if listening_active:
                    remaining = LISTEN_TIMEOUT_SECONDS - (now - last_speech_time)
                    if remaining <= 0:
                        listening_active = False
                        timer_text.set_text("")
                        status_text.set_text(f'Idle — say "{WAKE_WORD}" to start.')
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
                            status_text.set_text(f'Idle — say "{WAKE_WORD}" to start.')
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
        if speech_active and not args.no_asr:
            enqueue_utterance(utterance_blocks, audio_queue)
        try:
            audio_queue.put_nowait(None)
        except queue.Full:
            pass
        plt.close(figure)


if __name__ == "__main__":
    main()
