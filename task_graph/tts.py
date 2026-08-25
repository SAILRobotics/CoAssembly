#!/usr/bin/env python3
"""Interactive TTS tester — type a sentence, hear it spoken.

Alternates between two local TTS backends: Piper and NVIDIA NeMo
(FastPitch + HiFi-GAN). Each backend loads lazily on first use.

Run:
    python task_graph/tts.py
    python task_graph/tts.py --engine nemo
    python task_graph/tts.py --piper-model /path/to/voice.onnx --speech-rate 0.85

While running, type a sentence and press Enter to hear it. Commands:
    /piper          switch to the Piper backend
    /nemo           switch to the NeMo backend
    /quit or /exit  quit (Ctrl+D / Ctrl+C also work)

Piper requires a downloaded voice model (.onnx). If none is found at
--piper-model, an error explains how to fetch one.
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PIPER_MODEL = HERE / "tts_voices" / "en_US-lessac-medium.onnx"
NEMO_SPECTROGRAM_MODEL = "tts_en_fastpitch"
NEMO_VOCODER_MODEL = "tts_en_hifigan"


def play_wav(path: Path) -> None:
    subprocess.run(["paplay", str(path)], check=True)


class PiperBackend:
    name = "piper"

    def __init__(self, model_path: Path, speech_rate: float = 1.0):
        if float(speech_rate) <= 0.0:
            raise ValueError("speech_rate must be greater than zero")
        self.model_path = model_path
        self.speech_rate = float(speech_rate)
        self._voice = None

    def _load(self):
        if self._voice is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Piper voice model not found: {self.model_path}\n"
                "Install Piper and download a voice, e.g.:\n"
                "  pip install piper-tts\n"
                "  python -m piper.download_voices en_US-lessac-medium "
                f"--data-dir {self.model_path.parent}\n"
                "or download an .onnx voice manually from "
                "https://huggingface.co/rhasspy/piper-voices and pass "
                "--piper-model /path/to/voice.onnx"
            )
        from piper import PiperVoice

        print(f"Loading Piper voice: {self.model_path.name} ...")
        self._voice = PiperVoice.load(str(self.model_path))

    def synthesize(self, text: str, out_path: Path) -> None:
        self._load()
        try:
            from piper.config import SynthesisConfig
        except ImportError:
            # Compatibility with Piper releases that re-export the class.
            from piper import SynthesisConfig
        # Piper's length_scale is duration, while the public option is a
        # conventional rate: 0.85x => 1/0.85 times the normal duration.
        syn_config = SynthesisConfig(length_scale=1.0 / self.speech_rate)
        with wave.open(str(out_path), "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file, syn_config=syn_config)


class NemoBackend:
    name = "nemo"

    def __init__(self):
        self._fastpitch = None
        self._vocoder = None

    def _load(self):
        if self._fastpitch is not None:
            return
        import nemo.collections.tts as nemo_tts

        print(f"Loading {NEMO_SPECTROGRAM_MODEL} + {NEMO_VOCODER_MODEL} ...")
        self._fastpitch = nemo_tts.models.FastPitchModel.from_pretrained(
            NEMO_SPECTROGRAM_MODEL)
        self._vocoder = nemo_tts.models.HifiGanModel.from_pretrained(
            NEMO_VOCODER_MODEL)
        self._fastpitch.eval()
        self._vocoder.eval()

    def synthesize(self, text: str, out_path: Path) -> None:
        self._load()
        import numpy as np
        import torch

        with torch.no_grad():
            tokens = self._fastpitch.parse(text)
            spectrogram = self._fastpitch.generate_spectrogram(tokens=tokens)
            audio = self._vocoder.convert_spectrogram_to_audio(spec=spectrogram)

        samples = audio.squeeze().cpu().numpy().astype(np.float32)
        sample_rate = getattr(getattr(self._vocoder, "cfg", None),
                              "sample_rate", 22050)
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(out_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm16.tobytes())


class TTSService:
    """Non-blocking speech service for the task-graph application.

    Synthesis and playback run on one worker thread. ``replace=True`` removes
    stale queued guidance and interrupts current playback, which lets a safety
    warning take precedence over an older informational message.
    """

    def __init__(self, engine: str = "piper",
                 piper_model: Path = DEFAULT_PIPER_MODEL,
                 speech_rate: float = 1.0) -> None:
        self._backends = {
            "piper": PiperBackend(Path(piper_model), speech_rate=speech_rate),
            "nemo": NemoBackend(),
        }
        if engine not in self._backends:
            raise ValueError(f"Unknown TTS engine: {engine}")
        self.engine = engine
        self._jobs: "queue.Queue[str | None]" = queue.Queue(maxsize=8)
        self._events: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._player: subprocess.Popen | None = None
        self._player_lock = threading.Lock()
        self.is_speaking = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def speak(self, text: str, replace: bool = False) -> bool:
        text = " ".join(str(text).split())
        if not text:
            return False
        if replace:
            while True:
                try:
                    self._jobs.get_nowait()
                except queue.Empty:
                    break
            with self._player_lock:
                if self._player is not None and self._player.poll() is None:
                    self._player.terminate()
        try:
            self._jobs.put_nowait(text)
            return True
        except queue.Full:
            self._events.put(("error", "TTS queue is full; message skipped."))
            return False

    def poll(self) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def close(self) -> None:
        self._running = False
        with self._player_lock:
            if self._player is not None and self._player.poll() is None:
                self._player.terminate()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass

    def _loop(self) -> None:
        while self._running:
            try:
                text = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                break
            out_path: Path | None = None
            self.is_speaking = True
            self._events.put(("speaking", text))
            try:
                with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False) as handle:
                    out_path = Path(handle.name)
                self._backends[self.engine].synthesize(text, out_path)
                with self._player_lock:
                    self._player = subprocess.Popen(["paplay", str(out_path)])
                    player = self._player
                return_code = player.wait()
                if return_code not in (0, -15):
                    raise subprocess.CalledProcessError(
                        return_code, ["paplay", str(out_path)])
                self._events.put(("spoken", text))
            except Exception as error:
                self._events.put(("error", f"{self.engine}: {error}"))
            finally:
                with self._player_lock:
                    self._player = None
                if out_path is not None:
                    out_path.unlink(missing_ok=True)
                self.is_speaking = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["piper", "nemo"], default="piper",
                        help="TTS backend to start with (default: piper)")
    parser.add_argument("--piper-model", type=Path, default=DEFAULT_PIPER_MODEL,
                        help="Path to a Piper voice .onnx file")
    parser.add_argument(
        "--speech-rate", type=float, default=0.85,
        help="Piper speaking rate multiplier; below 1.0 is slower")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backends = {
        "piper": PiperBackend(args.piper_model, speech_rate=args.speech_rate),
        "nemo": NemoBackend(),
    }
    current = args.engine

    print("TTS tester — type a sentence and press Enter to hear it.")
    print("Commands: /piper  /nemo  /quit")
    print(f"Current engine: {current}\n")

    while True:
        try:
            text = input(f"[{current}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/piper":
            current = "piper"
            print(f"Switched to {current}.")
            continue
        if text == "/nemo":
            current = "nemo"
            print(f"Switched to {current}.")
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                out_path = Path(handle.name)
            backends[current].synthesize(text, out_path)
            play_wav(out_path)
        except Exception as error:
            print(f"[{current}] error: {error}", file=sys.stderr)
        finally:
            out_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
