"""voice_caption.py — Publish voice-assistant captions to the Quest (AR).

Python PUB connects to Unity, which binds the SUB (VoiceCaptionReceiver.cs),
matching the other Python→Unity channels. One JSON object per message:

    {"kind": "status", "status": "<state>", "level": 0..1}
    {"kind": "heard",  "text": "<transcript>"}
    {"kind": "reply",  "text": "<spoken text>", "awaiting_answer": bool}
    {"kind": "notice", "text": "<transient hint>"}

status values: loading, listening, hearing, transcribing, thinking, speaking,
muted, error. Status is resent on change, on meaningful mic-level change
(≤10 Hz), and every 2 s so a late-starting Unity app catches up.
"""

from __future__ import annotations

import json
import time


class VoiceCaptionPublisher:
    _LEVEL_INTERVAL_S = 0.1
    _HEARTBEAT_S      = 2.0

    def __init__(self, ip: str | None = None, port: int | None = None) -> None:
        import zmq
        if ip is None or port is None:
            import main_setting
            ip = ip or main_setting.UNITY_IP
            port = port or main_setting.VOICE_CAPTION_PORT
        self._pub = zmq.Context.instance().socket(zmq.PUB)
        self._pub.connect(f"tcp://{ip}:{port}")
        self._last_status: str | None = None
        self._last_level = 0.0
        self._last_status_t = 0.0
        print(f"[Caption] Publishing voice captions to tcp://{ip}:{port}")

    def _send(self, payload: dict) -> None:
        try:
            self._pub.send_string(json.dumps(payload))
        except Exception as error:
            print(f"[Caption] Publish error: {error}")

    def heard(self, text: str) -> None:
        self._send({"kind": "heard", "text": str(text)})

    def reply(self, text: str, awaiting_answer: bool | None = None) -> None:
        text = str(text)
        if awaiting_answer is None:
            awaiting_answer = text.rstrip().endswith("?")
        self._send({"kind": "reply", "text": text,
                    "awaiting_answer": bool(awaiting_answer)})

    def notice(self, text: str) -> None:
        self._send({"kind": "notice", "text": str(text)})

    def update_status(self, speech=None, tts=None, vlm=None) -> None:
        """Derive the single AR status from listener, TTS and VLM; call every frame."""
        level = 0.0
        if tts is not None and tts.is_speaking:
            status = "speaking"
        elif vlm is not None and getattr(vlm, "_current_status", "") == "thinking":
            status = "thinking"
        elif speech is None:
            status = "muted"
        else:
            level = min(float(speech.current_rms) * 10.0, 1.0)
            raw = speech.current_status
            if not getattr(speech, "input_enabled", True):
                status = "muted"
            elif raw == "speech":
                status = "hearing"
            elif raw in ("queued", "transcribing"):
                status = "transcribing"
            elif raw in ("loading", "error"):
                status = raw
            elif raw == "idle" and not speech.listening_active:
                status = "muted"   # waiting for a wake word
            else:
                status = "listening"

        now = time.monotonic()
        changed = status != self._last_status
        level_due = (abs(level - self._last_level) > 0.05
                     and now - self._last_status_t >= self._LEVEL_INTERVAL_S)
        heartbeat = now - self._last_status_t >= self._HEARTBEAT_S
        if not (changed or level_due or heartbeat):
            return
        self._last_status = status
        self._last_level = level
        self._last_status_t = now
        self._send({"kind": "status", "status": status, "level": round(level, 3)})

    def close(self) -> None:
        try:
            self._pub.close(linger=0)
        except Exception:
            pass
