"""Process-wide serialization for GPU-backed speech and vision inference.

The DearPyGui application runs Parakeet ASR and Qwen-VL in separate worker
threads.  Both libraries use the same CUDA context, and overlapping inference
has produced intermittent illegal-memory-access failures on this workstation.
Keeping model load and inference calls mutually exclusive avoids that overlap
without changing either model's public behavior.
"""

from __future__ import annotations

import threading


GPU_INFERENCE_LOCK = threading.RLock()
