"""Fixed factor levels and baselines for the VLM-unreliability audit harness.

Self-contained: this module has no third-party dependencies and is safe to
import without ``bpy`` or ``torch``.  It defines the canonical sweep axes used
by ``vlmunr_render.py`` together with a ``phase_levels`` helper that maps an
audit phase identifier to the factor values it varies.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Factor levels
# ---------------------------------------------------------------------------

RESOLUTIONS: List[int] = [224, 256, 384, 448, 512, 640, 768, 1024]
FOCAL_LENGTHS: List[int] = [24, 35, 50, 85, 100, 200]
BACKGROUND_GRAYS: List[int] = [0, 18, 65, 117, 128, 186, 204, 255]
HDRIS: List[str] = [
    "city",
    "courtyard",
    "forest",
    "interior",
    "night",
    "studio",
    "sunrise",
    "sunset",
]
PITCHES: List[int] = [0, 30, 60, 90]  # 0 == top-down in bpa convention
YAWS: List[int] = list(range(0, 360, 30))  # 0, 30, ..., 330

# ---------------------------------------------------------------------------
# Baseline configuration (the "default" point in factor space)
# ---------------------------------------------------------------------------

BASELINE_RES: int = 512
BASELINE_FOCAL: int = 50
BASELINE_BG: Tuple[int, int, int] = (128, 128, 128)
BASELINE_HDRI: str = "city"
BASELINE_PITCH: int = 0
BASELINE_YAW: int = 0


def gray(level: int) -> Tuple[int, int, int]:
    """Expand a single gray level into an ``(r, g, b)`` tuple."""

    return (level, level, level)


def phase_levels(phase: str) -> Dict[str, List]:
    """Return the factor values that the given audit phase sweeps.

    Each phase varies exactly one axis (or, for phase ``2``, the pitch x yaw
    cross-product) while holding every other factor at its baseline value.

    Returns a dict with the keys ``res``, ``focal``, ``bg`` (list of gray
    levels), ``hdri``, ``pitch`` and ``yaw``.  The varied axis contains all of
    its levels; the remaining axes contain a single baseline value.
    """

    base = {
        "res": [BASELINE_RES],
        "focal": [BASELINE_FOCAL],
        "bg": [BASELINE_BG[0]],
        "hdri": [BASELINE_HDRI],
        "pitch": [BASELINE_PITCH],
        "yaw": [BASELINE_YAW],
    }

    if phase == "1a":  # resolution sweep
        base["res"] = list(RESOLUTIONS)
    elif phase == "1b":  # background gray sweep
        base["bg"] = list(BACKGROUND_GRAYS)
    elif phase == "1c":  # HDRI sweep
        base["hdri"] = list(HDRIS)
    elif phase == "1d":  # focal length sweep
        base["focal"] = list(FOCAL_LENGTHS)
    elif phase == "2":  # camera pose sweep (pitch x yaw)
        base["pitch"] = list(PITCHES)
        base["yaw"] = list(YAWS)
    else:
        raise ValueError(f"Unknown phase: {phase!r}")

    return base


PHASES: List[str] = ["1a", "1b", "1c", "1d", "2"]
