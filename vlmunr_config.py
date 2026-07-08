"""Fixed factor levels and render-spec builders for the VLM-unreliability harness.

Self-contained: no third-party deps, safe to import without ``bpy`` or
``torch``.  Defines the canonical sweep axes, the baseline render point, the
single-image ``--render`` spec, and the six single-axis ``--render-all`` sweeps.

Filename convention (see ``vlmunr_render.master_filename`` /
``composite_filename``)::

    render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>.png
    render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>_bg-<r>-<g>-<b>.png

The transparent master (first form) is rendered once per
``(res, focal, pitch, yaw, env)``; each background color is composited onto
it to produce the second form.  ``fit_ratio=1`` (tight-fit) is used
throughout.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Factor levels
# ---------------------------------------------------------------------------

RESOLUTIONS: List[int] = [196, 224, 256, 336, 384, 448, 512, 768, 1024]
FOCAL_LENGTHS: List[int] = [16, 24, 35, 50, 85, 100, 200]

# Grayscale backgrounds (as equal-channel (r,g,b) tuples), ascending.
BACKGROUND_GRAYS: List[Tuple[int, int, int]] = [
    (0, 0, 0),
    (65, 65, 65),
    (118, 118, 118),
    (128, 128, 128),
    (186, 186, 186),
    (204, 204, 204),
    (255, 255, 255),
]

# Saturated chromatic backgrounds (pure R / G / B), composited like the grays.
BACKGROUND_CHROMATIC: List[Tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]

# The full background-color sweep: grays ascending, then chromatic.  Used by
# the ``--render-all`` background sweep (sweep #6).
BACKGROUNDS: List[Tuple[int, int, int]] = BACKGROUND_GRAYS + BACKGROUND_CHROMATIC

# Sentinel for a floor-texture background condition.  Documented for paper
# parity (Table 1) but NOT rendered by this harness -- there is no in-repo
# floor-texture compositing path, so it is intentionally excluded from the
# composited background sweeps below.
FLOOR_TEXTURE_BACKGROUND: str = "floor_texture"

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
PITCHES: List[int] = [0, 15, 30, 45, 60, 75, 90]  # 0 == top-down in bpa convention
YAWS: List[int] = [0, 45, 90, 135, 180, 225, 270, 315]

# ---------------------------------------------------------------------------
# Baseline configuration (the "default" point in factor space)
# ---------------------------------------------------------------------------
#
# The single-axis ``--render-all`` sweeps fix every non-swept factor at the
# baseline.  The fixed background for all sweeps is pure white (255,255,255);
# the only sweep that varies the background is sweep #6.

BASELINE_RES: int = 512
BASELINE_FOCAL: int = 50
BASELINE_BG: Tuple[int, int, int] = (255, 255, 255)
BASELINE_HDRI: str = "city"
BASELINE_PITCH: int = 0
BASELINE_YAW: int = 0
# Fixed pitch used when sweeping yaw (sweep #4 uses pitch=45, NOT baseline 0).
YAW_SWEEP_PITCH: int = 45

# Backwards-compatible alias retained from the prior harness spelling.
BASELINE_YAW_PITCH: int = YAW_SWEEP_PITCH


def gray(level: int) -> Tuple[int, int, int]:
    """Expand a single gray level into an ``(r, g, b)`` tuple."""

    return (level, level, level)


# ---------------------------------------------------------------------------
# Render specs
# ---------------------------------------------------------------------------

# A render spec is a dict with ``res, focal, pitch, yaw, env`` and a list of
# background ``(r, g, b)`` tuples ``bgs``.  One spec == one transparent master
# render plus its composites.  Specs use ``env`` (not ``hdri``) as the key
# name so the on-disk filename ``..._env-<name>.png`` maps 1:1 to the spec.
Spec = Dict


def _spec(
    res: int,
    focal: int,
    pitch: int,
    yaw: int,
    env: str,
    bgs: List[Tuple[int, int, int]],
) -> Spec:
    return {
        "res": res,
        "focal": focal,
        "pitch": pitch,
        "yaw": yaw,
        "env": env,
        "hdri": env,  # alias kept for callers that index by "hdri"
        "bgs": list(bgs),
    }


def single_render_spec() -> Spec:
    """The single ``--render`` image: 512 / 50mm / pitch 0 / yaw 0 / city.

    Renders the transparent master, then composites the white (255,255,255)
    background onto it.  Both files are saved (master + composite).
    """

    return _spec(
        BASELINE_RES,
        BASELINE_FOCAL,
        BASELINE_PITCH,
        BASELINE_YAW,
        BASELINE_HDRI,
        [BASELINE_BG],
    )


def all_sweep_specs() -> List[Spec]:
    """The six ``--render-all`` single-axis sweeps.

    Each sweep varies exactly one axis while holding every other factor at the
    baseline (white bg, 50mm, pitch 0, yaw 0, city) -- except the yaw sweep,
    which holds pitch at 45 so the oblique views look *into* the room rather
    than straight down.

    1. resolution sweep (bg white, 50mm, pitch 0, yaw 0, city)
    2. focal-length sweep (512, white, pitch 0, yaw 0, city)
    3. pitch sweep (512, white, 50mm, yaw 0, city)
    4. yaw sweep (512, white, 50mm, pitch 45, city)
    5. env-map sweep (512, white, 50mm, pitch 0, yaw 0)
    6. background sweep (512, 50mm, pitch 0, yaw 0, city)

    Returned un-merged: a master shared across sweeps (e.g. the baseline
    point 512/50/0/0/city) will appear in several specs.  The renderer merges
    these per master before rendering so each shared master is rendered once
    and all its backgrounds composited onto it.
    """

    white = BASELINE_BG
    specs: List[Spec] = []

    # 1. resolution
    for res in RESOLUTIONS:
        specs.append(_spec(res, BASELINE_FOCAL, BASELINE_PITCH, BASELINE_YAW, BASELINE_HDRI, [white]))
    # 2. focal length
    for focal in FOCAL_LENGTHS:
        specs.append(_spec(BASELINE_RES, focal, BASELINE_PITCH, BASELINE_YAW, BASELINE_HDRI, [white]))
    # 3. pitch
    for pitch in PITCHES:
        specs.append(_spec(BASELINE_RES, BASELINE_FOCAL, pitch, BASELINE_YAW, BASELINE_HDRI, [white]))
    # 4. yaw (pitch fixed at 45)
    for yaw in YAWS:
        specs.append(_spec(BASELINE_RES, BASELINE_FOCAL, YAW_SWEEP_PITCH, yaw, BASELINE_HDRI, [white]))
    # 5. env map
    for env in HDRIS:
        specs.append(_spec(BASELINE_RES, BASELINE_FOCAL, BASELINE_PITCH, BASELINE_YAW, env, [white]))
    # 6. background color
    specs.append(_spec(BASELINE_RES, BASELINE_FOCAL, BASELINE_PITCH, BASELINE_YAW, BASELINE_HDRI, BACKGROUNDS))

    return specs


# ---------------------------------------------------------------------------
# Legacy phase API (building blocks; kept for the vlmunr_render module CLI and
# for tests that exercise one axis at a time)
# ---------------------------------------------------------------------------


def phase_levels(phase: str) -> Dict[str, List]:
    """Return the factor values that the given audit phase sweeps.

    Each phase varies exactly one axis (or, for phase ``2``, the pitch x yaw
    cross-product) while holding every other factor at its baseline value.

    Returns a dict with the keys ``res``, ``focal``, ``bg`` (list of
    ``(r, g, b)`` tuples), ``hdri``, ``pitch`` and ``yaw``.  The varied axis
    contains all of its levels; the remaining axes contain a single baseline
    value.
    """

    base = {
        "res": [BASELINE_RES],
        "focal": [BASELINE_FOCAL],
        "bg": [BASELINE_BG],
        "hdri": [BASELINE_HDRI],
        "pitch": [BASELINE_PITCH],
        "yaw": [BASELINE_YAW],
    }

    if phase == "1a":  # resolution sweep
        base["res"] = list(RESOLUTIONS)
    elif phase == "1b":  # grayscale background sweep
        base["bg"] = list(BACKGROUND_GRAYS)
    elif phase == "1b_chroma":  # chromatic background sweep
        base["bg"] = list(BACKGROUND_CHROMATIC)
    elif phase == "1c":  # HDRI sweep
        base["hdri"] = list(HDRIS)
    elif phase == "1d":  # focal length sweep
        base["focal"] = list(FOCAL_LENGTHS)
    elif phase == "2":  # camera pose sweep (pitch x yaw)
        base["pitch"] = list(PITCHES)
        base["yaw"] = list(YAWS)
    elif phase == "2_pitch":  # pitch sweep at baseline yaw (0)
        base["pitch"] = list(PITCHES)
        base["yaw"] = [BASELINE_YAW]
    elif phase == "2_yaw":  # yaw sweep at fixed pitch (45)
        base["pitch"] = [YAW_SWEEP_PITCH]
        base["yaw"] = list(YAWS)
    else:
        raise ValueError(f"Unknown phase: {phase!r}")

    return base


PHASES: List[str] = ["1a", "1b", "1b_chroma", "1c", "1d", "2", "2_pitch", "2_yaw"]

# Backwards-compatible alias for the full phase list.
ALL_PHASES: List[str] = list(PHASES)
