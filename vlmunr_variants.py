"""Content-variant generation for the VLM-unreliability audit harness.

Given a scene directory containing a LayoutVLM input task JSON and its
``layout.json``, this writes sibling variant directories, each holding a
modified ``layout.json`` (plus a copy of the task JSON):

  * Removal variants -- a seeded random subset of instances is *kept*:
      - ``variant_half``    keeps round(n / 2)  (>= 1)
      - ``variant_quarter`` keeps round(n / 4)  (>= 1)
      - ``variant_eighth``  keeps round(n / 8)  (>= 1)

  * Worst-match variants -- a structured "retrieval" hook substitutes one or
    more assets with a low-ranked (poorly matching) library candidate:
      - ``variant_alt_0`` / ``variant_alt_2`` / ``variant_alt_4``
        (the suffix is the substitution rank offset; 0 = worst match).

  * Category-aware substitution variants -- worst-match constrained by
    category:
      - ``variant_subst_within`` (same category, different asset)
      - ``variant_subst_cross``  (different category)

  * Layout-scramble variant -- every instance is relocated to a random point
    within the floor polygon, preserving the instance set, rotation and z:
      - ``variant_scramble``

The removal + scramble logic is implemented as pure, importable functions and
is fully unit-tested.  The worst-match / category-substitution hooks lazily
import ``torch``/``transformers`` *inside* the function and degrade gracefully
(recording intent in the variant metadata, leaving the scene unchanged) when
models or an asset library are unavailable, so the pipeline never crashes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

REMOVAL_VARIANTS: List[Tuple[str, int]] = [
    ("variant_half", 2),
    ("variant_quarter", 4),
    ("variant_eighth", 8),
]

# (dir name, rank offset) -- 0 == worst match.
ALT_VARIANTS: List[Tuple[str, int]] = [
    ("variant_alt_0", 0),
    ("variant_alt_2", 2),
    ("variant_alt_4", 4),
]

# Category-aware worst-match substitution modes (dir name, mode).
#   * ``within`` -- swap toward a *different* asset of the *same* category.
#   * ``cross``  -- swap toward an asset of a *different* category.
SUBST_VARIANTS: List[Tuple[str, str]] = [
    ("variant_subst_within", "within"),
    ("variant_subst_cross", "cross"),
]

# Layout-scramble content variant (relocates every instance within the floor).
SCRAMBLE_VARIANT: str = "variant_scramble"


# ===========================================================================
# Pure removal logic (unit tested)
# ===========================================================================


def kept_count(n: int, k: int) -> int:
    """Number of instances to keep when downsampling ``n`` by factor ``k``.

    Uses banker's-rounding-free ``round(n / k)`` clamped to ``>= 1`` (when
    ``n >= 1``).  Returns 0 only for an empty scene.
    """

    if n <= 0:
        return 0
    return max(1, round(n / k))


def select_kept_ids(
    instance_ids: List[str], k: int, seed: int
) -> List[str]:
    """Deterministically choose which instance ids to keep.

    Uses ``random.Random(seed).sample`` over a *sorted* copy of the ids so the
    result is stable regardless of input ordering.  The returned list preserves
    the original ``instance_ids`` ordering for the chosen subset.
    """

    n = len(instance_ids)
    keep_n = kept_count(n, k)
    if keep_n >= n:
        return list(instance_ids)
    rng = random.Random(seed)
    chosen = set(rng.sample(sorted(instance_ids), keep_n))
    return [i for i in instance_ids if i in chosen]


def make_removal_layout(
    layout: Dict, k: int, seed: int
) -> Dict:
    """Return a new layout dict containing only the kept instances."""

    kept = select_kept_ids(list(layout.keys()), k, seed)
    return {i: layout[i] for i in kept}


# ===========================================================================
# Layout-scramble logic (unit tested)
# ===========================================================================


def floor_bbox(
    floor_vertices: List[List[float]],
) -> Tuple[float, float, float, float]:
    """Axis-aligned ``(min_x, min_y, max_x, max_y)`` of the floor polygon.

    ``floor_vertices`` are CCW ``[x, y, z]`` points with ``z == 0`` (world up
    is z).  Degenerate / empty inputs collapse to a zero-area box at the
    origin.
    """

    if not floor_vertices:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [float(v[0]) for v in floor_vertices]
    ys = [float(v[1]) for v in floor_vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _point_in_polygon(x: float, y: float, poly_xy: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (boundary counts as inside-ish)."""

    n = len(poly_xy)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly_xy[i]
        xj, yj = poly_xy[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def scramble_layout(
    layout: Dict,
    floor_vertices: List[List[float]],
    seed: int,
) -> Dict:
    """Relocate every instance to a random ``(x, y)`` within the floor.

    Pure + deterministic for a fixed ``seed``.  Each instance is moved to a
    random point inside the floor polygon's bounding box (and, when a valid
    polygon is available, rejection-sampled to land *inside* the polygon).
    The instance set, ids, count, ``rotation`` and ``z`` are all preserved
    (floor objects therefore keep their ``z = bbox.z / 2`` placement, which is
    encoded in the original layout's z and left untouched here).

    Instances are processed in sorted-id order so the result is independent of
    the input dict ordering.
    """

    min_x, min_y, max_x, max_y = floor_bbox(floor_vertices)
    poly_xy = [(float(v[0]), float(v[1])) for v in floor_vertices]
    use_polygon = len(poly_xy) >= 3 and max_x > min_x and max_y > min_y

    rng = random.Random(seed)
    new_layout: Dict = {}
    for inst_id in sorted(layout.keys()):
        place = layout[inst_id]
        pos = list(place.get("position", [0.0, 0.0, 0.0]))
        while len(pos) < 3:
            pos.append(0.0)
        z = float(pos[2])

        new_x = rng.uniform(min_x, max_x)
        new_y = rng.uniform(min_y, max_y)
        if use_polygon:
            # Rejection-sample a handful of times to land inside the polygon;
            # fall back to the bbox sample if the polygon is awkward.
            for _ in range(32):
                if _point_in_polygon(new_x, new_y, poly_xy):
                    break
                new_x = rng.uniform(min_x, max_x)
                new_y = rng.uniform(min_y, max_y)

        new_place = dict(place)
        new_place["position"] = [new_x, new_y, z]
        new_layout[inst_id] = new_place

    return new_layout


# ===========================================================================
# Worst-match (structured retrieval) hook
# ===========================================================================


def _text_for_asset(asset: Dict) -> str:
    cat = asset.get("category", "") or ""
    desc = asset.get("description", "") or ""
    return f"{cat}. {desc}".strip()


def score_candidates_worst_match(
    query_text: str,
    candidate_texts: List[str],
    rank_offset: int = 0,
) -> Optional[int]:
    """Pick a *poorly* matching candidate index by text similarity.

    When ``VLMUNR_USE_EMBED_MODEL`` is truthy, lazily imports
    ``torch``/``transformers`` and embeds the query + candidate texts, ranking
    candidates by ascending similarity (worst first).  Returns the index of the
    candidate at ``rank_offset`` in that ascending ranking, or ``None`` if
    there are no candidates (caller then degrades gracefully).

    By default (and whenever the ML stack cannot be loaded) it uses a
    pure-Python lexical (token-overlap) ranking, so the structure is exercised
    fully offline with no model download.
    """

    if not candidate_texts:
        return None

    sims: Optional[List[float]] = None
    # The embedding path is opt-in (it may download a model on first use).
    # Default to the offline lexical fallback so the harness never reaches out
    # to the network unless explicitly asked via VLMUNR_USE_EMBED_MODEL=1.
    if os.environ.get("VLMUNR_USE_EMBED_MODEL", "").lower() in ("1", "true", "yes"):
        try:  # ML path -- only attempted when opted in, never required.
            import torch  # noqa: F401  (lazy, optional)
            from transformers import AutoModel, AutoTokenizer

            model_name = os.environ.get(
                "VLMUNR_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            tok = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()

            def embed(texts: List[str]):
                enc = tok(
                    texts, padding=True, truncation=True, return_tensors="pt"
                )
                with torch.no_grad():
                    out = model(**enc)
                # Mean-pool token embeddings.
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (out.last_hidden_state * mask).sum(1)
                counts = mask.sum(1).clamp(min=1e-9)
                return torch.nn.functional.normalize(summed / counts, dim=-1)

            q = embed([query_text])
            c = embed(candidate_texts)
            sims = (c @ q.T).squeeze(-1).tolist()
        except Exception:
            sims = None

    if sims is None:
        # Lexical fallback: token-overlap similarity.
        q_tokens = set(query_text.lower().split())
        sims = []
        for t in candidate_texts:
            t_tokens = set(t.lower().split())
            if not q_tokens and not t_tokens:
                sims.append(0.0)
            else:
                inter = len(q_tokens & t_tokens)
                union = len(q_tokens | t_tokens) or 1
                sims.append(inter / union)

    # Ascending order -> worst match first.
    order = sorted(range(len(sims)), key=lambda i: sims[i])
    idx = min(rank_offset, len(order) - 1)
    return order[idx]


def make_worst_match_layout(
    task: Dict,
    layout: Dict,
    rank_offset: int,
) -> Tuple[Dict, Dict]:
    """Attempt a structured worst-match substitution.

    Returns ``(new_layout, intent)`` where ``new_layout`` mirrors ``layout``
    (placements are preserved -- only asset identity would change, which lives
    in the task asset map, not the layout) and ``intent`` records what the hook
    decided.

    Because this in-repo codebase ships *no* asset library / retrieval index,
    there are no candidate assets to substitute toward, so the hook records its
    intent and leaves the scene unchanged (graceful degradation).  When a
    library is wired in via ``VLMUNR_ASSET_LIBRARY`` (a JSON list of
    ``{"category","description","path"}`` records) the hook scores candidates
    and records the chosen substitutions.
    """

    intent: Dict = {
        "kind": "worst_match",
        "rank_offset": rank_offset,
        "substitutions": [],
        "degraded": False,
        "reason": "",
    }

    lib_path = os.environ.get("VLMUNR_ASSET_LIBRARY")
    candidates: List[Dict] = []
    if lib_path and os.path.exists(lib_path):
        try:
            with open(lib_path) as f:
                candidates = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            intent["degraded"] = True
            intent["reason"] = f"failed to read library: {exc}"
            return dict(layout), intent

    if not candidates:
        intent["degraded"] = True
        intent["reason"] = (
            "no asset library available (set VLMUNR_ASSET_LIBRARY to enable "
            "structured worst-match substitution); scene left unchanged"
        )
        return dict(layout), intent

    cand_texts = [_text_for_asset(c) for c in candidates]
    assets = task.get("assets", {})
    for inst_id in layout:
        asset = assets.get(inst_id)
        if asset is None:
            continue
        query = _text_for_asset(asset)
        pick = score_candidates_worst_match(query, cand_texts, rank_offset)
        if pick is None:
            continue
        intent["substitutions"].append(
            {
                "instance_id": inst_id,
                "from": query,
                "to": cand_texts[pick],
                "candidate_path": candidates[pick].get("path"),
            }
        )

    return dict(layout), intent


# ===========================================================================
# Category-aware substitution (within- vs cross-category)
# ===========================================================================


def make_category_subst_layout(
    task: Dict,
    layout: Dict,
    mode: str,
) -> Tuple[Dict, Dict]:
    """Attempt a category-aware worst-match substitution.

    ``mode`` is either ``"within"`` (swap toward a different asset of the
    *same* category) or ``"cross"`` (swap toward an asset of a *different*
    category).  Returns ``(new_layout, intent)``; placements are preserved
    (asset identity lives in the task asset map) so ``new_layout`` mirrors
    ``layout``.

    The intent's ``substitutions`` is a list of ``{instance_id: mode}`` records
    capturing what the hook decided per instance.  Like the rank-offset hook,
    candidate assets come from a JSON library at ``VLMUNR_ASSET_LIBRARY``
    (records of ``{"category","description","path"}``); when no library is
    present the hook records intent for every instance and degrades gracefully,
    leaving the scene unchanged.
    """

    if mode not in ("within", "cross"):
        raise ValueError(f"Unknown subst mode: {mode!r}")

    intent: Dict = {
        "kind": "category_subst",
        "mode": mode,
        "substitutions": [],
        "degraded": False,
        "reason": "",
    }

    assets = task.get("assets", {})

    lib_path = os.environ.get("VLMUNR_ASSET_LIBRARY")
    candidates: List[Dict] = []
    if lib_path and os.path.exists(lib_path):
        try:
            with open(lib_path) as f:
                candidates = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            intent["degraded"] = True
            intent["reason"] = f"failed to read library: {exc}"
            # Still record intent per instance so the probe is auditable.
            for inst_id in layout:
                if inst_id in assets:
                    intent["substitutions"].append({inst_id: mode})
            return dict(layout), intent

    if not candidates:
        intent["degraded"] = True
        intent["reason"] = (
            "no asset library available (set VLMUNR_ASSET_LIBRARY to enable "
            f"category {mode}-substitution); intent recorded, scene unchanged"
        )
        for inst_id in layout:
            if inst_id in assets:
                intent["substitutions"].append({inst_id: mode})
        return dict(layout), intent

    # Library present: score candidates per instance, filtering by category.
    for inst_id in layout:
        asset = assets.get(inst_id)
        if asset is None:
            continue
        src_cat = (asset.get("category", "") or "").strip().lower()
        if mode == "within":
            pool = [
                (i, c)
                for i, c in enumerate(candidates)
                if (c.get("category", "") or "").strip().lower() == src_cat
                and _text_for_asset(c) != _text_for_asset(asset)
            ]
        else:  # cross
            pool = [
                (i, c)
                for i, c in enumerate(candidates)
                if (c.get("category", "") or "").strip().lower() != src_cat
            ]
        if not pool:
            # No eligible candidate for this mode -- record intent only.
            intent["substitutions"].append({inst_id: mode})
            continue
        pool_texts = [_text_for_asset(c) for _, c in pool]
        pick = score_candidates_worst_match(_text_for_asset(asset), pool_texts, 0)
        if pick is None:
            intent["substitutions"].append({inst_id: mode})
            continue
        chosen_idx, chosen = pool[pick]
        intent["substitutions"].append(
            {
                "instance_id": inst_id,
                "mode": mode,
                "from": _text_for_asset(asset),
                "to": pool_texts[pick],
                "candidate_path": chosen.get("path"),
            }
        )

    return dict(layout), intent


def _resolve_inputs(scene_dir: str) -> Tuple[str, str]:
    layout_path = os.path.join(scene_dir, "layout.json")
    if not os.path.exists(layout_path):
        raise FileNotFoundError(f"layout.json not found in {scene_dir}")
    task_candidates = [
        f
        for f in sorted(os.listdir(scene_dir))
        if f.endswith(".json")
        and f != "layout.json"
        and not f.startswith("variant_")
    ]
    if not task_candidates:
        raise FileNotFoundError(f"No task JSON found in {scene_dir}")
    task_name = "task.json" if "task.json" in task_candidates else task_candidates[0]
    return os.path.join(scene_dir, task_name), task_name


def _write_variant(
    parent: str,
    variant_dir: str,
    task: Dict,
    task_name: str,
    new_layout: Dict,
    intent: Optional[Dict] = None,
) -> str:
    out = os.path.join(parent, variant_dir)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "layout.json"), "w") as f:
        json.dump(new_layout, f, indent=2)
    with open(os.path.join(out, task_name), "w") as f:
        json.dump(task, f, indent=2)
    if intent is not None:
        with open(os.path.join(out, "variant_intent.json"), "w") as f:
            json.dump(intent, f, indent=2)
    return out


def generate_variants(scene_dir: str, seed: int = 42) -> List[str]:
    """Generate all variant directories as siblings of ``scene_dir``.

    Writes removal (``half/quarter/eighth``), worst-match (``alt_0/2/4``),
    category-aware substitution (``subst_within``/``subst_cross``) and a
    ``scramble`` directory.
    """

    task_path, task_name = _resolve_inputs(scene_dir)
    layout_path = os.path.join(scene_dir, "layout.json")
    with open(task_path) as f:
        task = json.load(f)
    with open(layout_path) as f:
        layout = json.load(f)

    parent = os.path.dirname(os.path.abspath(scene_dir.rstrip("/")))
    written: List[str] = []

    for variant_dir, k in REMOVAL_VARIANTS:
        new_layout = make_removal_layout(layout, k, seed)
        written.append(
            _write_variant(parent, variant_dir, task, task_name, new_layout)
        )

    for variant_dir, rank_offset in ALT_VARIANTS:
        new_layout, intent = make_worst_match_layout(task, layout, rank_offset)
        written.append(
            _write_variant(
                parent, variant_dir, task, task_name, new_layout, intent
            )
        )

    for variant_dir, mode in SUBST_VARIANTS:
        new_layout, intent = make_category_subst_layout(task, layout, mode)
        written.append(
            _write_variant(
                parent, variant_dir, task, task_name, new_layout, intent
            )
        )

    # Layout-scramble: relocate every instance within the floor polygon.
    floor_vertices = (
        task.get("boundary", {}).get("floor_vertices", []) or []
    )
    scrambled = scramble_layout(layout, floor_vertices, seed)
    written.append(
        _write_variant(parent, SCRAMBLE_VARIANT, task, task_name, scrambled)
    )

    return written


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    written = generate_variants(args.scene_dir, seed=args.seed)
    print(f"Wrote {len(written)} variant dirs:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
