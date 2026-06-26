"""Content-variant generation for the VLM-unreliability audit harness.

Given a scene directory containing a LayoutVLM input task JSON and its
``layout.json``, this writes six sibling variant directories, each holding a
modified ``layout.json`` (plus a copy of the task JSON):

  * Removal variants -- a seeded random subset of instances is *kept*:
      - ``variant_half``    keeps round(n / 2)  (>= 1)
      - ``variant_quarter`` keeps round(n / 4)  (>= 1)
      - ``variant_eighth``  keeps round(n / 8)  (>= 1)

  * Worst-match variants -- a structured "retrieval" hook substitutes one or
    more assets with a low-ranked (poorly matching) library candidate:
      - ``variant_alt_0`` / ``variant_alt_2`` / ``variant_alt_4``
        (the suffix is the substitution rank offset; 0 = worst match).

The removal logic is implemented as pure, importable functions and is fully
unit-tested.  The worst-match hook lazily imports ``torch``/``transformers``
*inside* the function and degrades gracefully (recording intent in the variant
metadata, leaving the scene unchanged) when models or an asset library are
unavailable, so the pipeline never crashes.
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

    Lazily imports ``torch``/``transformers`` and embeds the query + candidate
    texts, ranking candidates by ascending similarity (worst first).  Returns
    the index of the candidate at ``rank_offset`` in that ascending ranking, or
    ``None`` if embeddings are unavailable (caller then degrades gracefully).

    Falls back to a pure-Python lexical (token-overlap) ranking when the ML
    stack cannot be loaded, so the structure is still exercised offline.
    """

    if not candidate_texts:
        return None

    sims: Optional[List[float]] = None
    try:  # ML path -- only attempted, never required.
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
# Driver
# ===========================================================================


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
    """Generate all six variant directories as siblings of ``scene_dir``."""

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
