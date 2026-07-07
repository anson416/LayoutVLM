"""Text->scene specification for LayoutVLM (Q1).

LayoutVLM's :class:`~src.layoutvlm.layoutvlm.LayoutVLM.solve` consumes a
*task JSON* (``task_description`` + ``layout_criteria`` + ``boundary`` +
``assets``).  None of those come from a free-text prompt on their own:
the description is fed to the LLM, but the room geometry and the asset set
must be supplied.  This module fills that gap with a single LLM call that
turns a natural-language scene description into a structured task spec:

    {
      "task_description": <echo of the prompt>,
      "layout_criteria": <criteria the LLM derives>,
      "boundary": {
        "floor_vertices": [[x,y,0], ...],
        "wall_height": <h>
      },
      "asset_spec": [ {"category": "...", "description": "...", "count": N}, ... ]
    }

The ``asset_spec`` list is *not* yet resolved to real Objaverse assets -- it
is a declarative shopping list.  Resolution to actual ``<uid>`` assets happens
in :func:`resolve_asset_spec`, which looks each category/description up in the
processed-asset index produced by :func:`prepare_task_assets`.  When no match
is found the entry is dropped (with a warning) rather than aborting, so a
partial scene is still generatable.

The LLM call is OpenAI-compatible (langchain_openai.ChatOpenAI) and accepts
``base_url`` / ``api_key`` / ``model`` / ``temperature`` so it can be pinned
from the CLI.  It is the *only* extra LLM cost beyond the normal solve.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SCENE_SPEC_SYSTEM = (
    "You are a 3D scene designer. Given a textual scene description, output "
    "ONLY a JSON object (no prose, no markdown fences) describing a single "
    "rectangular room and the objects to place in it."
)

SCENE_SPEC_USER_TMPL = """\
Scene description: {prompt}

Produce a JSON object with EXACTLY these keys:
- "task_description": the scene description, lightly cleaned.
- "layout_criteria": one or two sentences of layout guidance derived from the
  description (e.g. "place the bed against the longer wall; nightstand beside it").
- "boundary": an object with:
    - "floor_vertices": a list of 4 [x, y, 0] points (meters), CCW, origin at a
      corner, describing a RECTANGULAR floor. Keep the room modest (<= 8m per side).
    - "wall_height": a single float (meters), typically 2.5-3.0.
- "asset_spec": a list of objects, each:
    {{"category": <short noun, e.g. "bed">, "description": <one phrase>,
      "count": <integer >= 1>}}

Use 4 to 16 asset_spec entries. Do not include paths or uids. Output JSON only.
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[dict]:
    """Pull a JSON object out of an LLM response (handles ```json fences)."""

    # Prefer a fenced block.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fall back to the first {...} in the text.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _coerce_boundary(spec: dict) -> dict:
    """Validate / repair the boundary into a rectangular floor + wall height."""

    boundary = spec.get("boundary") or {}
    verts = boundary.get("floor_vertices")
    height = boundary.get("wall_height")

    # Default to a 4x5 room if anything is missing/malformed.
    if not isinstance(verts, list) or len(verts) < 3:
        verts = [[0, 0, 0], [4, 0, 0], [4, 5, 0], [0, 5, 0]]
    else:
        cleaned = []
        for v in verts:
            if not isinstance(v, (list, tuple)) or len(v) < 2:
                continue
            cleaned.append([float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 0.0])
        verts = cleaned or [[0, 0, 0], [4, 0, 0], [4, 5, 0], [0, 5, 0]]
    if not isinstance(height, (int, float)) or height <= 0:
        height = 2.7
    return {"floor_vertices": verts, "wall_height": float(height)}


def _coerce_asset_spec(spec: dict) -> List[dict]:
    """Validate the asset_spec list into clean records."""

    raw = spec.get("asset_spec")
    if not isinstance(raw, list):
        return []
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        cat = str(entry.get("category", "")).strip()
        desc = str(entry.get("description", "")).strip()
        try:
            count = int(entry.get("count", 1))
        except (TypeError, ValueError):
            count = 1
        if count < 1:
            count = 1
        if not cat:
            continue
        out.append({"category": cat, "description": desc or cat, "count": count})
    return out


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def generate_scene_spec(
    prompt: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> dict:
    """Ask the LLM to turn ``prompt`` into a structured scene spec dict.

    Returns a validated dict with keys ``task_description``, ``layout_criteria``,
    ``boundary`` (``floor_vertices`` + ``wall_height``) and ``asset_spec``.
    Raises ``RuntimeError`` if the LLM output cannot be parsed as JSON after
    a small number of retries.
    """

    llm = ChatOpenAI(
        model_name=model, base_url=base_url, api_key=api_key,
        temperature=float(temperature), max_tokens=max_tokens,
    )
    user = SCENE_SPEC_USER_TMPL.format(prompt=prompt)
    last_err = None
    for _ in range(3):
        try:
            resp = llm.invoke([HumanMessage(content=[
                {"type": "text", "text": SCENE_SPEC_SYSTEM + "\n\n" + user},
            ])])
            parsed = _extract_json(resp.content)
            if parsed is None:
                last_err = "could not extract JSON from LLM response"
                continue
            parsed["task_description"] = str(parsed.get("task_description", prompt))
            parsed.setdefault("layout_criteria", "")
            parsed["boundary"] = _coerce_boundary(parsed)
            parsed["asset_spec"] = _coerce_asset_spec(parsed)
            return parsed
        except Exception as exc:  # network / API errors -> retry
            last_err = str(exc)
    raise RuntimeError(f"scene-spec generation failed: {last_err}")


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------

def build_asset_index(asset_dir: str) -> List[dict]:
    """Scan a processed-asset directory into a flat list of asset records.

    Each ``<asset_dir>/<uid>/data.json`` is expected to carry ``annotations``
    (``category``/``description``/on* flags) and ``assetMetadata.boundingBox``.
    Returns records of ``{uid, path, category, description, bbox, onFloor,
    onCeiling, onWall}``.  Missing/invalid entries are skipped with a warning.
    """

    index: List[dict] = []
    if not asset_dir or not os.path.isdir(asset_dir):
        print(f"[scene_spec] asset_dir not found: {asset_dir!r}")
        return index
    for name in sorted(os.listdir(asset_dir)):
        dpath = os.path.join(asset_dir, name, "data.json")
        if not os.path.exists(dpath):
            continue
        try:
            with open(dpath) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[scene_spec] skipping {name}: bad data.json ({exc})")
            continue
        ann = data.get("annotations", {}) or {}
        bbox = (data.get("assetMetadata", {}) or {}).get("boundingBox", {})
        if not ann or not bbox:
            continue
        index.append({
            "uid": name,
            "path": os.path.join(asset_dir, name, f"{name}.glb"),
            "category": str(ann.get("category", "")).strip(),
            "description": str(ann.get("description", "")).strip(),
            "bbox": bbox,
            "onFloor": bool(ann.get("onFloor", True)),
            "onCeiling": bool(ann.get("onCeiling", False)),
            "onWall": bool(ann.get("onWall", False)),
        })
    return index


def _tokenize(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def resolve_asset_spec(
    scene_spec: dict, asset_dir: str
) -> Tuple[dict, List[str]]:
    """Resolve a scene_spec's ``asset_spec`` into concrete asset uids.

    For each requested ``(category, count)`` we greedily pick assets from the
    index whose category matches best (exact category match preferred; else the
    highest token-overlap with the requested category+description).  Picked
    uids are not re-used across requests.  Returns ``(task, warnings)`` where
    ``task`` is a *raw* LayoutVLM task JSON whose ``assets`` are keyed
    ``<uid>-<idx>`` with a ``count`` of how many instances to place, and
    ``warnings`` lists any requests that could not be fully satisfied.

    The returned task is *raw* (entries carry only ``uid`` + ``count``); it is
    meant to be passed through :func:`main.prepare_task_assets`, which reloads
    full metadata (path, bounding box, annotations, on* flags) from
    ``<asset_dir>/<uid>/data.json`` and produces the prepared-task shape the
    solver expects.  Keeping resolution and normalization decoupled avoids
    duplicating the data.json parsing logic here.

    There is no explicit "best vs worst" sort here -- the worst-object fork is
    applied later, post-solve, against a separate asset library
    (see :mod:`vlmunr_variants.make_worst_object_layout`).
    """

    index = build_asset_index(asset_dir)
    warnings: List[str] = []
    assets: Dict[str, dict] = {}

    # Group index by normalized category for exact lookup.
    by_cat: Dict[str, List[dict]] = {}
    for rec in index:
        by_cat.setdefault(rec["category"].lower(), []).append(rec)

    used_uids = set()
    for req in scene_spec.get("asset_spec", []):
        cat = req["category"].lower()
        desc = req["description"]
        count = req["count"]
        # Exact-category matches, filtered to not-yet-used uids.
        pool = [r for r in by_cat.get(cat, []) if r["uid"] not in used_uids]
        if not pool:
            # Fuzzy: token-overlap against category+description.
            q = _tokenize(f"{cat} {desc}")
            scored = []
            for r in index:
                if r["uid"] in used_uids:
                    continue
                t = _tokenize(f"{r['category']} {r['description']}")
                inter = len(q & t)
                union = len(q | t) or 1
                scored.append((inter / union, r))
            scored.sort(key=lambda x: -x[0])
            pool = [r for _, r in scored if _[0] > 0]
        if not pool:
            warnings.append(f"no asset found for {cat!r} ({desc!r})")
            continue
        placed = 0
        for r in pool:
            if placed >= count:
                break
            # One asset uid can supply multiple instances; emit one entry per
            # requested instance so prepare_task_assets sees count==1 entries
            # and the sandbox can place each independently.
            inst_id = f"{r['uid']}-{placed}"
            if inst_id in assets:
                continue
            assets[inst_id] = {"uid": r["uid"], "count": 1}
            used_uids.add(r["uid"])
            placed += 1
        if placed < count:
            warnings.append(
                f"only {placed}/{count} assets found for {cat!r} ({desc!r})"
            )

    task = {
        "task_description": scene_spec.get("task_description", ""),
        "layout_criteria": scene_spec.get("layout_criteria", ""),
        "boundary": scene_spec["boundary"],
        "assets": assets,
    }
    return task, warnings


# ---------------------------------------------------------------------------
# Mock (no-LLM) spec for --mock mode
# ---------------------------------------------------------------------------

def mock_scene_spec(prompt: str) -> dict:
    """Build a minimal valid scene spec without any LLM call.

    Used by the CLI ``--mock`` flag so the full pipeline (spec -> resolve ->
    solve-or-mock -> variants) is exercisable offline.  The boundary is a fixed
    4x5 room and the asset_spec is a single placeholder chair.
    """

    return {
        "task_description": prompt,
        "layout_criteria": "arrange according to the floor plan layout of the room",
        "boundary": {
            "floor_vertices": [[0, 0, 0], [4, 0, 0], [4, 5, 0], [0, 5, 0]],
            "wall_height": 2.7,
        },
        "asset_spec": [
            {"category": "chair", "description": "a placeholder chair", "count": 2},
        ],
    }
