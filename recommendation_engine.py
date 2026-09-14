"""
recommendation_engine.py

The missing intelligence layer: turns a UserProfile + wardrobe + catalog +
weather into exactly three composed outfit looks.

This module is a PURE FUNCTION over the data it's given -- it does not call
closet_manager, catalog, or weather_service itself. app.py is responsible
for gathering those inputs (it already fetches the wardrobe today) and
passing them in. That keeps this file trivially unit-testable and decoupled
from I/O, per SOLID / separation-of-concerns.

Kept deliberately rule-based (no ML), per the project's MVP constraints --
but every scoring function below is a clean, separately-callable unit, so a
future ML-based re-ranker can be swapped in by replacing score_garment_fit()
and color_harmony_bonus() without touching the composition logic around them.

Reuses, unmodified:
  - closet_manager.get_user_closet() output shape (category, label,
    dominant_hex, dominant_hue, image_b64, ...)
  - yolo_outfit_detect.py's label vocabulary (tshirt, shirt, short sleeve
    shirt, long sleeve shirt, sweater, jacket, dress / pants, jeans,
    trousers, skirt / shoes / bag)
  - styling_rules.py's body-type key normalization ("Pear" -> "triangle")
  - app.py's exact body_type / face_shape / undertone value strings
"""

import colorsys
import hashlib
import json
import os


def _garment_id(garment):
    """
    Stable identity for a garment regardless of source:
      - wardrobe items from closet_manager.py already have "image_hash"
      - catalog items have an explicit "id"
      - freshly-detected items from THIS request (before they're persisted
        to the closet) have neither -- derive a stable id from their
        content so exclusion/dedup logic works the same for all three.
    """
    if garment.get("id"):
        return garment["id"]
    if garment.get("image_hash"):
        return garment["image_hash"]
    basis = f"{garment.get('category')}:{garment.get('label')}:{garment.get('dominant_hex')}"
    return hashlib.md5(basis.encode("utf-8")).hexdigest()

# ─────────────────────────────────────────────────────────────────────────
# Tunable weights -- named constants, not magic numbers buried in logic.
# This is the one place to adjust behavior without touching the algorithm,
# and the natural seam for a future ML-based re-ranker.
# ─────────────────────────────────────────────────────────────────────────
W_BODY = 0.20
W_FACE = 0.10
W_UNDERTONE = 0.20
W_WEATHER = 0.15
W_COLOR_HARMONY = 0.30

W_RECENCY = 0.15  # bonus for garments detected in THIS request, not pulled from older closet history

MAX_RATIONALE_ITEMS = 4

# ─────────────────────────────────────────────────────────────────────────
# Undertone color preferences -- the concrete fix for the "computed then
# discarded" undertone bug flagged in the audit. Content lives in
# styling_data/undertones/<undertone>.json, not here -- this module is a
# consumer of that styling data, not its owner, same separation already
# used for body-type content (styling_data/*.json + styling_rules.py).
# ─────────────────────────────────────────────────────────────────────────
_UNDERTONE_STYLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styling_data", "undertones")
_undertone_style_cache = {}


def _load_undertone_style(undertone):
    key = (undertone or "").strip().lower()
    if key not in {"warm", "cool", "neutral"}:
        key = "neutral"
    if key in _undertone_style_cache:
        return _undertone_style_cache[key]
    path = os.path.join(_UNDERTONE_STYLE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return _load_undertone_style("neutral") if key != "neutral" else {}
    with open(path, "r") as f:
        data = json.load(f)
    _undertone_style_cache[key] = data
    return data


def _flatten_ideal_colors(style):
    """style["ideal_colors"] is grouped by color family (reds, blues, ...) --
    scoring/display just needs one flat list of names, regardless of family."""
    ideal_colors = style.get("ideal_colors", {})
    flat = []
    for family_colors in ideal_colors.values():
        flat.extend(family_colors)
    return flat

# ─────────────────────────────────────────────────────────────────────────
# Body-shape hints. Two things live here:
#   1. LABEL preferences, used for real per-garment scoring (only works
#      with the actual label vocabulary the detector can produce today).
#   2. CUT descriptions, used only for the "styling" text payload sent to
#      SuggestionFlow.jsx -- these are display strings, not scoring inputs.
# Covers all 5 body types classify_body_type_v2() can output.
# ─────────────────────────────────────────────────────────────────────────
BODY_SHAPE_LABEL_PREFERENCE = {
    "inverted_triangle": {
        "top": {"prefer": [], "avoid": ["jacket"]},
        "bottom": {"prefer": ["skirt", "trousers"], "avoid": []},
    },
    "rectangle": {
        "top": {"prefer": ["jacket", "sweater"], "avoid": []},
        "bottom": {"prefer": [], "avoid": []},
    },
    "hourglass": {
        "top": {"prefer": ["shirt", "dress"], "avoid": []},
        "bottom": {"prefer": [], "avoid": []},
    },
    "triangle": {  # "Pear" is normalized to this key, matching styling_rules.py
        "top": {"prefer": ["sweater", "jacket"], "avoid": []},
        "bottom": {"prefer": ["trousers", "jeans"], "avoid": ["skirt"]},
    },
    "apple": {
        "top": {"prefer": ["dress", "sweater", "long sleeve shirt"], "avoid": ["short sleeve shirt"]},
        "bottom": {"prefer": [], "avoid": []},
    },
}

BODY_CUT_HINTS = {
    "inverted_triangle": {
        "tops": ["Relaxed crew-neck", "Simple scoop neck"],
        "bottoms": ["Wide-leg trousers", "A-line skirt"],
        "dresses": ["A-line dress"],
    },
    "rectangle": {
        "tops": ["Structured blazer", "Layered shirt"],
        "bottoms": ["Belted trousers"],
        "dresses": ["Wrap dress"],
    },
    "hourglass": {
        "tops": ["Fitted wrap top", "Tailored shirt"],
        "bottoms": ["High-waisted straight leg"],
        "dresses": ["Wrap dress", "Bodycon dress"],
    },
    "triangle": {
        "tops": ["Boat neck top", "Ruffle-detail top"],
        "bottoms": ["Dark straight-leg trousers"],
        "dresses": ["A-line dress"],
    },
    "apple": {
        "tops": ["Flowy V-neck", "Open cardigan"],
        "bottoms": ["Straight-leg trousers"],
        "dresses": ["Empire-waist dress"],
    },
}


def _normalize_body_key(body_type):
    """Same remap styling_rules.py already applies, kept consistent here."""
    key = (body_type or "").lower().replace(" ", "_")
    if key == "pear":
        key = "triangle"
    return key


def _normalize_type_string(value):
    """Same normalization styling_rules.py already applies, kept consistent
    here (see _normalize_body_key above for why this file duplicates small
    pure helpers instead of importing them)."""
    if not value:
        return ""
    return value.lower().replace("_", " ").replace("-", " ").strip()


# ─────────────────────────────────────────────────────────────────────────
# Silhouette-level body-shape matching -- reads styling_data/body_type/
# <key>.json's per-category recommended_items/avoid_items (jeans silhouette
# + rise, shirt/top fit), matching a garment's type/fit/subcategory against
# them. This is a finer signal than BODY_SHAPE_LABEL_PREFERENCE below (which
# only knows "jeans" vs "skirt" vs "jacket" at the category/label level, not
# "bootcut" vs "skinny"). Content lives in styling_data/, not here -- same
# separation already used for undertone content via _load_undertone_style.
# ─────────────────────────────────────────────────────────────────────────
_BODY_TYPE_STYLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styling_data", "body_type")
_body_type_style_cache = {}

_VALID_BODY_KEYS = {"inverted_triangle", "rectangle", "hourglass", "triangle", "apple"}

# (garment category, garment label) -> which style_guide key holds its
# silhouette guidance. Matches exactly how catalog.py shapes shirts/jeans/
# tops rows (category="bottom"/label="jeans", category="top"/label="shirt"
# or "tshirt").
_SILHOUETTE_CATEGORY_KEY = {
    ("bottom", "jeans"): "jeans",
    ("top", "shirt"): "shirt_fits",
    ("top", "tshirt"): "top_fits",
    ("dress", "dress"): "dresses",
}


def _load_body_type_style(body_type):
    key = _normalize_body_key(body_type)
    if key not in _VALID_BODY_KEYS:
        return {}
    if key in _body_type_style_cache:
        return _body_type_style_cache[key]
    path = os.path.join(_BODY_TYPE_STYLE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    style_guide = data.get("style_guide", {}).get("Female", {})
    _body_type_style_cache[key] = style_guide
    return style_guide


def _silhouette_bonus(garment, body_type):
    """
    Matches a garment's type/fit (catalog items) or subcategory/fit
    (current-request items enriched by garment_classifier.py -- see
    styling_rules.py's _match_current_item_to_style_guide, which reads the
    same two fields) against this body type's recommended_items/
    avoid_items, via the same normalize+substring approach styling_rules.py
    already uses. Older persisted wardrobe items don't carry these fields
    at all (closet_manager.py doesn't save them) so this correctly falls
    through to a neutral (0.0, None) for them, not a crash or false match.
    """
    style_guide = _load_body_type_style(body_type)
    if not style_guide:
        return 0.0, None

    guide_key = _SILHOUETTE_CATEGORY_KEY.get((garment.get("category"), (garment.get("label") or "").lower()))
    if not guide_key:
        return 0.0, None

    category_data = style_guide.get(guide_key)
    if not category_data:
        return 0.0, None

    candidates = [c for c in (
        _normalize_type_string(garment.get("type")),
        _normalize_type_string(garment.get("fit")),
        _normalize_type_string(garment.get("subcategory")),
    ) if c]
    if not candidates:
        return 0.0, None

    for entry in category_data.get("recommended_items", []):
        entry_type = _normalize_type_string(entry.get("type"))
        if entry_type and any(entry_type in c or c in entry_type for c in candidates):
            display = entry_type.title()
            return 1.0, entry.get("advice") or f"{display} suits your {body_type} shape"

    for entry in category_data.get("avoid_items", []):
        entry_type = _normalize_type_string(entry.get("type"))
        if entry_type and any(entry_type in c or c in entry_type for c in candidates):
            return -0.8, entry.get("advice") or None

    return 0.0, None


# ─────────────────────────────────────────────────────────────────────────
# Color helpers
# ─────────────────────────────────────────────────────────────────────────

def _hex_to_hsv(hex_color):
    hex_color = (hex_color or "").lstrip("#")
    if len(hex_color) != 6:
        return None
    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
    except ValueError:
        return None
    return colorsys.rgb_to_hsv(r, g, b)


def _approximate_color_name(hue, hex_color):
    """
    Fallback for wardrobe items, which (per closet_manager.py) store
    dominant_hex/dominant_hue but NOT color_name -- only freshly-detected
    items from the current /analyze call have color_name. This is a
    deliberately crude hue-bucket approximation; color_utils.get_color_name()
    is more accurate but needs saturation/value data the closet doesn't
    persist. Good enough for scoring; not a replacement for that function.
    """
    if hue is None:
        return None
    if hue < 20 or hue >= 345:
        return "Red"
    if hue < 50:
        return "Brown"
    if hue < 70:
        return "Yellow"
    if hue < 170:
        return "Green"
    if hue < 260:
        return "Navy Blue"
    if hue < 320:
        return "Purple"
    return "Pink"


def _color_name_of(garment):
    return garment.get("color_name") or _approximate_color_name(
        garment.get("dominant_hue"), garment.get("dominant_hex")
    )


def color_harmony_bonus(hex_a, hex_b, prefer_contrast=False):
    """
    Default mode (Look A / C -- "best overall" pairing):
      +1.0  complementary pairing (hues roughly opposite)
      +0.5  tonal/monochrome pairing (hues close together)
      +1.0  either garment is a neutral (white/gray/black) -- pairs with anything
      -0.2  hues clash (neither close nor complementary)

    prefer_contrast=True (Look B -- "Elevated Contrast" archetype):
      Explicitly seeks complementary pairings over tonal ones, rather than
      treating both as generically "harmonious." This is what makes Look B
      a genuine style STRATEGY, not just a different candidate pool with the
      same scoring preferences as the other two looks.
      +1.5  complementary pairing (rewarded more than in default mode)
       0.0  tonal/monochrome pairing (no longer rewarded -- Look B isn't
            looking for "safe," it's looking for contrast)
      +0.5  neutral involved (still pairs safely, just not the goal)
      -0.2  hues clash
    """
    hsv_a = _hex_to_hsv(hex_a)
    hsv_b = _hex_to_hsv(hex_b)
    if not hsv_a or not hsv_b:
        return 0.0

    h_a, s_a, _ = hsv_a
    h_b, s_b, _ = hsv_b
    is_neutral = s_a < 0.15 or s_b < 0.15

    diff = abs(h_a - h_b) * 360
    diff = min(diff, 360 - diff)
    is_complementary = 150 <= diff <= 210
    is_tonal = diff <= 30

    if prefer_contrast:
        if is_complementary:
            return 1.5
        if is_neutral:
            return 0.5
        if is_tonal:
            return 0.0
        return -0.2

    if is_neutral:
        return 1.0
    if is_tonal:
        return 0.5
    if is_complementary:
        return 1.0
    return -0.2


# ─────────────────────────────────────────────────────────────────────────
# Per-garment scoring
# ─────────────────────────────────────────────────────────────────────────

def _body_shape_bonus(garment, body_type):
    """
    Two signals, in strict priority order -- NOT combined by magnitude:
      1. Specific silhouette match against styling_data/body_type/*.json
         (bootcut/wide_leg/fitted/oversized/etc -- _silhouette_bonus above).
         If the garment's exact type/fit is listed in that body type's
         recommended_items OR avoid_items, that verdict wins outright.
      2. Coarse category/label preference (jacket/sweater/skirt/trousers --
         BODY_SHAPE_LABEL_PREFERENCE, the original logic) -- used ONLY when
         the silhouette check found nothing specific to say.
    A magnitude comparison would let a strong generic positive (e.g. "jeans
    suit a Triangle body" at +1.0) bury a specific negative (e.g. "but not
    skinny ones" at -0.8) just because it's numerically bigger -- verified
    against real data: skinny jeans for a Triangle profile scored positive
    under magnitude comparison despite being an explicit avoid_items entry.
    Presence, not size, decides which signal applies.
    """
    silhouette_score, silhouette_reason = _silhouette_bonus(garment, body_type)
    if silhouette_score != 0.0:
        return silhouette_score, silhouette_reason

    key = _normalize_body_key(body_type)
    rules = BODY_SHAPE_LABEL_PREFERENCE.get(key)
    if not rules:
        return 0.0, None

    category = garment.get("category")
    label = (garment.get("label") or "").lower()
    slot = rules.get(category)
    if not slot:
        return 0.0, None

    if label in slot.get("prefer", []):
        return 1.0, f"{label.title()} suits your {body_type} shape"
    if label in slot.get("avoid", []):
        return -0.5, None
    return 0.0, None


def _face_shape_bonus(garment, face_shape):
    """
    Intentionally a no-op today. styling_database.json's face_rules are
    keyed on NECKLINE (e.g. "V-neck", "Boat neck"), but no garment
    attribute in this codebase captures neckline -- YOLO and SegFormer
    both stop at category/label, not cut detail. Rather than fake a score
    from data that doesn't exist, this returns neutral and is kept as a
    real function (with its weight already reserved above) so it can be
    wired in the moment neckline metadata exists -- manual tagging, a
    future classifier, or catalog-provided attributes.
    """
    return 0.0, None


def _undertone_bonus(garment, undertone):
    style = _load_undertone_style(undertone)
    palette = _flatten_ideal_colors(style)
    if not palette:
        return 0.0, None

    color_name = _color_name_of(garment)
    if not color_name:
        return 0.0, None

    if any(p.lower() in color_name.lower() for p in palette):
        return 1.0, f"{color_name} complements your {(undertone or 'neutral').lower()} undertone"

    # Each undertone owns its own avoid list (no cross-file inference).
    # Warm's avoid_colors is real content; Cool/Neutral are still empty
    # until equivalent avoid-guidance is supplied for them.
    avoid = style.get("avoid_colors", [])
    if avoid and any(p.lower() in color_name.lower() for p in avoid):
        return -0.5, None

    return 0.0, None


# ─────────────────────────────────────────────────────────────────────────
# Weather styling rules -- condition -> favor/avoid content lives in
# styling_data/weather_styling_rules.json, not here, same separation
# already used for undertone and body-type content above. Keyed by the
# same hot/cold/rain/mild buckets weather_service.py derives from the
# real OpenWeatherMap temperature reading (_bucket_condition), so this
# module never re-guesses temperature thresholds -- it just consumes
# whichever bucket the real reading landed in.
# ─────────────────────────────────────────────────────────────────────────
_WEATHER_STYLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styling_data", "weather_styling_rules.json")
_weather_style_cache = None

_WEATHER_AVOID_SCORE = {"hot": -1.0, "cold": -0.5}


def _load_weather_style():
    global _weather_style_cache
    if _weather_style_cache is None:
        if not os.path.exists(_WEATHER_STYLE_PATH):
            _weather_style_cache = {}
        else:
            with open(_WEATHER_STYLE_PATH, "r") as f:
                _weather_style_cache = json.load(f)
    return _weather_style_cache


def _weather_bonus(garment, weather):
    """
    Deliberately conservative: the detector's label vocabulary has no
    fabric/material attribute (e.g. no "suede" tag exists anywhere in the
    pipeline), so the "avoid suede in rain" rule from the spec can only be
    partially honored -- footwear gets a small positive nudge in rain
    rather than a confident "waterproof" claim, since we can't actually
    tell the difference with current data.

    `weather` is the real {"temperature_c", "condition"} reading passed
    through from weather_service.py -- condition picks the rule set,
    temperature_c is surfaced in the rationale so the suggestion reads as
    grounded in today's actual weather rather than a generic bucket.
    """
    weather = weather or {}
    condition = weather.get("condition")
    temp_c = weather.get("temperature_c")

    style = _load_weather_style().get(condition or "", {})
    if not style:
        return 0.0, None

    label = (garment.get("label") or "").lower()
    category = garment.get("category")

    temp_suffix = f" ({temp_c:.0f}°C)" if isinstance(temp_c, (int, float)) else ""

    if label in style.get("favor_labels", []):
        return 1.0, f"Good pick for today's {condition} weather{temp_suffix}"

    if label in style.get("avoid_labels", []):
        return _WEATHER_AVOID_SCORE.get(condition, -0.5), None

    if category in style.get("favor_categories", []):
        return 0.3, None

    return 0.0, None


def score_garment_fit(garment, profile):
    """
    Returns (score, rationale_list). Pairing-independent -- color harmony
    between two chosen garments is scored separately, at look-assembly time,
    since it depends on which two garments end up together.
    """
    score = 0.0
    rationale = []

    b_score, b_reason = _body_shape_bonus(garment, profile.get("body_type"))
    score += W_BODY * b_score
    if b_reason:
        rationale.append(b_reason)

    f_score, f_reason = _face_shape_bonus(garment, profile.get("face_shape"))
    score += W_FACE * f_score
    if f_reason:
        rationale.append(f_reason)

    u_score, u_reason = _undertone_bonus(garment, profile.get("undertone"))
    score += W_UNDERTONE * u_score
    if u_reason:
        rationale.append(u_reason)

    weather = profile.get("weather") or {}
    w_score, w_reason = _weather_bonus(garment, weather)
    score += W_WEATHER * w_score
    if w_reason:
        rationale.append(w_reason)

    if garment.get("is_current"):
        score += W_RECENCY

    return score, rationale


# ─────────────────────────────────────────────────────────────────────────
# Outfit composition
# ─────────────────────────────────────────────────────────────────────────

def _group_by_category(items):
    grouped = {}
    for item in items or []:
        grouped.setdefault(item.get("category"), []).append(item)
    return grouped


def _best_candidate(candidates, profile, excluded_ids=None):
    """
    candidates: list of garment dicts. Returns (score, garment, rationale) or None.

    excluded_ids: garment IDs already used by an earlier look in this same
    generate_three_looks() call. Filtered out first -- but if that would
    leave zero candidates (a small wardrobe/catalog genuinely has nothing
    else in this category), we fall back to the full candidate list rather
    than returning an empty category. That fallback is intentional and
    logged, not a silent failure.
    """
    if not candidates:
        return None

    excluded_ids = excluded_ids or set()
    filtered = [g for g in candidates if _garment_id(g) not in excluded_ids]
    pool = filtered if filtered else candidates

    scored = [(*score_garment_fit(g, profile), g) for g in pool]
    scored.sort(key=lambda t: t[0], reverse=True)
    score, rationale, garment = scored[0]
    return score, garment, rationale


def compose_look(look_id, wardrobe_by_cat, catalog_by_cat, profile, catalog_allowed_categories,
                  excluded_ids=None, require_current=False, prefer_contrast=False,
                  archetype=None, archetype_description=None):
    """
    catalog_allowed_categories: set of categories permitted to pull from the
    catalog. Categories NOT in this set still fall back to catalog if the
    wardrobe has zero items there -- an empty wardrobe slot is not a reason
    to leave a look incomplete.

    excluded_ids: garment IDs already selected by an earlier look in this
    same request -- prevents Look B/C from silently converging on Look A's
    picks. This is what actually guarantees diversity, not a side effect of
    scoring differently.

    require_current: when True (used for Look A), any category where the
    just-uploaded photo actually detected a garment is LOCKED to that
    garment -- not just score-favored via the recency bonus, which weather/
    body/undertone scoring can legitimately outweigh. This is what makes
    "Look A reflects the photo you just uploaded" a structural guarantee
    instead of a probabilistic outcome. Categories the photo didn't detect
    anything in still fall through to normal scoring.
    """
    excluded_ids = set(excluded_ids or set())
    chosen = {}
    total_score = 0.0
    rationale = []
    wardrobe_count = 0
    total_count = 0

    # A dress is a real, separate garment category (see yolo_outfit_detect.py)
    # -- it replaces the top+bottom slots entirely rather than pairing with
    # them. Two cases use it instead of the normal top/bottom loop below:
    # (1) the just-uploaded photo actually detected a dress (mirrors the
    #     same require_current structural guarantee used for top/bottom), or
    # (2) there's simply no top/bottom candidate data anywhere (wardrobe or
    #     catalog) but dress candidates exist -- better than a look with
    #     missing garments.
    dress_wardrobe = [g for g in wardrobe_by_cat.get("dress", []) if _garment_id(g) not in excluded_ids]
    dress_current = [g for g in dress_wardrobe if g.get("is_current")]
    dress_catalog = catalog_by_cat.get("dress", []) if "dress" in catalog_allowed_categories else []
    no_top_bottom_signal = not (
        wardrobe_by_cat.get("top") or wardrobe_by_cat.get("bottom")
        or catalog_by_cat.get("top") or catalog_by_cat.get("bottom")
    )
    use_dress = bool(require_current and dress_current) or bool(
        no_top_bottom_signal and (dress_wardrobe or dress_catalog)
    )

    if use_dress:
        dress_candidates = dress_current if (require_current and dress_current) else (dress_wardrobe + dress_catalog)
        best = _best_candidate(dress_candidates, profile, excluded_ids)
        if best:
            score, garment, reason = best
            chosen["dress"] = garment
            total_score += score
            rationale.extend(reason)
            total_count += 1
            if garment.get("source") == "wardrobe":
                wardrobe_count += 1
            print(f"[RECOM_ENGINE] Look {look_id} / dress: id={_garment_id(garment)} "
                  f"label={garment.get('label')} source={garment.get('source')} score={round(score, 3)}")

    categories_to_fill = ("footwear",) if chosen.get("dress") else ("top", "bottom", "footwear")

    for category in categories_to_fill:
        wardrobe_candidates = wardrobe_by_cat.get(category, [])
        catalog_candidates = catalog_by_cat.get(category, [])
        wardrobe_after_exclusion = [g for g in wardrobe_candidates if _garment_id(g) not in excluded_ids]

        current_here = [g for g in wardrobe_after_exclusion if g.get("is_current")]
        if require_current and current_here:
            # Structural guarantee, not a scoring nudge: if the just-uploaded
            # photo detected something in this category, use it -- full stop.
            candidates = current_here
        elif category in catalog_allowed_categories:
            candidates = wardrobe_after_exclusion + catalog_candidates
        elif wardrobe_after_exclusion:
            candidates = wardrobe_after_exclusion
        else:
            # Wardrobe is either empty or every item was already used by an
            # earlier look -- widen to catalog rather than falling back to
            # re-picking an already-used wardrobe item. This is the fix for
            # the residual A==B duplicate: without this branch, a single-item
            # wardrobe with catalog disallowed had nowhere else to go.
            candidates = catalog_candidates

        best = _best_candidate(candidates, profile, excluded_ids)
        if not best:
            continue

        score, garment, reason = best
        chosen[category] = garment
        total_score += score
        rationale.extend(reason)
        total_count += 1
        if garment.get("source") == "wardrobe":
            wardrobe_count += 1
        print(f"[RECOM_ENGINE] Look {look_id} / {category}: id={_garment_id(garment)} "
              f"label={garment.get('label')} source={garment.get('source')} score={round(score, 3)}")

    # Accessories: wardrobe-only for MVP (no accessory items exist in the
    # catalog seed data today -- Ads.jsx never had any). Up to 2 items.
    acc_candidates = [g for g in wardrobe_by_cat.get("accessories", []) if _garment_id(g) not in excluded_ids]
    if not acc_candidates:
        acc_candidates = wardrobe_by_cat.get("accessories", [])
    acc_scored = sorted(
        (score_garment_fit(g, profile) + (g,) for g in acc_candidates),
        key=lambda t: t[0],
        reverse=True,
    )
    accessories = [g for _, _, g in acc_scored[:2]]

    if chosen.get("top") and chosen.get("bottom"):
        harmony = color_harmony_bonus(
            chosen["top"].get("dominant_hex"), chosen["bottom"].get("dominant_hex"),
            prefer_contrast=prefer_contrast,
        )
        total_score += W_COLOR_HARMONY * harmony
        if harmony > 0:
            rationale.append(
                "Bold, contrasting color pairing" if prefer_contrast and harmony >= 1.5
                else "Top and bottom colors work well together"
            )

    wardrobe_ratio = round(wardrobe_count / total_count, 2) if total_count else 0.0

    return {
        "look_id": look_id,
        "archetype": archetype,
        "archetype_description": archetype_description,
        "top": chosen.get("top"),
        "bottom": chosen.get("bottom"),
        "dress": chosen.get("dress"),
        "footwear": chosen.get("footwear"),
        "accessories": accessories,
        "wardrobe_ratio": wardrobe_ratio,
        "score": round(total_score, 3),
        "rationale": rationale[:MAX_RATIONALE_ITEMS],
    }


def _look_garment_ids(look):
    ids = set()
    for category in ("top", "bottom", "dress", "footwear"):
        g = look.get(category)
        if g:
            ids.add(_garment_id(g))
    for g in look.get("accessories") or []:
        ids.add(_garment_id(g))
    return ids


_WEATHER_ADVICE_LABEL = {"hot": "Hot Day", "cold": "Cold Day", "rain": "Rainy Day", "mild": "Mild Day"}


def _build_weather_advice(weather):
    """
    Global (not per-garment) weather styling tip for the frontend's
    "Weather & Styling Tip" widget -- e.g. "30°C · Hot Day: Wear loose,
    flowy silhouettes and choose breathable natural fabrics like cotton or
    linen." Surfaces description/fit/fabrics.notes from
    styling_data/weather_styling_rules.json as-is (for a UI that wants the
    raw pieces) plus a single composed `tip` sentence (for a UI that just
    wants one line). Deliberately global, not scored per garment -- that's
    what lets this work today, ahead of any fabric/cut classification.

    regional_notes and per-garment fabric scoring are intentionally not
    surfaced here -- reserved for when fabric/cut classification or manual
    tagging exists.
    """
    weather = weather or {}
    condition = weather.get("condition")
    temp_c = weather.get("temperature_c")

    style = _load_weather_style().get(condition or "", {})
    if not style:
        return None

    temp_part = f"{temp_c:.0f}°C" if isinstance(temp_c, (int, float)) else None
    day_label = _WEATHER_ADVICE_LABEL.get(condition, "Today")
    headline = f"{temp_part} · {day_label}" if temp_part else day_label

    fabrics = style.get("fabrics") or {}
    fabric_notes = fabrics.get("notes")

    # One "how to wear it" line -- whichever of these the condition defines.
    fit_line = style.get("fit") or style.get("base") or style.get("hems") or style.get("layering")

    tip_parts = [p for p in (fit_line, fabric_notes) if p]
    tip = " ".join(tip_parts) if tip_parts else style.get("description")

    return {
        "condition": condition,
        "temperature_c": temp_c,
        "headline": headline,
        "description": style.get("description"),
        "fit": style.get("fit"),
        "fabric_notes": fabric_notes,
        "tip": f"{headline}: {tip}" if tip else headline,
    }


def _build_styling_payload(profile, looks):
    """
    Shaped specifically to match what SuggestionFlow.jsx already expects
    (analysisData.styling.{clothing_recommendations, color_palette,
    visual_prompt}) -- per the audit, that UI exists today and silently
    falls back to generic text because the backend never populated this.

    Also adds `weather_advice` (not yet consumed by SuggestionFlow.jsx --
    new field for a future "Weather & Styling Tip" widget).
    """
    body_key = _normalize_body_key(profile.get("body_type"))
    cuts = BODY_CUT_HINTS.get(body_key, BODY_CUT_HINTS["rectangle"])

    undertone = profile.get("undertone")
    best_colors = _flatten_ideal_colors(_load_undertone_style(undertone))

    best_look = max(looks, key=lambda l: l["score"]) if looks else None
    gender = (profile.get("gender") or "").lower() or "your"

    if best_look and best_look.get("dress"):
        dress_label = best_look["dress"]["label"]
        visual_prompt = (
            f"A {gender} look for a {profile.get('body_type', 'balanced')} silhouette "
            f"with {(undertone or 'neutral').lower()} undertones, built around a {dress_label}."
        )
    else:
        top_label = best_look["top"]["label"] if best_look and best_look.get("top") else "a top"
        bottom_label = best_look["bottom"]["label"] if best_look and best_look.get("bottom") else "a bottom"
        visual_prompt = (
            f"A {gender} look for a {profile.get('body_type', 'balanced')} silhouette "
            f"with {(undertone or 'neutral').lower()} undertones, pairing {top_label} with {bottom_label}."
        )

    return {
        "clothing_recommendations": cuts,
        "color_palette": {"best_colors": best_colors},
        "visual_prompt": visual_prompt,
        "weather_advice": _build_weather_advice(profile.get("weather")),
    }


def _merge_current_and_historical(current_outfit_items, wardrobe_items):
    """
    Merges THIS request's freshly-detected garments with the user's broader
    closet history, with current items taking priority on id collisions and
    flagged is_current=True so scoring favors them.
    """
    merged = {}

    for item in current_outfit_items or []:
        item = dict(item)
        item.setdefault("source", "wardrobe")
        item["is_current"] = True
        merged[_garment_id(item)] = item

    for item in wardrobe_items or []:
        gid = _garment_id(item)
        if gid in merged:
            continue  # current-request version already wins
        item = dict(item)
        item.setdefault("source", "wardrobe")
        item.setdefault("is_current", False)
        merged[gid] = item

    return list(merged.values())


def generate_three_looks(profile, current_outfit_items, wardrobe_items, catalog_items, request_id=None):
    """
    Main entry point.

    profile: {
        "body_type": str,     # e.g. "Hourglass", "Pear", "Inverted Triangle", ...
        "face_shape": str,
        "skin_tone": str,
        "undertone": str,     # "Warm" | "Cool" | "Neutral"
        "gender": str,
        "weather": {"temperature_c": float, "condition": str},
    }
    current_outfit_items: garments detected in THIS /analyze call (flatten
        the `outfits` dict app.py already builds via detect_outfits() --
        e.g. [item for items in outfits.values() for item in items]).
        These are prioritized over older closet history, which is what
        makes recommendations actually track the photo just uploaded
        instead of drifting toward whatever scored highest historically.
    wardrobe_items: the user's full closet from closet_manager.get_user_closet()
        -- used as supplementary candidates (e.g. for categories the current
        photo didn't show), never as the dominant signal.
    catalog_items:  list of garment dicts from catalog.get_catalog_items(...).
    request_id: optional identifier for log correlation (e.g. the uploaded
        filename or a request UUID) -- makes it possible to grep logs and
        confirm each upload really is processed independently.

    Returns: {
        "looks": [look_a, look_b, look_c],
        "styling": {...}   # matches SuggestionFlow.jsx's expected shape
    }
    """
    tag = f"[RECOM_ENGINE:{request_id}]" if request_id else "[RECOM_ENGINE]"

    owned_items = _merge_current_and_historical(current_outfit_items, wardrobe_items)
    print(f"{tag} current_items={len(current_outfit_items or [])} "
          f"historical_items={len(wardrobe_items or [])} "
          f"merged_unique={len(owned_items)} catalog_items={len(catalog_items or [])}")
    print(f"{tag} profile: body_type={profile.get('body_type')} undertone={profile.get('undertone')} "
          f"weather={profile.get('weather')}")

    wardrobe_by_cat = _group_by_category(owned_items)
    catalog_by_cat = _group_by_category(catalog_items)

    used_ids = set()

    # ── Look A: "Smart Casual / Current Focus" ──────────────────────────
    # Structurally guaranteed to reflect the just-uploaded photo (require_current)
    # wherever the photo actually detected a garment. Standard (non-contrast)
    # color scoring -- this look is about "you, styled well," not a bold statement.
    look_a = compose_look(
        "A", wardrobe_by_cat, catalog_by_cat, profile,
        catalog_allowed_categories=set(), excluded_ids=used_ids, require_current=True,
        archetype="Smart Casual",
        archetype_description="Styled around what you're already wearing",
    )
    used_ids |= _look_garment_ids(look_a)

    # ── Look B: "Elevated Contrast" ──────────────────────────────────────
    # Full catalog access (not just one swapped category) + contrast-seeking
    # color scoring. This is a distinct STRATEGY, not just a different pool --
    # per the review, that's what makes it feel genuinely different, not just
    # technically non-duplicate.
    look_b = compose_look(
        "B", wardrobe_by_cat, catalog_by_cat, profile,
        catalog_allowed_categories={"top", "bottom", "footwear"},
        excluded_ids=used_ids, prefer_contrast=True,
        archetype="Elevated Contrast",
        archetype_description="A bolder pairing with complementary colors and layering",
    )
    used_ids |= _look_garment_ids(look_b)

    # ── Look C: "Full Style Transformation" ──────────────────────────────
    # Unconstrained, standard scoring, excludes everything A/B already used --
    # the "complete fresh outfit" option.
    look_c = compose_look(
        "C", wardrobe_by_cat, catalog_by_cat, profile,
        catalog_allowed_categories={"top", "bottom", "footwear"},
        excluded_ids=used_ids,
        archetype="Full Style Transformation",
        archetype_description="A complete new pairing to try something different",
    )

    looks = [look_a, look_b, look_c]

    # Final diversity check -- log rather than silently trust the exclusion
    # logic above. If a genuine duplicate still slips through (only possible
    # when the wardrobe+catalog pool is too small to give every look a
    # distinct option), we want that visible in the logs, not hidden.
    signatures = [tuple(sorted(_look_garment_ids(l))) for l in looks]
    if len(set(signatures)) < len(signatures):
        print(f"{tag} WARNING: duplicate look(s) detected even after exclusion -- "
              f"wardrobe+catalog pool is too small to produce 3 distinct looks. "
              f"signatures={signatures}")
    else:
        print(f"{tag} diversity check passed: 3 distinct looks generated")

    styling = _build_styling_payload(profile, looks)

    return {"looks": looks, "styling": styling}


# ─────────────────────────────────────────────────────────────────────────
# New Outfit Suggestions -- a separate, display-only feature from the three
# looks above. Intentionally independent of generate_three_looks()/compose_look()
# and never read by virtual_tryon.py or VirtualTryOn.jsx: Look A above is
# designed to MIRROR the current photo (require_current), which is the
# opposite of what these cards are for. Reuses the same per-garment scoring
# (score_garment_fit, _silhouette_bonus, _body_shape_bonus, _undertone_bonus,
# color_harmony_bonus) so "read the body-shape rule, read the undertone
# colors, then search wardrobe+catalog" is the same underlying mechanism --
# just applied without the require_current lock and without footwear/
# accessories, since these cards show only the core outfit.
# ─────────────────────────────────────────────────────────────────────────

def _detect_outfit_mode(current_outfit_items):
    """
    "dress" if the just-analyzed photo included a dress, else "top_bottom".
    Deliberately simple for this first pass -- a photo with both a dress and
    a separately-detected top (e.g. a cardigan), or with nothing detected at
    all, isn't handled specially yet; it just falls through to "top_bottom".
    """
    for item in current_outfit_items or []:
        if item.get("category") == "dress":
            return "dress"
    return "top_bottom"


def compose_new_suggestion(look_id, candidates_by_cat, profile, mode, excluded_ids):
    """
    One NEW outfit-suggestion card: a single dress (dress mode) or a
    top+bottom pair (top_bottom mode), scored and picked from a candidate
    pool that the caller has already stripped of the current-detected
    garment(s). excluded_ids additionally keeps this card distinct from
    cards already built earlier in this same batch.
    """
    chosen = {}
    rationale = []
    total_score = 0.0

    if mode == "dress":
        best = _best_candidate(candidates_by_cat.get("dress", []), profile, excluded_ids)
        if not best:
            return None
        score, garment, reason = best
        chosen["dress"] = garment
        total_score += score
        rationale.extend(reason)
    else:
        for category in ("top", "bottom"):
            best = _best_candidate(candidates_by_cat.get(category, []), profile, excluded_ids)
            if not best:
                continue
            score, garment, reason = best
            chosen[category] = garment
            total_score += score
            rationale.extend(reason)

        if not chosen:
            return None

        if chosen.get("top") and chosen.get("bottom"):
            harmony = color_harmony_bonus(chosen["top"].get("dominant_hex"), chosen["bottom"].get("dominant_hex"))
            total_score += W_COLOR_HARMONY * harmony
            if harmony > 0:
                rationale.append("Top and bottom colors work well together")

    return {
        "look_id": look_id,
        "dress": chosen.get("dress"),
        "top": chosen.get("top"),
        "bottom": chosen.get("bottom"),
        "score": round(total_score, 3),
        "rationale": rationale[:MAX_RATIONALE_ITEMS],
    }


def generate_new_outfit_suggestions(profile, current_outfit_items, wardrobe_items, catalog_items, request_id=None):
    """
    Three NEW outfit-suggestion cards for display only (no footwear/
    accessories, no try-on wiring). All three share the same garment-type
    mode as whatever the current photo showed (all dresses, or all
    top+bottom), and none of them can be the current-detected garment
    itself -- that's excluded from the candidate pool entirely, not just
    de-prioritized, so every card is a genuinely new suggestion.
    """
    tag = f"[NEW_SUGGESTIONS:{request_id}]" if request_id else "[NEW_SUGGESTIONS]"

    mode = _detect_outfit_mode(current_outfit_items)
    current_ids = {_garment_id(item) for item in (current_outfit_items or [])}
    # _garment_id() alone isn't enough to exclude the current photo's own
    # garment(s) from wardrobe_items: a fresh detection dict has no "id"/
    # "image_hash" yet, so _garment_id() falls back to hashing category+
    # label+dominant_hex -- but by the time this runs, app.py has already
    # persisted that same garment into the closet (add_items_to_closet),
    # and get_user_closet() hands it back with an "image_hash" keyed off
    # the image bytes instead. Those two hashes never match, so id-only
    # exclusion silently lets the just-worn garment back in. Matching on
    # the (category, label, color) content signature instead catches it
    # regardless of which representation it shows up in.
    current_signatures = {
        (item.get("category"), (item.get("label") or "").lower(), (item.get("dominant_hex") or "").lower())
        for item in (current_outfit_items or [])
    }

    # wardrobe history + catalog, minus anything already in the current photo.
    pool_items = [
        item for item in (wardrobe_items or []) + (catalog_items or [])
        if _garment_id(item) not in current_ids
        and (item.get("category"), (item.get("label") or "").lower(), (item.get("dominant_hex") or "").lower())
        not in current_signatures
    ]
    candidates_by_cat = _group_by_category(pool_items)
    print(f"{tag} mode={mode} pool_size={len(pool_items)}")

    looks = []
    used_ids = set()
    for look_id in ("A", "B", "C"):
        look = compose_new_suggestion(look_id, candidates_by_cat, profile, mode, used_ids)
        if not look:
            print(f"{tag} WARNING: no candidate available for look {look_id} (mode={mode})")
            continue
        looks.append(look)
        for category in ("dress", "top", "bottom"):
            garment = look.get(category)
            if garment:
                used_ids.add(_garment_id(garment))

    signatures = [tuple(sorted(
        _garment_id(g) for g in (look.get("dress"), look.get("top"), look.get("bottom")) if g
    )) for look in looks]
    if len(set(signatures)) < len(signatures):
        print(f"{tag} WARNING: duplicate suggestion(s) detected even after exclusion -- "
              f"pool is too small to produce 3 distinct suggestions. signatures={signatures}")
    else:
        print(f"{tag} diversity check passed: {len(looks)} distinct suggestion(s) generated")

    return {"looks": looks, "mode": mode}
