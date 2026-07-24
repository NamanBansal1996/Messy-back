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

MAX_RATIONALE_ITEMS = 4

# ─────────────────────────────────────────────────────────────────────────
# Undertone color preferences -- the concrete fix for the "computed then
# discarded" undertone bug flagged in the audit.
# ─────────────────────────────────────────────────────────────────────────
UNDERTONE_PALETTES = {
    "Warm": ["Olive", "Mustard", "Brown", "Rust", "Beige"],
    "Cool": ["Navy", "Navy Blue", "Burgundy", "Emerald", "Purple"],
    "Neutral": ["Olive", "Navy", "Burgundy", "Emerald", "Brown"],
}

# ─────────────────────────────────────────────────────────────────────────
# Body-shape hints. Two things live here:
#   1. LABEL preferences, used for real per-garment scoring (only works
#      with the actual label vocabulary the detector can produce today).
#   2. CUT descriptions, used only for the "styling" text payload sent to
#      SuggestionFlow.jsx -- these are display strings, not scoring inputs.
# Covers all 7 body types classify_body_type_v2() can output, including
# "spoon" and "diamond", which styling_database.json is currently missing
# (a separate, earlier audit finding -- filled here as a side benefit).
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
    "spoon": {
        "top": {"prefer": ["jacket", "sweater"], "avoid": []},
        "bottom": {"prefer": ["trousers"], "avoid": ["skirt"]},
    },
    "diamond": {
        "top": {"prefer": ["shirt", "dress"], "avoid": []},
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
    "spoon": {
        "tops": ["Structured jacket", "V-neck top"],
        "bottoms": ["Straight-leg trousers"],
        "dresses": ["Fit-and-flare dress"],
    },
    "diamond": {
        "tops": ["Tailored shirt", "V-neck top"],
        "bottoms": ["Straight-leg trousers"],
        "dresses": ["Wrap dress"],
    },
}


def _normalize_body_key(body_type):
    """Same remap styling_rules.py already applies, kept consistent here."""
    key = (body_type or "").lower().replace(" ", "_")
    if key == "pear":
        key = "triangle"
    return key


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


def color_harmony_bonus(hex_a, hex_b):
    """
    +1.0  complementary pairing (hues roughly opposite)
    +0.5  tonal/monochrome pairing (hues close together)
    +1.0  either garment is a neutral (white/gray/black) -- pairs with anything
    -0.2  hues clash (neither close nor complementary)
     0.0  can't determine (missing/invalid hex)
    """
    hsv_a = _hex_to_hsv(hex_a)
    hsv_b = _hex_to_hsv(hex_b)
    if not hsv_a or not hsv_b:
        return 0.0

    h_a, s_a, _ = hsv_a
    h_b, s_b, _ = hsv_b

    if s_a < 0.15 or s_b < 0.15:
        return 1.0

    diff = abs(h_a - h_b) * 360
    diff = min(diff, 360 - diff)

    if diff <= 30:
        return 0.5
    if 150 <= diff <= 210:
        return 1.0
    return -0.2


# ─────────────────────────────────────────────────────────────────────────
# Per-garment scoring
# ─────────────────────────────────────────────────────────────────────────

def _body_shape_bonus(garment, body_type):
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
    palette = UNDERTONE_PALETTES.get(undertone)
    if not palette:
        return 0.0, None

    color_name = _color_name_of(garment)
    if not color_name:
        return 0.0, None

    if any(p.lower() in color_name.lower() for p in palette):
        return 1.0, f"{color_name} complements your {undertone.lower()} undertone"

    opposite = None
    if undertone == "Warm":
        opposite = UNDERTONE_PALETTES["Cool"]
    elif undertone == "Cool":
        opposite = UNDERTONE_PALETTES["Warm"]

    if opposite and any(p.lower() in color_name.lower() for p in opposite):
        return -0.5, None

    return 0.0, None


def _weather_bonus(garment, condition):
    """
    Deliberately conservative: the detector's label vocabulary has no
    fabric/material attribute (e.g. no "suede" tag exists anywhere in the
    pipeline), so the "avoid suede in rain" rule from the spec can only be
    partially honored -- footwear gets a small positive nudge in rain
    rather than a confident "waterproof" claim, since we can't actually
    tell the difference with current data.
    """
    label = (garment.get("label") or "").lower()

    if condition == "hot":
        if label in {"tshirt", "short sleeve shirt", "shirt"}:
            return 1.0, "Breathable choice for today's warm weather"
        if label in {"jacket", "sweater"}:
            return -1.0, None

    elif condition == "cold":
        if label in {"jacket", "sweater", "long sleeve shirt"}:
            return 1.0, "Warm layer for today's cold weather"
        if label in {"short sleeve shirt", "tshirt"}:
            return -0.5, None

    elif condition == "rain":
        if garment.get("category") == "footwear":
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
    w_score, w_reason = _weather_bonus(garment, weather.get("condition"))
    score += W_WEATHER * w_score
    if w_reason:
        rationale.append(w_reason)

    return score, rationale


# ─────────────────────────────────────────────────────────────────────────
# Outfit composition
# ─────────────────────────────────────────────────────────────────────────

def _group_by_category(items):
    grouped = {}
    for item in items or []:
        grouped.setdefault(item.get("category"), []).append(item)
    return grouped


def _best_candidate(candidates, profile):
    """candidates: list of garment dicts. Returns (score, garment, rationale) or None."""
    if not candidates:
        return None
    scored = [(*score_garment_fit(g, profile), g) for g in candidates]
    # scored items are (score, rationale, garment) -- resort by score desc
    scored.sort(key=lambda t: t[0], reverse=True)
    score, rationale, garment = scored[0]
    return score, garment, rationale


def compose_look(look_id, wardrobe_by_cat, catalog_by_cat, profile, catalog_allowed_categories):
    """
    catalog_allowed_categories: set of categories permitted to pull from the
    catalog. Categories NOT in this set still fall back to catalog if the
    wardrobe has zero items there -- an empty wardrobe slot is not a reason
    to leave a look incomplete.
    """
    chosen = {}
    total_score = 0.0
    rationale = []
    wardrobe_count = 0
    total_count = 0

    for category in ("top", "bottom", "footwear"):
        candidates = list(wardrobe_by_cat.get(category, []))
        wardrobe_empty_here = len(candidates) == 0

        if category in catalog_allowed_categories or wardrobe_empty_here:
            candidates += catalog_by_cat.get(category, [])

        best = _best_candidate(candidates, profile)
        if not best:
            continue

        score, garment, reason = best
        chosen[category] = garment
        total_score += score
        rationale.extend(reason)
        total_count += 1
        if garment.get("source") == "wardrobe":
            wardrobe_count += 1

    # Accessories: wardrobe-only for MVP (no accessory items exist in the
    # catalog seed data today -- Ads.jsx never had any). Up to 2 items.
    acc_candidates = wardrobe_by_cat.get("accessories", [])
    acc_scored = sorted(
        (score_garment_fit(g, profile) + (g,) for g in acc_candidates),
        key=lambda t: t[0],
        reverse=True,
    )
    accessories = [g for _, _, g in acc_scored[:2]]

    if chosen.get("top") and chosen.get("bottom"):
        harmony = color_harmony_bonus(
            chosen["top"].get("dominant_hex"), chosen["bottom"].get("dominant_hex")
        )
        total_score += W_COLOR_HARMONY * harmony
        if harmony > 0:
            rationale.append("Top and bottom colors work well together")

    wardrobe_ratio = round(wardrobe_count / total_count, 2) if total_count else 0.0

    return {
        "look_id": look_id,
        "top": chosen.get("top"),
        "bottom": chosen.get("bottom"),
        "footwear": chosen.get("footwear"),
        "accessories": accessories,
        "wardrobe_ratio": wardrobe_ratio,
        "score": round(total_score, 3),
        "rationale": rationale[:MAX_RATIONALE_ITEMS],
    }


def _best_catalog_swap_category(wardrobe_by_cat, catalog_by_cat, profile, look_a):
    """
    Picks the single category where swapping to a catalog item improves the
    score the most -- this is what makes Look B a genuine "mix," not an
    arbitrary variation. Categories where Look A already had to use the
    catalog (empty wardrobe slot) are skipped since there's nothing to swap.
    """
    best_category = None
    best_delta = 0.0

    for category in ("top", "bottom", "footwear"):
        current = look_a.get(category)
        if not current or current.get("source") != "wardrobe":
            continue  # already catalog, or no pick at all -- nothing to swap

        catalog_candidates = catalog_by_cat.get(category, [])
        if not catalog_candidates:
            continue

        current_score, _ = score_garment_fit(current, profile)
        best_catalog = _best_candidate(catalog_candidates, profile)
        if not best_catalog:
            continue
        catalog_score, _, _ = best_catalog

        delta = catalog_score - current_score
        if delta > best_delta:
            best_delta = delta
            best_category = category

    return best_category


def _build_styling_payload(profile, looks):
    """
    Shaped specifically to match what SuggestionFlow.jsx already expects
    (analysisData.styling.{clothing_recommendations, color_palette,
    visual_prompt}) -- per the audit, that UI exists today and silently
    falls back to generic text because the backend never populated this.
    """
    body_key = _normalize_body_key(profile.get("body_type"))
    cuts = BODY_CUT_HINTS.get(body_key, BODY_CUT_HINTS["rectangle"])

    undertone = profile.get("undertone")
    best_colors = UNDERTONE_PALETTES.get(undertone, UNDERTONE_PALETTES["Neutral"])
    if undertone == "Warm":
        avoid_colors = UNDERTONE_PALETTES["Cool"]
    elif undertone == "Cool":
        avoid_colors = UNDERTONE_PALETTES["Warm"]
    else:
        avoid_colors = []

    best_look = max(looks, key=lambda l: l["score"]) if looks else None
    top_label = best_look["top"]["label"] if best_look and best_look.get("top") else "a top"
    bottom_label = best_look["bottom"]["label"] if best_look and best_look.get("bottom") else "a bottom"
    gender = (profile.get("gender") or "").lower() or "your"

    visual_prompt = (
        f"A {gender} look for a {profile.get('body_type', 'balanced')} silhouette "
        f"with {(undertone or 'neutral').lower()} undertones, pairing {top_label} with {bottom_label}."
    )

    return {
        "clothing_recommendations": cuts,
        "color_palette": {"best_colors": best_colors, "avoid_colors": avoid_colors},
        "visual_prompt": visual_prompt,
    }


def generate_three_looks(profile, wardrobe_items, catalog_items):
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
    wardrobe_items: list of garment dicts from closet_manager.get_user_closet()
    catalog_items:  list of garment dicts from catalog.get_catalog_items(...),
                     each tagged with "source": "catalog"

    Returns: {
        "looks": [look_a, look_b, look_c],
        "styling": {...}   # matches SuggestionFlow.jsx's expected shape
    }
    """
    # Wardrobe items don't carry a "source" field today -- tag them here
    # rather than modifying closet_manager.py.
    for item in wardrobe_items or []:
        item.setdefault("source", "wardrobe")

    wardrobe_by_cat = _group_by_category(wardrobe_items)
    catalog_by_cat = _group_by_category(catalog_items)

    look_a = compose_look(
        "A", wardrobe_by_cat, catalog_by_cat, profile, catalog_allowed_categories=set()
    )

    swap_category = _best_catalog_swap_category(wardrobe_by_cat, catalog_by_cat, profile, look_a)
    look_b = compose_look(
        "B", wardrobe_by_cat, catalog_by_cat, profile,
        catalog_allowed_categories={swap_category} if swap_category else set(),
    )

    look_c = compose_look(
        "C", wardrobe_by_cat, catalog_by_cat, profile,
        catalog_allowed_categories={"top", "bottom", "footwear"},
    )

    styling = _build_styling_payload(profile, [look_a, look_b, look_c])

    return {"looks": [look_a, look_b, look_c], "styling": styling}
