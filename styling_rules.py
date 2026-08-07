import json
import os

from recommendation_engine import _load_undertone_style, _flatten_ideal_colors

def _is_color_bright(color_name):
    if not color_name: return False
    brights = ["red", "orange", "yellow", "pink", "lime", "cyan", "magenta", "light blue", "mint green"]
    return any(b in color_name.lower() for b in brights)

def _is_color_pastel(color_name):
    if not color_name: return False
    pastels = ["light pink", "light blue", "mint", "peach", "lavender", "baby blue"]
    return any(p in color_name.lower() for p in pastels)

def _get_color_family(color_name):
    if not color_name: return "unknown"
    c = color_name.lower()
    if "blue" in c or "navy" in c: return "blue"
    if "red" in c or "burgundy" in c: return "red"
    if "green" in c or "olive" in c: return "green"
    if "pink" in c or "magenta" in c: return "magenta"
    if "purple" in c: return "purple"
    if "yellow" in c or "mustard" in c: return "yellow"
    return "neutral"

def _normalize_type_string(value):
    if not value:
        return ""
    return value.lower().replace("_", " ").replace("-", " ").strip()


def _relevant_style_guide_categories(outfits):
    """
    Which style_guide categories are worth showing given what's actually in
    THIS photo -- e.g. a detected dress has no business showing "Pants"/
    "Jeans" cards. "sleeves" is deliberately never included here: its top
    pick is folded into the generated body-type suggestion sentence instead
    of getting its own detail card (see _generate_body_type_suggestion_body).
    """
    relevant = set()
    if outfits.get("dress"):
        relevant |= {"dresses", "necklines"}
    if outfits.get("top"):
        relevant |= {"shirts", "tops", "tshirts", "necklines"}
    for item in outfits.get("bottom", []):
        label = (item.get("label") or "").lower()
        subcat = (item.get("subcategory") or "").lower()
        if "skirt" in label or "skirt" in subcat:
            relevant.add("skirts")
        elif "jean" in subcat:
            relevant.add("jeans")
        elif subcat:
            relevant.add("pants")
        else:
            # SegFormer's "pants" label covers both jeans and trousers; no
            # classified subcategory to disambiguate yet, so show both.
            relevant |= {"pants", "jeans"}
    return relevant


_CATEGORY_STEMS = {"dresses": "dress", "tops": "top", "shirts": "shirt", "tshirts": "shirt",
                    "pants": "pant", "jeans": "jean", "skirts": "skirt", "necklines": "neck", "sleeves": "sleeve"}

_CATEGORY_NOUNS = {"dresses": "dress", "tops": "top", "shirts": "shirt", "tshirts": "t-shirt",
                   "pants": "pants", "jeans": "jeans", "skirts": "skirt", "necklines": "neckline",
                   "sleeves": "sleeves"}


def _sample_style_items(category_data, n=2):
    if "recommended_items" in category_data:
        return [item["type"].replace("_", " ").title() for item in category_data["recommended_items"][:n]]
    return category_data.get("do", [])[:n]


def _format_item(item_text, category_key):
    """Appends the category noun unless the item text already names itself
    (e.g. "Swing style dresses", "Deep V-neck", "Draped sleeves" all already
    say what they are -- appending "dress"/"neckline"/"sleeves" again would
    read as "Swing style dresses dress"). Checked against real data from
    every body type before landing on this design."""
    stem = _CATEGORY_STEMS.get(category_key, "")
    if stem and stem in item_text.lower():
        return item_text
    noun = _CATEGORY_NOUNS.get(category_key, "")
    return f"{item_text} {noun}".strip()


def _generate_body_type_suggestion_body(style_guide, outfits):
    """
    Builds real, personalized advice text from this body type's own
    style_guide (garment + neckline + top sleeve pick) instead of the
    static sentence in body_rules. Returns None if the data needed isn't
    there (e.g. Male style_guide content doesn't exist yet for any body
    type) so the caller can fall back to the original static text.

    The garment category referenced is whatever's actually detected in the
    photo (reusing the same _relevant_style_guide_categories() the Style
    Guide panel filters by, so the two always agree) -- e.g. someone
    photographed in a shirt and jeans gets top-based advice, not a dress
    suggestion just because dresses happen to be listed first for their
    body type. Falls back to the original dresses-first priority only when
    nothing was actually detected (e.g. a face-only selfie).
    """
    priority = ("dresses", "tops", "shirts", "tshirts")
    relevant = _relevant_style_guide_categories(outfits)
    detected = [c for c in priority if c in relevant and c in style_guide]
    primary = detected[0] if detected else next((c for c in priority if c in style_guide), None)
    necklines = style_guide.get("necklines")
    if not primary or not necklines:
        return None

    garments = [_format_item(i, primary) for i in _sample_style_items(style_guide[primary])]
    necks = [_format_item(i, "necklines") for i in _sample_style_items(necklines)]
    if not garments or not necks:
        return None

    sentence = f"Try {' or '.join(garments)} with {' or '.join(necks)}"

    sleeves = style_guide.get("sleeves")
    if sleeves:
        sleeve_pick = _sample_style_items(sleeves, n=1)
        if sleeve_pick:
            sentence += f" and {_format_item(sleeve_pick[0], 'sleeves')}"

    return sentence + " to bring out your best proportions."


def _generate_contrast_suggestion_body(undertone, default_body):
    """Real colors from the user's own undertone file instead of the generic
    'a complementary color' text. Falls back to the original static body if
    the undertone has no usable color data (e.g. undertone="Unknown")."""
    style = _load_undertone_style(undertone)
    colors = _flatten_ideal_colors(style)
    if len(colors) < 2:
        return default_body
    c1, c2 = colors[0].title(), colors[1].title()
    return (
        f"Your current outfit relies on very similar shades. Try adding a pop of {c1} or {c2} "
        f"— colors that suit your {(undertone or 'neutral').lower()} undertone — for more visual interest."
    )


def _match_current_item_to_style_guide(category_data, outfits):
    """
    category_data: a style_guide category dict in the rich format
    (recommended_items/avoid_items with "type" fields, e.g. inverted_triangle
    Female's "jeans"). Looks across every detected outfit item for one whose
    classified subcategory/fit (garment_classifier.py, when available)
    matches one of this category's item types via normalized substring
    containment -- same style already used by _get_color_family/
    _undertone_bonus elsewhere in this codebase, no rigid label-to-category
    map. Returns a match dict, or None if nothing matched (including when no
    item was classified at all -- e.g. no ANTHROPIC_API_KEY set).
    """
    recommended = category_data.get("recommended_items", [])
    avoid = category_data.get("avoid_items", [])
    if not recommended and not avoid:
        return None

    all_items = []
    for cat_items in outfits.values():
        if isinstance(cat_items, list):
            all_items.extend(cat_items)

    for item in all_items:
        candidates = [c for c in (
            _normalize_type_string(item.get("subcategory")),
            _normalize_type_string(item.get("fit")),
        ) if c]
        if not candidates:
            continue

        for entry in recommended:
            entry_type = _normalize_type_string(entry.get("type"))
            if entry_type and any(entry_type in c or c in entry_type for c in candidates):
                return {"detected_type": entry.get("type"), "verdict": "recommended", "advice": entry.get("advice", "")}

        for entry in avoid:
            entry_type = _normalize_type_string(entry.get("type"))
            if entry_type and any(entry_type in c or c in entry_type for c in candidates):
                return {"detected_type": entry.get("type"), "verdict": "avoid", "advice": entry.get("advice", "")}

    return None


def evaluate_condition(condition, outfits, gender="Unisex"):
    if not condition:
        return True # Empty condition always applies
        
    for key, value in condition.items():
        if key == "gender":
            if value.lower() != "unisex" and value.lower() != gender.lower():
                return False
        elif key == "bottom_type":
            bottoms = outfits.get("bottom", [])
            has_type = any(value.lower() in b.get("label", "").lower() for b in bottoms)
            if not has_type: return False
            
        elif key == "top_color_type":
            tops = outfits.get("top", [])
            has_color_type = False
            for t in tops:
                c_name = t.get("color_name", "")
                if value == "bright" and _is_color_bright(c_name): has_color_type = True
                if value == "vibrant" and _is_color_bright(c_name): has_color_type = True
                if value == "pastel" and _is_color_pastel(c_name): has_color_type = True
            if not has_color_type: return False
            
        elif key == "color_family_group":
            all_items = []
            for cat in outfits.values():
                all_items.extend(cat)
            
            # Check if any item's color family is in the value list
            has_family = any(_get_color_family(i.get("color_name", "")) in [v.lower() for v in value] for i in all_items)
            if not has_family: return False

        elif key == "color_contrast":
            all_items = [i for cat in outfits.values() for i in cat]
            families = [_get_color_family(i.get("color_name", "")) for i in all_items if i.get("color_name")]
            # Can't assess contrast with fewer than 2 colored items, and if
            # the colors span more than one family they're already varied --
            # "low" contrast only means every item reads as the same family.
            if len(families) < 2 or len(set(families)) > 1:
                return False

        # Add more conditions as needed (outfit_fit, waist_emphasis, etc.)

    return True

def get_styling_recommendations(body_type, face_shape, skin_tone, undertone="Neutral", outfits=None, gender="Female"):
    """
    Evaluates dynamic rules from styling_database.json based on current outfits.
    """
    if outfits is None: outfits = {}

    # Load shared JSON database (color_rules + face_rules -- not body-type-specific)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "styling_database.json")

    with open(db_path, "r") as f:
        styling_db = json.load(f)

    # Convert incoming body type to match the per-body-type file name
    body_key = body_type.lower().replace(" ", "_")
    if body_key == "pear": body_key = "triangle"

    # Body-type-specific content (rules + style guide) lives in its own file
    # under styling_data/, one per body type -- keeps future additions (new
    # garment categories, more body types) to small, isolated file diffs.
    body_file = os.path.join(base_dir, "styling_data", f"{body_key}.json")
    body_data = {}
    if os.path.exists(body_file):
        with open(body_file, "r") as f:
            body_data = json.load(f)

    # Gender + the unfiltered style guide are needed up front now -- body-type
    # suggestions (below) generate their text from this body type's own
    # style_guide, not just the display-filtered version built later.
    gender_key = gender.capitalize() if gender else "Female"
    unfiltered_style_guide = body_data.get("style_guide", {}).get(gender_key, {})

    # 1. Evaluate Body Rules -- suggestion body text is generated from real
    # style_guide data (garment + neckline + top sleeve pick) when available,
    # falling back to the static JSON text otherwise (e.g. Male, which has
    # no style_guide content yet for any body type).
    actionable_suggestions = []

    body_rules = body_data.get("rules", [])
    for rule in body_rules:
        if evaluate_condition(rule.get("condition", {}), outfits, gender):
            suggestion = dict(rule.get("suggestion", {}))
            generated_body = _generate_body_type_suggestion_body(unfiltered_style_guide, outfits)
            if generated_body:
                suggestion["body"] = generated_body
            actionable_suggestions.append(suggestion)

    # 2. Evaluate Color Rules -- the rule marked "dynamic_body":
    # "undertone_colors" (currently just "Introduce Some Contrast") gets its
    # body text generated from the user's real undertone colors instead of
    # the generic static text.
    color_rules = styling_db.get("color_rules", [])
    for rule in color_rules:
        if evaluate_condition(rule.get("condition", {}), outfits, gender):
            suggestion = dict(rule.get("suggestion", {}))
            if rule.get("dynamic_body") == "undertone_colors":
                suggestion["body"] = _generate_contrast_suggestion_body(undertone, suggestion.get("body", ""))
            actionable_suggestions.append(suggestion)

    # Limit suggestions to top 2 so we don't overwhelm the user
    actionable_suggestions = actionable_suggestions[:2]

    # 3. Get Face Rules based on Gender
    face_key = face_shape.capitalize() if face_shape else "Oval"
    face_rules = styling_db.get("face_rules", {}).get(face_key, {})
    gendered_face_rules = face_rules.get(gender_key, face_rules.get("suggestions", {}))

    # 4. Style guide, filtered down to categories relevant to what's actually
    # in this photo (e.g. a detected dress won't show Pants/Jeans cards).
    relevant_categories = _relevant_style_guide_categories(outfits)
    style_guide = {
        k: v for k, v in unfiltered_style_guide.items()
        if k == "characteristics" or k in relevant_categories
    }

    # 5. Match the current outfit's classified attributes (garment_classifier.py,
    # when available) against rich-format style_guide categories -- turns
    # static reference content into "your bootcut jeans are a great match"
    # for whichever categories have a detected, classified item.
    for category_data in style_guide.values():
        if not isinstance(category_data, dict):
            continue
        if "recommended_items" not in category_data and "avoid_items" not in category_data:
            continue
        match = _match_current_item_to_style_guide(category_data, outfits)
        if match:
            category_data["matched_current_item"] = match

    return {
        "actionable_suggestions": actionable_suggestions,
        "face_recommendations": gendered_face_rules,
        "style_guide": style_guide
    }
