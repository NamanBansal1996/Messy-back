import json
import os

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
            
        # Add more conditions as needed (outfit_fit, color_contrast, waist_emphasis, etc.)
            
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

    # 1. Evaluate Body Rules
    actionable_suggestions = []

    body_rules = body_data.get("rules", [])
    for rule in body_rules:
        if evaluate_condition(rule.get("condition", {}), outfits, gender):
            actionable_suggestions.append(rule.get("suggestion"))
            
    # 2. Evaluate Color Rules
    color_rules = styling_db.get("color_rules", [])
    for rule in color_rules:
        if evaluate_condition(rule.get("condition", {}), outfits, gender):
            actionable_suggestions.append(rule.get("suggestion"))
            
    # Limit suggestions to top 2 so we don't overwhelm the user
    actionable_suggestions = actionable_suggestions[:2]
            
    # 3. Get Face Rules based on Gender
    face_key = face_shape.capitalize() if face_shape else "Oval"
    face_rules = styling_db.get("face_rules", {}).get(face_key, {})
    
    # Get the specific gender rules ("Female" or "Male"). Fallback to female "suggestions" if missing (legacy)
    gender_key = gender.capitalize() if gender else "Female"
    gendered_face_rules = face_rules.get(gender_key, face_rules.get("suggestions", {}))

    # 4. Static reference style guide (Do's/Avoid's per garment category) for this body type + gender
    style_guide = body_data.get("style_guide", {}).get(gender_key, {})

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
