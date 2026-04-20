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
    
    # Load JSON Database
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "styling_database.json")
    
    with open(db_path, "r") as f:
        styling_db = json.load(f)

    # Convert incoming body type to match JSON keys
    body_key = body_type.lower()
    if body_key == "pear": body_key = "triangle"
    
    # 1. Evaluate Body Rules
    actionable_suggestions = []
    
    body_rules = styling_db.get("body_rules", {}).get(body_key, {}).get("rules", [])
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

    return {
        "actionable_suggestions": actionable_suggestions,
        "face_recommendations": gendered_face_rules
    }
