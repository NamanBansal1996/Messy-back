def get_styling_recommendations(body_type, face_shape, skin_tone):
    """
    Generates styling recommendations based on user attributes.
    """
    
    # ---------------- DATABASE OF RULES ----------------
    
    BODY_RULES = {
        "hourglass": {
            "tops": ["V-neck", "Wrap tops", "Fitted shirts", "Crop tops"],
            "bottoms": ["High-waisted jeans", "Pencil skirts", "Bootcut trousers"],
            "dresses": ["Wrap dresses", "Bodycon", "Fit and flare"],
            "avoid": ["Boxy tunics", "Oversized shapeless clothes"]
        },
        "triangle": { # Pear
            "tops": ["Boat neck", "Cowl neck", "Ruffled sleeves", "Bright colors on top"],
            "bottoms": ["Dark colored bottoms", "Straight leg pants", "A-line skirts"],
            "dresses": ["A-line", "Empire waist"],
            "avoid": ["Skinny jeans with light colors", "Side pockets"]
        },
        "inverted_triangle": {
            "tops": ["V-neck", "Peplum", "Dark colors on top"],
            "bottoms": ["Wide-leg pants", "Full skirts", "Bright colored bottoms", "Cargo pants"],
            "dresses": ["A-line", "Shift dresses"],
            "avoid": ["Shoulder pads", "Boat necks", "Spaghetti straps"]
        },
        "apple": {
            "tops": ["Empire waist", "Tunic tops", "V-neck", "Flowy fabrics"],
            "bottoms": ["Bootcut", "Straight leg", "Mid-rise"],
            "dresses": ["Empire waist", "Shift dresses", "Wrap dresses"],
            "avoid": ["Tight belts", "High-waisted pants", "Clingy fabrics"]
        },
        "rectangle": {
            "tops": ["Ruffled tops", "Off-shoulder", "Scoop neck", "Sweetheart neckline"],
            "bottoms": ["Bootcut jeans", "Flared pants", "Skirts with details"],
            "dresses": ["Belted dresses", "Ruched dresses", "Cut-out dresses"],
            "avoid": ["Square necklines", "Straight shapeless dresses"]
        }
    }

    FACE_RULES = {
        "Round": {
            "necklines": ["V-neck", "Sweetheart", "Scoop"],
            "accessories": ["Long drop earrings", "Rectangular sunglasses"],
            "hair": ["Side part", "Long layers"]
        },
        "Oval": {
            "necklines": ["Any neckline (Scoop, V, Boat)"],
            "accessories": ["Studs", "Hoops", "Any frame sunglasses"],
            "hair": ["Bob", "Waves", "Slicked back"]
        },
        "Square": {
            "necklines": ["Round", "Scoop", "Cowl"],
            "accessories": ["Hoop earrings", "Round sunglasses"],
            "hair": ["Soft waves", "Side swept bangs"]
        },
        "Heart": {
            "necklines": ["Boat neck", "Off-shoulder", "Sweetheart"],
            "accessories": ["Chandelier earrings", "Aviator sunglasses"],
            "hair": ["Chin-length bob", "Side part", "Pixie cut"]
        },
        "Diamond": {
            "necklines": ["V-neck", "Deep scoop", "Strapless"],
            "accessories": ["Studs", "Oversized sunglasses", "Hoops"],
            "hair": ["Short layers", "Fringes", "Up-dos"]
        },
        "Triangle": {
            "necklines": ["Boat neck", "Scoop", "Cowl"],
            "accessories": ["Layered necklaces", "Cat-eye sunglasses"],
            "hair": ["Long layers", "Top knots"]
        },
        "Oblong": {
            "necklines": ["Boat neck", "Turtle neck", "Short necklaces"],
            "accessories": ["Studs", "Large sunglasses"],
            "hair": ["Bangs", "Chin-length bobs", "Waves"]
        }
    }

    SKIN_TONE_RULES = {
        "Fair": {
            "best_colors": ["Emerald Green", "Navy Blue", "Royal Blue", "Burgundy", "Berry", "Black"],
            "neutrals": ["Charcoal", "Silver", "Taupe"],
            "avoid": ["Pale pastels (can wash out)", "Bright yellow"]
        },
        "Medium": {
            "best_colors": ["Mustard", "Warm Green", "Teal", "Coral", "Bronze", "Gold"],
            "neutrals": ["Beige", "Cream", "Tan"],
            "avoid": ["Icy blues", "Silver"]
        },
        "Dark": {
            "best_colors": ["Cobalt Blue", "Fuchsia", "Bright Yellow", "Red", "Orange", "White"],
            "neutrals": ["Black", "Gunmetal", "Copper"],
            "avoid": ["Browns (if too close to skin tone)", "Navy (if too dark)"]
        }
    }
    
    # ---------------- LOGIC ----------------
    
    # Defaults
    b_rec = BODY_RULES.get(body_type, BODY_RULES["rectangle"])
    f_rec = FACE_RULES.get(face_shape, FACE_RULES["Oval"])
    s_rec = SKIN_TONE_RULES.get(skin_tone, SKIN_TONE_RULES["Medium"])
    
    # Construct Visual Prompt
    # E.g. "A fashionable outfit featuring a red blouse and bootcut jeans...",
    
    top_choice = b_rec["tops"][0]
    bottom_choice = b_rec["bottoms"][0]
    color_choice = s_rec["best_colors"][0]
    
    visual_prompt = f"A photorealistic fashion shot of a person with {body_type} body type and {skin_tone} skin tone wearing a {color_choice} {top_choice} and {bottom_choice}, high quality, editorial lighting"

    # Compile Final JSON
    return {
        "clothing_recommendations": {
            "tops": b_rec["tops"],
            "bottoms": b_rec["bottoms"],
            "dresses": b_rec["dresses"],
            "avoid": b_rec["avoid"]
        },
        "face_recommendations": {
            "necklines": f_rec.get("necklines", []),
            "accessories": f_rec.get("accessories", [])
        },
        "color_palette": {
            "best_colors": s_rec["best_colors"],
            "neutrals": s_rec["neutrals"]
        },
        "visual_prompt": visual_prompt
    }
