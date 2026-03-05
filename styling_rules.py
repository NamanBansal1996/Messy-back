def get_styling_recommendations(body_type, face_shape, skin_tone, undertone="Neutral"):
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

    # ---------------- HUB-BAND FILTERING SYSTEM ----------------
    
    # Mathematical Hue Bands (0-360) for HSV color space
    WARM_BAND = [
        (0, 50, "Red/Rust/Coral/Orange"),
        (40, 80, "Mustard/Gold/Yellow"),
        (80, 150, "Olive/Warm Green")
    ]
    COOL_BAND = [
        (180, 240, "Navy/Royal/Blue"),
        (140, 180, "Emerald/Teal/Cool Green"),
        (260, 320, "Purple/Plum")
    ]
    NEUTRAL_BAND = [
        (0, 30, "Soft Red"), 
        (70, 130, "Muted Green"), 
        (200, 250, "Classic Blue")
    ]
    
    # Select the allowed color palette based primarily on Undertone
    if undertone == "Warm":
        best_colors = [item[2] for item in WARM_BAND]
        neutrals = ["Camel", "Tan", "Olive", "Warm Brown", "Cream"]
    elif undertone == "Cool":
        best_colors = [item[2] for item in COOL_BAND]
        neutrals = ["Charcoal", "Navy", "Silver", "Stark White", "Cool Gray"]
    else:
        best_colors = [item[2] for item in NEUTRAL_BAND]
        neutrals = ["Beige", "Taupe", "Off-White", "Charcoal"]
        
    # We still use skin_tone (light/medium/deep) to adjust contrast/intensity
    if skin_tone == "Fair":
        avoid_colors = ["Pale pastels (can wash out)", "Harsh Neon"]
    elif skin_tone == "Dark":
        avoid_colors = ["Browns (if too close to skin)", "Muddy tones"]
    else:
        avoid_colors = ["Icy blues", "Silver (if too harsh)"]
        
    # Defaults from original logic
    b_rec = BODY_RULES.get(body_type, BODY_RULES["rectangle"])
    f_rec = FACE_RULES.get(face_shape, FACE_RULES["Oval"])

    # Construct Visual Prompt
    top_choice = b_rec["tops"][0]
    bottom_choice = b_rec["bottoms"][0]
    color_choice = best_colors[0] if best_colors else "Neutral"
    
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
            "best_colors": best_colors,
            "neutrals": neutrals,
            "avoid": avoid_colors
        },
        "visual_prompt": visual_prompt
    }
