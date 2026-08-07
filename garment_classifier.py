"""
garment_classifier.py

Fine-grained garment attribute classification via Claude Vision -- enriches
each detected item (top/bottom/dress only, see CLASSIFIABLE_CATEGORIES) with
subcategory/fit/rise/pattern/formality, on top of the coarse category
SegFormer already provides (yolo_outfit_detect.py).

Optional and additive: with no ANTHROPIC_API_KEY set (or the feature
explicitly disabled), every item passes through unchanged -- no network
calls, no crash. Every detected garment already carries a base64 PNG crop
(item["image"], set in yolo_outfit_detect.py) which is exactly what this
module needs; no new cropping logic required.
"""

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

# Swappable without a code change -- see the model-ID mistake this was added
# to prevent (a stale/invalid model string should be a config fix, not a
# redeploy). Confirmed directly against a real API key: claude-3-5-haiku-20241022
# and claude-3-5-haiku-latest both return HTTP 404 (Anthropic's own deprecation
# notice says that model line reached end-of-life Feb 19, 2026); this one
# returned a correct classification in the same test.
DEFAULT_MODEL = os.environ.get("GARMENT_CLASSIFIER_MODEL", "claude-haiku-4-5-20251001")

# Only these broad categories get classified. No styling_data/*.json
# style_guide (as of this session) references footwear or accessories, so
# classifying belts/bags/scarves/shoes would spend real money on data
# nothing downstream consumes.
CLASSIFIABLE_CATEGORIES = {"top", "bottom", "dress"}

_RISE_VALUES = {"high_rise", "mid_rise", "low_rise"}
_PATTERN_VALUES = {"solid", "striped", "plaid", "floral", "animal_print", "checkered", "other"}
_FORMALITY_VALUES = {"casual", "smart_casual", "business_casual", "formal"}

_MAX_FREE_TEXT_LEN = 80
_MAX_DETAILS_LEN = 300

# In-memory only, per process lifetime -- avoids re-billing if the same
# photo gets analyzed twice in one session. Not persisted to disk; historical
# wardrobe items are never classified in the first place (see
# CLASSIFIABLE_CATEGORIES usage in enrich_outfits_with_attributes), so
# there's no long-lived cache to maintain here.
_cache = {}

_PROMPT_TEMPLATE = """You are a fashion expert analyzing a single cropped clothing item photo.
The item's broad category is: {category}.

Respond with ONLY a raw JSON object (no markdown, no prose, no code fences) with exactly these fields:
{{
  "subcategory": "specific cut/style name, lowercase_with_underscores, e.g. bootcut_jeans, wrap_top, a_line_skirt",
  "fit": "how it fits, lowercase_with_underscores, e.g. slim, oversized, tailored, flared",
  "rise": "one of: high_rise, mid_rise, low_rise -- or null if not applicable to this garment",
  "pattern": "one of: solid, striped, plaid, floral, animal_print, checkered, other",
  "formality": "one of: casual, smart_casual, business_casual, formal",
  "details": "one short sentence describing the garment, max ~30 words"
}}

Use null for any field that genuinely doesn't apply to this garment (e.g. rise for a top)."""


def _get_client_and_model(model=None):
    """Returns (client, resolved_model) or (None, None) if the feature is
    unavailable/disabled -- callers treat that as "skip, no API call"."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, None

    enabled = os.environ.get("ENABLE_GARMENT_CLASSIFICATION", "true").strip().lower()
    if enabled in ("false", "0", "no"):
        return None, None

    try:
        import anthropic  # local import: a missing/broken package should never break the rest of the app
    except ImportError:
        return None, None

    return anthropic.Anthropic(api_key=api_key), (model or DEFAULT_MODEL)


def _normalize_token(value):
    """Lowercase, underscore-joined, alnum-only -- or None if not a usable string."""
    if not isinstance(value, str):
        return None
    value = value.strip().lower().replace(" ", "_").replace("-", "_")
    value = re.sub(r"[^a-z0-9_]", "", value)
    if not value or len(value) > _MAX_FREE_TEXT_LEN:
        return None
    return value


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def classify_garment_attributes(image_b64, category, model=None):
    """
    Classifies a single cropped garment image (base64 PNG) into fine-grained
    attributes. Returns a dict of validated fields, or None on any failure
    (missing/disabled API key, network error, timeout, malformed response)
    -- never raises, so callers never need their own try/except.
    """
    if not image_b64:
        return None

    client, resolved_model = _get_client_and_model(model)
    if client is None:
        return None

    image_hash = hashlib.md5(image_b64.encode("utf-8")).hexdigest()
    if image_hash in _cache:
        return _cache[image_hash]

    try:
        response = client.messages.create(
            model=resolved_model,
            max_tokens=300,
            timeout=10.0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
                    },
                    {"type": "text", "text": _PROMPT_TEMPLATE.format(category=category)},
                ],
            }],
        )
        raw_text = response.content[0].text
        parsed = _extract_json(raw_text)
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    result = {}

    subcategory = _normalize_token(parsed.get("subcategory"))
    if subcategory:
        result["subcategory"] = subcategory

    fit = _normalize_token(parsed.get("fit"))
    if fit:
        result["fit"] = fit

    rise = _normalize_token(parsed.get("rise"))
    if rise in _RISE_VALUES:
        result["rise"] = rise

    pattern = _normalize_token(parsed.get("pattern"))
    if pattern in _PATTERN_VALUES:
        result["pattern"] = pattern

    formality = _normalize_token(parsed.get("formality"))
    if formality in _FORMALITY_VALUES:
        result["formality"] = formality

    details = parsed.get("details")
    if isinstance(details, str) and details.strip():
        result["details"] = details.strip()[:_MAX_DETAILS_LEN]

    if not result:
        return None

    _cache[image_hash] = result
    return result


def enrich_outfits_with_attributes(outfits, timeout=5.0, model=None):
    """
    For every item in outfits[category] where category is in
    CLASSIFIABLE_CATEGORIES, classifies it via Claude Vision (in parallel)
    and attaches the resulting attributes onto the item dict in place.
    Bounded by `timeout` seconds total, not per item -- whichever calls
    haven't finished by then are simply left without attributes, same as
    a missing API key or any other failure. Footwear/accessories items are
    never touched. Always returns outfits in the same shape it was given;
    never raises.
    """
    if not outfits:
        return outfits

    jobs = []
    for category, items in outfits.items():
        if category not in CLASSIFIABLE_CATEGORIES:
            continue
        for item in items or []:
            if item.get("image"):
                jobs.append((item, item["image"], category))

    if not jobs:
        return outfits

    executor = ThreadPoolExecutor(max_workers=len(jobs))
    try:
        future_to_item = {
            executor.submit(classify_garment_attributes, image_b64, category, model): item
            for item, image_b64, category in jobs
        }
        try:
            for future in as_completed(future_to_item, timeout=timeout):
                item = future_to_item[future]
                try:
                    attributes = future.result()
                except Exception:
                    attributes = None
                if attributes:
                    item.update(attributes)
        except FuturesTimeoutError:
            pass  # whichever calls didn't finish in time just don't get attributes
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return outfits
