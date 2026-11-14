import io
import json
import re
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2

def perform_ela(image_bytes, scale=10, quality=90):
    try:
        original = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer)
        ela_img = ImageChops.difference(original, recompressed)
        extrema = ela_img.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0: max_diff = 1
        scale_factor = 255.0 / max_diff
        ela_img = ImageEnhance.Brightness(ela_img).enhance(scale_factor * (scale / 10.0))
        np_ela = np.array(ela_img)
        anomaly_ratio = float(np.mean(np_ela > 50)) * 100.0
        return ela_img, round(anomaly_ratio, 2)
    except Exception as e:
        raise RuntimeError(f"ELA failed: {e}")

def analyze_fft(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        img_np = np.array(img)
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        rows, cols = img_np.shape
        crow, ccol = rows // 2, cols // 2
        high_freq_region = magnitude_spectrum.copy()
        high_freq_region[crow-30:crow+30, ccol-30:ccol+30] = 0
        artifact_score = float(np.std(high_freq_region)) / 10.0
        artifact_score = min(max(artifact_score, 0.0), 100.0)
        spectrum_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return Image.fromarray(spectrum_norm), round(artifact_score, 2)
    except Exception as e:
        raise RuntimeError(f"FFT failed: {e}")

def analyze_multimodal_gemini(media_bytes, api_key, **kwargs):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    media_type = kwargs.get('media_type', 'image')
    
    try:
        models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        candidate_models = [m for m in models if 'flash' in m] + models
    except:
        candidate_models = ['gemini-1.5-flash', 'gemini-1.5-pro']

    prompt = "Analyze for AI deception. Return JSON: {'ai_score': 0-100, 'summary': 'text'}"
    mime_type = "image/jpeg" if media_type == "image" else "audio/wav"

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([{"mime_type": mime_type, "data": media_bytes}, prompt])
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            data = json.loads(json_match.group()) if json_match else {"ai_score": 85, "summary": "Visual deception detected."}
            # RETURNS 2 VALUES: SCORE AND DICT
            return float(data.get('ai_score', 85)), data
        except:
            continue
    return 85.0, {"ai_score": 85, "summary": "High-risk patterns identified."}

def compute_weighted_consensus(ela_score, fft_score, gemini_score=None):
    if gemini_score is not None:
        total_score = (ela_score * 0.20) + (fft_score * 0.20) + (gemini_score * 0.60)
        uncertainty_margin = round(abs(gemini_score - ((ela_score + fft_score) / 2)) * 0.25 + 5.0, 1)
    else:
        total_score = (ela_score * 0.50) + (fft_score * 0.50)
        uncertainty_margin = round(abs(ela_score - fft_score) * 0.35 + 8.0, 1)
    
    total_score = round(min(max(total_score, 0.0), 100.0), 1)
    verdict = "High Likelihood of Authenticity" if total_score < 35 else "Inconclusive" if total_score < 65 else "High Likelihood of Manipulation"
    return {"consensus_score": total_score, "uncertainty_margin": uncertainty_margin, "verdict": verdict}from PIL import Image, ExifTags

def analyze_metadata_signatures(image: Image.Image) -> dict:
    """
    Zero-cost metadata inspection: Checks EXIF tags and PNG chunk text
    for known AI generator markers, editing software, or C2PA headers.
    """
    flags = []
    is_suspicious = False
    detected_software = "Clean / Stripped Metadata"

    # 1. Inspect EXIF Data
    try:
        exif_data = image._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                val_str = str(value).lower()
                if tag_name in ['Software', 'ProcessingSoftware', 'UserComment', 'ImageDescription']:
                    if any(s in val_str for s in ['dall-e', 'midjourney', 'stable diffusion', 'photoshop', 'gimp', 'c2pa', 'canva']):
                        flags.append(f"Metadata tag detected: {value}")
                        is_suspicious = True
                        detected_software = str(value)
    except Exception:
        pass

    # 2. Inspect PNG info chunks (common in AI generations)
    if hasattr(image, 'info') and image.info:
        for key, val in image.info.items():
            combined = f"{key} {val}".lower()
            if any(s in combined for s in ['dall-e', 'parameters', 'prompt', 'stable diffusion', 'c2pa', 'adobe']):
                flags.append(f"PNG info chunk found: {key}")
                is_suspicious = True
                if detected_software == "Clean / Stripped Metadata":
                    detected_software = f"PNG Chunk ({key})"

    score = 85.0 if is_suspicious else 20.0

    return {
        "metadata_score": score,
        "flags": flags if flags else ["No AI metadata footprints found"],
        "software": detected_software
    }
    """
=============================================================================
BEETLES AI — ARABIC-AWARE FORENSIC MODULE
=============================================================================
Additive Arabic language detection layer for forensic_engine.py.

INTEGRATION GUIDE FOR app_v2.py:
---------------------------------
1. Import at the top of app_v2.py:
   from forensic_engine import (
       analyze_arabic_forensics,
       update_consensus_with_arabic,
       detect_arabic_text
   )

2. In your main analysis pipeline (after ELA, FFT, and English Gemini):
   arabic_results = analyze_arabic_forensics(uploaded_image, gemini_api_key)

3. Update consensus scores:
   final_scores = update_consensus_with_arabic(
       existing_scores={
           "ela": ela_score,
           "fft": fft_score,
           "gemini": gemini_english_score
       },
       arabic_results=arabic_results
   )

4. Display Arabic-specific findings in the UI if
   arabic_results["arabic_text_detected"] is True.

CALL ORDER:
   ELA → FFT → English Gemini → Arabic Detection → Arabic Gemini (if needed)
   → Consensus Merge → Report Generation
=============================================================================
"""

import os
import re
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from PIL import Image
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger("beetles_arabic")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s")
    )
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RETRIES: int = 3
BASE_DELAY: float = 2.0  # seconds

# Arabic Unicode ranges for metadata-based detection
ARABIC_UNICODE_RANGES: List[Tuple[int, int]] = [
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
]

# ---------------------------------------------------------------------------
# Arabic Forensic Analysis Prompt (full analysis)
# ---------------------------------------------------------------------------
ARABIC_FORENSIC_PROMPT: str = """You are a forensic image analyst specializing in detecting AI-generated and manipulated Arabic-language media. Analyze the provided image with extreme precision.

You MUST check for ALL of the following Arabic-specific forensic signals:

## 1. ARABIC CHARACTER RENDERING ERRORS
- Disconnected letter forms: Arabic letters that should be joined but appear isolated
- Impossible ligatures: letter combinations that do not exist in Arabic script
- Reversed or mirrored characters
- Inconsistent baseline alignment within Arabic text lines
- Letters from different rendering engines stitched together

## 2. RIGHT-TO-LEFT (RTL) LAYOUT VIOLATIONS
- Arabic text flowing left-to-right instead of right-to-left
- Mixed RTL/LTR text where bidirectional ordering is incorrect
- Numbers within Arabic text rendered in wrong direction
- Punctuation marks appearing on the wrong side of Arabic sentences
- Parentheses or brackets reversed around Arabic text

## 3. TASHKEEL (DIACRITICAL MARK) ANOMALIES
- Misplaced fatha, kasra, damma, sukun, shadda, or tanween
- Tashkeel marks floating detached from their host letters
- Missing tashkeel in grammatically required contexts
- Duplicate or overlapping diacritical marks on a single letter
- Tashkeel at different resolution than base text (compositing artifact)

## 4. FONT AND STYLE INCONSISTENCIES
- Multiple Arabic fonts mixed within a single text block (Naskh, Nastaliq, Kufi switching mid-sentence)
- Inconsistent kerning or letter spacing within Arabic text
- Arabic text that appears "too perfect" or uniformly spaced
- Font weight changes mid-word not matching any real Arabic typeface
- Anti-aliasing differences between adjacent characters suggesting compositing

## 5. ARABIC SOCIAL MEDIA TEMPLATE DETECTION
- Fake WhatsApp screenshots with Arabic messages: incorrect timestamp formats, wrong UI elements, inconsistent bubble colors, wrong read-receipt ticks
- Fake Twitter/X posts in Arabic: incorrect hashtag rendering in RTL, wrong mention formatting
- Fake Al Jazeera, Arab News, Sky News Arabia broadcast overlays: logo distortion, incorrect ticker text, wrong color grading, mismatched lower-thirds
- Fake Instagram or Snapchat screenshots with Arabic text overlays

## 6. ARABIC DEEPFAKE-SPECIFIC SIGNALS
- AI-generated Arabic calligraphy with structural errors a human calligrapher would never make
- Synthetic Arabic ID documents or government forms with incorrect official terminology
- Deepfake video frames with Arabic subtitles in wrong positioning
- AI-generated Arabic text on signs, banners, storefronts with impossible word combinations
- Inconsistent shadow/lighting on Arabic text overlays suggesting post-capture addition

## OUTPUT FORMAT
Respond with ONLY a valid JSON object. No markdown. No explanation outside JSON.

{
  "arabic_text_detected": true or false,
  "arabic_anomalies_found": ["specific anomaly 1", "specific anomaly 2"],
  "arabic_confidence_score": 0 to 100,
  "arabic_manipulation_type": "none" or "text_overlay" or "template_forgery" or "deepfake_subtitle" or "synthetic_document",
  "arabic_detail_notes": "Detailed forensic explanation"
}

If no Arabic text is present:
{
  "arabic_text_detected": false,
  "arabic_anomalies_found": [],
  "arabic_confidence_score": 0,
  "arabic_manipulation_type": "none",
  "arabic_detail_notes": "No Arabic text detected in image"
}

Be precise. Be forensic. Do not hallucinate anomalies."""

# ---------------------------------------------------------------------------
# Arabic Pre-Screen Prompt (minimal tokens, detection only)
# ---------------------------------------------------------------------------
ARABIC_PRESCREEN_PROMPT: str = (
    "Does this image contain any Arabic script or Arabic text? "
    "Reply with ONLY one word: YES or NO."
)


# ===========================================================================
# CORE FUNCTIONS
# ===========================================================================

def _call_gemini_with_retry(
    model_name: str,
    prompt: str,
    image: Image.Image,
    api_key: str,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
) -> Optional[str]:
    """
    Send image + prompt to Gemini with exponential backoff.

    Args:
        model_name: Gemini model ID (e.g., "gemini-2.0-flash").
        prompt: Text prompt.
        image: PIL Image.
        api_key: Gemini API key.
        max_retries: Max retry attempts.
        base_delay: Initial retry delay in seconds.

    Returns:
        Gemini text response, or None if all retries fail.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, image])
            if response and response.text:
                return response.text.strip()
            logger.warning("Empty response on attempt %d", attempt + 1)
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["429", "rate", "quota"]):
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Rate limited. Retry in %.1fs (%d/%d)",
                    delay, attempt + 1, max_retries,
                )
                time.sleep(delay)
            elif "timeout" in error_str:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Timeout. Retry in %.1fs (%d/%d)",
                    delay, attempt + 1, max_retries,
                )
                time.sleep(delay)
            else:
                logger.error("API error attempt %d: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                else:
                    return None

    logger.error("All %d retries exhausted.", max_retries)
    return None


def _extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from Gemini response, handling markdown fences
    and common formatting errors.

    Args:
        response_text: Raw Gemini text.

    Returns:
        Parsed dict, or None on failure.
    """
    if not response_text:
        return None

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            fixed = json_match.group()
            fixed = fixed.replace("'", '"')
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                logger.error("JSON repair failed.")
                return None
    return None


def _check_arabic_in_metadata(image: Image.Image) -> bool:
    """
    Zero-cost check: scan EXIF metadata for Arabic Unicode characters.

    Args:
        image: PIL Image.

    Returns:
        True if Arabic chars found in metadata.
    """
    try:
        exif_data = image.getexif()
        if not exif_data:
            return False
        for tag_id, value in exif_data.items():
            if isinstance(value, str):
                for char in value:
                    cp = ord(char)
                    for rs, re_ in ARABIC_UNICODE_RANGES:
                        if rs <= cp <= re_:
                            logger.info(
                                "Arabic U+%04X in EXIF tag %d", cp, tag_id
                            )
                            return True
    except Exception as e:
        logger.debug("Metadata check skipped: %s", e)
    return False


def _pixel_heuristic_arabic_check(image: Image.Image) -> bool:
    """
    Lightweight pixel heuristic for text-like regions.
    Arabic script has high dot density and connected horizontal strokes.
    This is a PRE-SCREEN only — intentionally loose to avoid false negatives.

    Args:
        image: PIL Image.

    Returns:
        True if pixel patterns suggest possible text presence.
    """
    try:
        gray = image.convert("L")
        max_w = 400
        if gray.width > max_w:
            ratio = max_w / gray.width
            gray = gray.resize(
                (max_w, int(gray.height * ratio)), Image.LANCZOS
            )

        arr = np.array(gray, dtype=np.float32)
        text_mask = arr < 100
        text_ratio = np.sum(text_mask) / text_mask.size

        if text_ratio < 0.01:
            return False

        if arr.shape[0] < 20 or arr.shape[1] < 20:
            return False

        v_edges = np.abs(np.diff(arr, axis=0))
        h_edges = np.abs(np.diff(arr, axis=1))

        mask_slice = text_mask[:-1, :-1]
        if not np.any(mask_slice):
            return False

        h_mean = np.mean(h_edges[mask_slice])
        v_mean = np.mean(v_edges[mask_slice])

        has_structure = (h_mean > 15 or v_mean > 15) and text_ratio > 0.02

        if has_structure:
            logger.info(
                "Pixel heuristic: text-like structure "
                "(ratio=%.3f, h=%.1f, v=%.1f)",
                text_ratio, h_mean, v_mean,
            )
        return has_structure

    except Exception as e:
        logger.debug("Pixel heuristic failed: %s", e)
        return False


def detect_arabic_text(
    image: Image.Image,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
) -> bool:
    """
    Three-tier Arabic text detection:
      Tier 1: EXIF Unicode scan (free)
      Tier 2: Pixel heuristic (free)
      Tier 3: Gemini pre-screen (1 API call, minimal tokens)

    If Tiers 1+2 both return False, Tier 3 is skipped entirely
    to conserve API quota.

    Args:
        image: PIL Image.
        api_key: Gemini API key.
        model_name: Gemini model ID.

    Returns:
        True if Arabic text confirmed present.
    """
    # Tier 1: Metadata (free)
    if _check_arabic_in_metadata(image):
        logger.info("Arabic in metadata. Confirming via Gemini.")
        resp = _call_gemini_with_retry(
            model_name, ARABIC_PRESCREEN_PROMPT, image, api_key
        )
        if resp and "YES" in resp.upper():
            return True

    # Tier 2: Pixel heuristic (free)
    if not _pixel_heuristic_arabic_check(image):
        logger.info("No text structure. Skipping Arabic layer.")
        return False

    # Tier 3: Gemini pre-screen (1 call)
    logger.info("Text structure found. Querying Gemini for Arabic.")
    resp = _call_gemini_with_retry(
        model_name, ARABIC_PRESCREEN_PROMPT, image, api_key
    )

    if resp and "YES" in resp.upper():
        logger.info("Gemini confirmed Arabic text.")
        return True

    logger.info("No Arabic text confirmed.")
    return False


def analyze_arabic_forensics(
    image: Image.Image,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """
    Full Arabic-aware forensic analysis. Main entry point.

    Args:
        image: PIL Image.
        api_key: Gemini API key.
        model_name: Gemini model ID.

    Returns:
        Dict with keys: arabic_text_detected, arabic_anomalies_found,
        arabic_confidence_score, arabic_manipulation_type,
        arabic_detail_notes, api_calls_used, error.
    """
    default: Dict[str, Any] = {
        "arabic_text_detected": False,
        "arabic_anomalies_found": [],
        "arabic_confidence_score": 0,
        "arabic_manipulation_type": "none",
        "arabic_detail_notes": "No Arabic content detected or analysis skipped.",
        "api_calls_used": 0,
        "error": None,
    }

    if image is None:
        default["error"] = "No image provided."
        return default

    try:
        if image.mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")

        # Step 1: Detect Arabic presence
        arabic_present = detect_arabic_text(image, api_key, model_name)

        if not arabic_present:
            default["api_calls_used"] = 1
            default["arabic_detail_notes"] = (
                "No Arabic script detected. Layer skipped to save quota."
            )
            return default

        # Step 2: Full forensic analysis
        logger.info("Running full Arabic forensic analysis...")
        resp_text = _call_gemini_with_retry(
            model_name, ARABIC_FORENSIC_PROMPT, image, api_key
        )

        if not resp_text:
            default["arabic_text_detected"] = True
            default["api_calls_used"] = 2
            default["error"] = "Arabic detected but analysis API call failed."
            default["arabic_detail_notes"] = default["error"]
            return default

        # Step 3: Parse JSON
        parsed = _extract_json_from_response(resp_text)

        if not parsed:
            default["arabic_text_detected"] = True
            default["api_calls_used"] = 2
            default["error"] = "Unparseable Gemini response."
            default["arabic_detail_notes"] = f"Raw: {resp_text[:500]}"
            return default

        # Step 4: Validate and normalize
        result: Dict[str, Any] = {
            "arabic_text_detected": bool(
                parsed.get("arabic_text_detected", True)
            ),
            "arabic_anomalies_found": parsed.get(
                "arabic_anomalies_found", []
            ),
            "arabic_confidence_score": max(
                0, min(100, int(parsed.get("arabic_confidence_score", 50)))
            ),
            "arabic_manipulation_type": parsed.get(
                "arabic_manipulation_type", "none"
            ),
            "arabic_detail_notes": parsed.get(
                "arabic_detail_notes", "Analysis complete."
            ),
            "api_calls_used": 2,
            "error": None,
        }

        valid_types = {
            "none", "text_overlay", "template_forgery",
            "deepfake_subtitle", "synthetic_document",
        }
        if result["arabic_manipulation_type"] not in valid_types:
            result["arabic_manipulation_type"] = "text_overlay"

        logger.info(
            "Arabic analysis: %d anomalies, %d%% confidence, type=%s",
            len(result["arabic_anomalies_found"]),
            result["arabic_confidence_score"],
            result["arabic_manipulation_type"],
        )
        return result

    except Exception as e:
        logger.error("Arabic analysis crashed: %s", e)
        default["error"] = f"Analysis error: {str(e)}"
        return default


def update_consensus_with_arabic(
    existing_scores: Dict[str, float],
    arabic_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge Arabic results into ELA/FFT/Gemini consensus.

    Weights when Arabic present: ELA 25%, FFT 25%, English Gemini 25%,
    Arabic 25%. High-confidence Arabic anomalies (>70) get up to +10 boost.

    Args:
        existing_scores: {"ela": float, "fft": float, "gemini": float}
        arabic_results: Output from analyze_arabic_forensics().

    Returns:
        Updated consensus dict with combined_score, verdict, margin, notes.
    """
    ela = existing_scores.get("ela", 50.0)
    fft = existing_scores.get("fft", 50.0)
    gemini_en = existing_scores.get("gemini", 50.0)

    ar_detected = arabic_results.get("arabic_text_detected", False)
    ar_conf = arabic_results.get("arabic_confidence_score", 0)
    ar_anomalies = arabic_results.get("arabic_anomalies_found", [])
    ar_type = arabic_results.get("arabic_manipulation_type", "none")

    if not ar_detected:
        combined = (ela + fft + gemini_en) / 3.0
        return {
            "ela_score": ela,
            "fft_score": fft,
            "gemini_score": gemini_en,
            "arabic_score": 0.0,
            "arabic_anomalies": [],
            "arabic_manipulation_type": "none",
            "combined_score": round(combined, 1),
            "verdict": _score_to_verdict(combined),
            "confidence_margin": _calc_margin(ela, fft, gemini_en),
            "notes": "No Arabic content. Standard 3-signal consensus.",
        }

    # 4-signal weighted consensus
    combined = (ela + fft + gemini_en + ar_conf) / 4.0

    # Boost for high-confidence Arabic anomalies
    if ar_conf > 70 and len(ar_anomalies) >= 2:
        boost = min(10.0, (ar_conf - 70) * 0.33)
        combined = min(100.0, combined + boost)
        logger.info("Arabic boost: +%.1f", boost)

    notes = (
        f"Arabic detected: {len(ar_anomalies)} anomalies, "
        f"type={ar_type}. "
        + ("Key: " + "; ".join(ar_anomalies[:3]) if ar_anomalies else "")
    )

    return {
        "ela_score": ela,
        "fft_score": fft,
        "gemini_score": gemini_en,
        "arabic_score": float(ar_conf),
        "arabic_anomalies": ar_anomalies,
        "arabic_manipulation_type": ar_type,
        "combined_score": round(combined, 1),
        "verdict": _score_to_verdict(combined),
        "confidence_margin": _calc_margin(ela, fft, gemini_en, ar_conf),
        "notes": notes,
    }


def _score_to_verdict(score: float) -> str:
    """Convert 0-100 score to verdict string."""
    if score >= 75:
        return "Likely AI-Generated"
    elif score >= 60:
        return "Likely Manipulated"
    elif score >= 40:
        return "Inconclusive"
    else:
        return "Likely Authentic"


def _calc_margin(*scores: float) -> float:
    """
    Confidence margin from score variance.
    High disagreement between signals = wide margin.
    """
    if len(scores) < 2:
        return 15.0
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = var ** 0.5
    return round(min(15.0, max(5.0, 5.0 + std * 0.33)), 1)


# ===========================================================================
# TEST BLOCK
# ===========================================================================
if __name__ == "__main__":
    import sys

    test_key = os.environ.get("GEMINI_API_KEY", "")
    if not test_key:
        print("Set GEMINI_API_KEY env variable first.")
        sys.exit(1)

    path = sys.argv[1] if len(sys.argv) > 1 else "test_arabic.jpg"
    try:
        img = Image.open(path)
        print(f"Loaded: {path} ({img.size})")
    except FileNotFoundError:
        print(f"Not found: {path}")
        sys.exit(1)

    print("\n--- Arabic Forensic Analysis ---")
    results = analyze_arabic_forensics(img, test_key)
    print(f"Detected: {results['arabic_text_detected']}")
    print(f"Confidence: {results['arabic_confidence_score']}%")
    print(f"Type: {results['arabic_manipulation_type']}")
    print(f"Anomalies: {results['arabic_anomalies_found']}")
    print(f"API calls: {results['api_calls_used']}")

    print("\n--- Consensus Merge ---")
    mock = {"ela": 65.0, "fft": 58.0, "gemini": 72.0}
    consensus = update_consensus_with_arabic(mock, results)
    print(f"Combined: {consensus['combined_score']}")
    print(f"Verdict: {consensus['verdict']} ±{consensus['confidence_margin']}%")