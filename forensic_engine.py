"""
Beetles AI - Forensic Engine v2.5 (Multilingual CJK & Global Script Moat)
Pure analysis functions. No Streamlit UI here.
Integrated: ELA, 2D-FFT, EXIF Provenance, Gemini Multimodal,
and Native Script Forensics for English, Arabic, Chinese (Hanzi), and Korean (Hangul).
"""

from __future__ import annotations

import io
import os
import re
import json
import hashlib
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageChops, ImageEnhance, ExifTags

# Optional CV
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Optional Gemini
try:
    import google.generativeai as genai
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False


# -----------------------------
# Score helpers
# -----------------------------

def _to_01(score: Optional[float]) -> float:
    if score is None:
        return 0.0
    try:
        s = float(score)
    except Exception:
        return 0.0
    if s != s:
        return 0.0
    if s > 1.0:
        s = s / 100.0
    return max(0.0, min(1.0, s))


def _to_100(score: Optional[float]) -> float:
    if score is None:
        return 0.0
    try:
        s = float(score)
    except Exception:
        return 0.0
    if s != s:
        return 0.0
    if s <= 1.0:
        s = s * 100.0
    return max(0.0, min(100.0, s))


# -----------------------------
# ELA (Error Level Analysis)
# -----------------------------

def perform_ela(img_bytes: bytes, quality: int = 90) -> Tuple[Optional[Image.Image], float]:
    """
    Error Level Analysis.
    Returns (ela_preview_image, suspicion_score_0_to_100).
    """
    try:
        original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buf = io.BytesIO()
        original.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")

        ela = ImageChops.difference(original, compressed)
        extrema = ela.getextrema()
        max_diff = max(ex[1] for ex in extrema) if extrema else 1
        scale = 255.0 / max_diff if max_diff > 0 else 1.0
        ela = ImageEnhance.Brightness(ela).enhance(scale)

        # Score: mean intensity of ELA map (higher = more re-compression / edit artifacts)
        arr = np.asarray(ela).astype(np.float32)
        mean_val = float(arr.mean())
        score = max(0.0, min(100.0, (mean_val / 40.0) * 100.0))
        return ela, score
    except Exception:
        return None, 0.0


# -----------------------------
# FFT (2D Fast Fourier Transform)
# -----------------------------

def analyze_fft(img_bytes: bytes) -> Tuple[Optional[Image.Image], float]:
    """
    2D FFT spectral residual heatmap.
    Returns (heatmap_image, suspicion_score_0_to_100).
    """
    try:
        original = Image.open(io.BytesIO(img_bytes)).convert("L")
        arr = np.asarray(original).astype(np.float32)

        # FFT
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1.0)

        # Normalize to 0-255 for display
        mag_norm = (magnitude - magnitude.min()) / (magnitude.ptp() + 1e-8)
        heat_arr = (mag_norm * 255.0).astype(np.uint8)
        heat_img = Image.fromarray(heat_arr).convert("RGB")

        # Heuristic: strong mid-frequency peaks indicate GAN/diffusion grid artifacts
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        mask = (r > min(h, w) * 0.08) & (r < min(h, w) * 0.35)
        ring_mean = float(magnitude[mask].mean()) if mask.any() else 0.0
        global_mean = float(magnitude.mean())
        ratio = ring_mean / (global_mean + 1e-8)
        score = max(0.0, min(100.0, (ratio - 1.0) * 120.0))
        return heat_img, score
    except Exception:
        return None, 0.0


# -----------------------------
# Metadata Signatures
# -----------------------------

def analyze_metadata_signatures(image: Image.Image) -> Dict[str, Any]:
    """
    EXIF / software provenance flags.
    """
    result: Dict[str, Any] = {"software": "Unknown", "flags": []}
    try:
        exif_raw = image.getexif() if hasattr(image, "getexif") else None
        if not exif_raw:
            result["flags"].append("No EXIF — possible export/screenshot/social re-encode")
            result["software"] = "None / Stripped"
            return result

        exif = {}
        for tag_id, value in exif_raw.items():
            name = ExifTags.TAGS.get(tag_id, tag_id)
            exif[name] = value

        software = str(exif.get("Software", "") or "")
        if software:
            result["software"] = software
            sw_l = software.lower()
            ai_markers = ["midjourney", "dall", "stable diffusion", "firefly", "leonardo", "generative"]
            if any(m in sw_l for m in ai_markers):
                result["flags"].append(f"AI software marker in EXIF: {software}")
        else:
            result["software"] = "Not declared"

        if "Orientation" in exif and not software:
            result["flags"].append("Orientation present but Software missing — common in mobile re-saves")

        if not result["flags"]:
            result["flags"].append("EXIF present — no obvious AI software string")
    except Exception as e:
        result["flags"].append(f"EXIF parse error: {e}")
    return result


# -----------------------------
# Gemini Multimodal Core
# -----------------------------

def analyze_multimodal_gemini(
    media_bytes: bytes,
    mime_type: str,
    file_type: str,
    api_key: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Multimodal visual and auditory forensic analysis via Gemini 1.5 Flash.
    Returns (ai_probability_0_to_100, report_dict).
    """
    if not _HAS_GENAI:
        return 50.0, {"summary": "google-generativeai not installed", "findings": []}
    if not api_key:
        return 50.0, {"summary": "Missing API key", "findings": []}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        if file_type == "image":
            prompt = (
                "You are an expert digital forensics examiner. Analyze this image for AI generation, manipulation, or text forgery.\n"
                "Pay strict attention to:\n"
                "1. Lighting/shadow physics and texture continuity.\n"
                "2. Multilingual text integrity (Latin kerning, Chinese Hanzi radicals, Korean Hangul syllabic blocks, Arabic ligatures).\n"
                "3. Compression and pixel interpolation anomalies.\n"
                "Return STRICT JSON only, no markdown, schema:\n"
                "{\n"
                '  "ai_probability": <float 0-100>,\n'
                '  "summary": <string>,\n'
                '  "findings": [{"category": <string>, "evidence": <string>, "severity": <float 0-1>}]\n'
                "}\n"
                "Calibrate: 0=clearly authentic camera capture, 100=clearly AI synthetic. Avoid absolute binary claims."
            )
            img = Image.open(io.BytesIO(media_bytes)).convert("RGB")
            resp = model.generate_content([prompt, img])
        else:
            prompt = (
                "You are an audio deepfake examiner. Analyze this audio for voice cloning, speech synthesis, and phase distortion.\n"
                "Return STRICT JSON only, no markdown, schema:\n"
                "{\n"
                '  "ai_probability": <float 0-100>,\n'
                '  "summary": <string>,\n'
                '  "findings": [{"category": <string>, "evidence": <string>, "severity": <float 0-1>}]\n'
                "}"
            )
            part = {"mime_type": mime_type, "data": media_bytes}
            resp = model.generate_content([prompt, part])

        text = (resp.text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        score = _to_100(data.get("ai_probability", 50.0))
        report = {
            "summary": data.get("summary", ""),
            "findings": data.get("findings", []) if isinstance(data.get("findings"), list) else [],
        }
        return score, report
    except Exception as e:
        return 50.0, {"summary": f"Gemini multimodal error: {e}", "findings": []}


# -----------------------------------------------------------
# Multilingual Script Forensics (Arabic, Chinese, Korean, Latin)
# -----------------------------------------------------------

def analyze_multilingual_forensics(image: Image.Image, api_key: str) -> Dict[str, Any]:
    """
    World-first quad-script forensic layer (Arabic, Chinese Hanzi, Korean Hangul, Latin/English).
    Evaluates radical integrity, syllabic blocks, ligatures, and localized interface tampering.
    """
    empty = {
        "text_detected": False,
        "languages_detected": [],
        "anomalies_found": [],
        "manipulation_type": "none",
        "script_confidence_score": 0,
        "detail_notes": "",
        # Backward-compatibility keys for Arabic UI
        "arabic_text_detected": False,
        "arabic_anomalies_found": [],
        "arabic_manipulation_type": "none",
        "arabic_confidence_score": 0,
        "arabic_detail_notes": "",
    }
    if not _HAS_GENAI or not api_key:
        return empty

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "You are a forensic document examiner specializing in Asian, Middle Eastern, and Western scripts.\n"
            "Inspect this image for textual evidence manipulation across four language families:\n"
            "1. ARABIC (RTL): Broken letter ligatures, reverse text direction, incorrect tashkeel, synthetic WhatsApp overlays.\n"
            "2. CHINESE (Hanzi Simplified/Traditional): Hallucinated pseudo-characters, missing/merged radicals, incorrect stroke counts, WeChat/Weibo font anti-aliasing mismatches.\n"
            "3. KOREAN (Hangul): Syllabic block misalignments (Choseong/Jungseong/Jongseong), irregular baseline spacing, KakaoTalk font forgeries.\n"
            "4. LATIN/ENGLISH: Kerning collisions, font baseline shifts, localized bounding box haloing.\n\n"
            "Return STRICT JSON only, no markdown, schema:\n"
            "{\n"
            '  "text_detected": <bool>,\n'
            '  "languages_detected": ["arabic"|"chinese"|"korean"|"english", ...],\n'
            '  "anomalies_found": [<string>, ...],\n'
            '  "manipulation_type": "none"|"text_overlay"|"template_forgery"|"synthetic_script"|"glyph_hallucination",\n'
            '  "script_confidence_score": <int 0-100>,\n'
            '  "detail_notes": <string>\n'
            "}"
        )

        resp = model.generate_content([prompt, image.convert("RGB")])
        text = (resp.text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)

        langs = data.get("languages_detected", []) or []
        anomalies = data.get("anomalies_found", []) or []
        manip_type = data.get("manipulation_type", "none") or "none"
        conf_score = int(_to_100(data.get("script_confidence_score", 0)))
        notes = str(data.get("detail_notes", "") or "")
        has_arabic = "arabic" in [l.lower() for l in langs]

        return {
            "text_detected": bool(data.get("text_detected", False)),
            "languages_detected": langs,
            "anomalies_found": anomalies,
            "manipulation_type": manip_type,
            "script_confidence_score": conf_score,
            "detail_notes": notes,
            # Backward-compatibility mappings
            "arabic_text_detected": has_arabic,
            "arabic_anomalies_found": anomalies if has_arabic else [],
            "arabic_manipulation_type": manip_type if has_arabic else "none",
            "arabic_confidence_score": conf_score if has_arabic else 0,
            "arabic_detail_notes": notes,
        }
    except Exception as e:
        empty["detail_notes"] = f"Multilingual script layer error: {e}"
        empty["arabic_detail_notes"] = f"Script layer error: {e}"
        return empty


# Backward-compatible alias for existing imports
def analyze_arabic_forensics(image: Image.Image, api_key: str) -> Dict[str, Any]:
    """Backward-compatible wrapper for multilingual forensic audit."""
    return analyze_multilingual_forensics(image, api_key)


def detect_arabic_text(image: Image.Image, api_key: str = "") -> bool:
    """Legacy helper preserved for backward compatibility."""
    return False


def analyze_cjk_forensics(image: Image.Image, api_key: str) -> Dict[str, Any]:
    """Dedicated CJK (Chinese & Korean) forensic pass."""
    res = analyze_multilingual_forensics(image, api_key)
    cjk_langs = [l for l in res.get("languages_detected", []) if l.lower() in ("chinese", "korean")]
    return {
        "cjk_detected": len(cjk_langs) > 0,
        "languages": cjk_langs,
        "anomalies": res.get("anomalies_found", []),
        "confidence": res.get("script_confidence_score", 0),
        "notes": res.get("detail_notes", ""),
    }


# -----------------------------
# Consensus Math
# -----------------------------

def update_consensus_with_multilingual(
    existing_scores: Dict[str, float],
    script_results: Dict[str, Any],
) -> Dict[str, float]:
    """
    Merge Multilingual (Arabic/CJK/Latin) script signal as 4th weight into consensus.
    existing_scores keys: ela, fft, gemini (0-100).
    """
    ela = _to_100(existing_scores.get("ela"))
    fft = _to_100(existing_scores.get("fft"))
    gem = _to_100(existing_scores.get("gemini"))
    script_conf = _to_100(
        script_results.get("script_confidence_score", script_results.get("arabic_confidence_score", 0))
    )

    manip = str(script_results.get("manipulation_type", script_results.get("arabic_manipulation_type", "none")))
    has_text = script_results.get("text_detected", script_results.get("arabic_text_detected", False))

    if manip == "none" or not has_text:
        combined = compute_weighted_consensus(ela, fft, gem)
        return {
            "combined_score": float(combined["consensus_score"]),
            "confidence_margin": float(combined["uncertainty_margin"]),
        }

    # Weights: ELA 25%, FFT 25%, Gemini Multimodal 30%, Script Anomaly 20%
    combined_score = 0.25 * ela + 0.25 * fft + 0.30 * gem + 0.20 * script_conf
    vals = [ela, fft, gem, script_conf]
    spread = max(vals) - min(vals)
    margin = max(8.0, min(25.0, spread * 0.25 + 5.0))

    return {
        "combined_score": float(max(0.0, min(100.0, combined_score))),
        "confidence_margin": float(margin),
    }


# Backward-compatible alias
def update_consensus_with_arabic(
    existing_scores: Dict[str, float],
    arabic_results: Dict[str, Any],
) -> Dict[str, float]:
    return update_consensus_with_multilingual(existing_scores, arabic_results)


def compute_weighted_consensus(
    ela_score: Optional[float],
    fft_score: Optional[float],
    gemini_score: Optional[float],
) -> Dict[str, float]:
    """
    All inputs treated as 0-100.
    Returns dict: consensus_score (0-100), uncertainty_margin (0-100 scale points).
    """
    ela = _to_100(ela_score)
    fft = _to_100(fft_score)

    weights = [0.30, 0.30]
    values = [ela, fft]

    if gemini_score is not None:
        weights.append(0.40)
        values.append(_to_100(gemini_score))
    else:
        weights = [0.5, 0.5]
        values = [ela, fft]

    wsum = sum(weights) or 1.0
    weights = [w / wsum for w in weights]
    consensus = sum(w * v for w, v in zip(weights, values))

    spread = max(values) - min(values) if values else 0.0
    base_margin = 12.0 if gemini_score is None else 8.0
    margin = max(base_margin, min(25.0, spread * 0.22 + base_margin))

    return {
        "consensus_score": float(max(0.0, min(100.0, consensus))),
        "uncertainty_margin": float(margin),
    }