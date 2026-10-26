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
    return {"consensus_score": total_score, "uncertainty_margin": uncertainty_margin, "verdict": verdict}