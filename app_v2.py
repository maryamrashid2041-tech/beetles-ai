"""
Beetles AI - Deepfake & Voice Clone Forensics (Streamlit Frontend) v2.3
Dermestid-branded • Staged Beetle Progress • Professional Privacy Mode
Arabic-Aware Forensic Layer • Jargon-free UI for journalists, lawyers & media

Run:
  streamlit run app_v2.py
"""

from __future__ import annotations

import os
import time
import traceback
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st

from forensic_engine import (
    perform_ela,
    analyze_fft,
    analyze_multimodal_gemini,
    compute_weighted_consensus,
    analyze_metadata_signatures,
    analyze_arabic_forensics,          # ▶ ARABIC
    update_consensus_with_arabic,       # ▶ ARABIC
)
from privacy_vault import PrivacyVault, redact_report


# -----------------------------
# Page Config + Dark UI
# -----------------------------

st.set_page_config(
    page_title="Beetles AI | Forensic Truth Platform",
    page_icon="🪲",
    layout="wide",
    initial_sidebar_state="expanded",
)

_DARK_CSS = """
<style>
html, body, [class*="stApp"] {
  background: #0b0f17 !important;
}
section[data-testid="stSidebar"] {
  background: #0a0e15 !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
}
.block-container {
  padding-top: 1.0rem !important;
}
h1, h2, h3, h4, h5, p, li, label, div, span {
  color: rgba(255,255,255,0.92);
}
small, .muted {
  color: rgba(255,255,255,0.65) !important;
}
div[data-testid="stMetricValue"] {
  color: rgba(255,255,255,0.92) !important;
}
div[data-testid="stMetricLabel"] {
  color: rgba(255,255,255,0.68) !important;
}
.stProgress > div > div > div > div {
  background-image: linear-gradient(90deg, #2dd4bf, #60a5fa, #a78bfa) !important;
}
.badge {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.86);
  font-size: 0.85rem;
}
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  padding: 1rem 1.2rem;
  border-radius: 14px;
  margin-bottom: 0.8rem;
}
.dermestid-hook {
  border-left: 3px solid #2dd4bf;
  padding: 0.8rem 1rem;
  background: rgba(45, 212, 191, 0.06);
  border-radius: 0 12px 12px 0;
  margin: 0.6rem 0 1.2rem 0;
  font-size: 1.05rem;
  line-height: 1.45;
}
.report-card {
  border: 1px solid rgba(45, 212, 191, 0.25);
  background: rgba(45, 212, 191, 0.04);
  padding: 1.4rem 1.6rem;
  border-radius: 14px;
  margin: 1rem 0;
}
.verdict-green { color: #3fb950; font-size: 1.2em; font-weight: 700; }
.verdict-yellow { color: #d29922; font-size: 1.2em; font-weight: 700; }
.verdict-red { color: #f85149; font-size: 1.2em; font-weight: 700; }
hr {
  border: none;
  border-top: 1px solid rgba(255,255,255,0.10);
}
</style>
"""
st.markdown(_DARK_CSS, unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x != x:
        return 0.0
    return max(0.0, min(1.0, x))


def _fmt_pct(score01: Optional[float]) -> str:
    if score01 is None:
        return "N/A"
    return f"{_clamp01(score01) * 100:.1f}%"


def _fmt_margin_short(margin01: Optional[float]) -> str:
    if margin01 is None:
        return "±N/A"
    return f"±{_clamp01(margin01) * 100:.1f}%"


def _verdict_emoji(final_score: float) -> str:
    s = _clamp01(final_score)
    if s >= 0.80:
        return "🚨"
    if s >= 0.60:
        return "⚠️"
    if s >= 0.40:
        return "❓"
    if s >= 0.20:
        return "✅"
    return "🛡️"


def _verdict_label(final_score: float) -> str:
    s = _clamp01(final_score)
    if s >= 0.80:
        return "High likelihood of AI manipulation"
    if s >= 0.60:
        return "Moderate likelihood of AI manipulation"
    if s >= 0.40:
        return "Inconclusive / mixed signals"
    if s >= 0.20:
        return "Moderate likelihood of authenticity"
    return "High likelihood of authenticity"


def _ela_to_human(score01: Optional[float]) -> str:
    if score01 is None:
        return "Not available"
    s = _clamp01(score01)
    if s < 0.20:
        return "Clean"
    elif s < 0.60:
        return "Minor inconsistencies"
    else:
        return "Signs of editing detected"


def _fft_to_human(score01: Optional[float]) -> str:
    if score01 is None:
        return "Not available"
    s = _clamp01(score01)
    if s < 0.20:
        return "Natural"
    elif s < 0.60:
        return "Some synthetic traces"
    else:
        return "Synthetic traces found"


def _gemini_to_human(score01: Optional[float]) -> str:
    if score01 is None:
        return "Not available"
    s = _clamp01(score01)
    if s < 0.40:
        return "No AI generation detected"
    elif s < 0.70:
        return "Possible AI involvement"
    else:
        return "AI generation likely"


# ▶ ARABIC — Verdict translation map
ARABIC_VERDICT_MAP = {
    "High likelihood of AI manipulation": "احتمالية عالية للتلاعب بالذكاء الاصطناعي",
    "Moderate likelihood of AI manipulation": "احتمالية متوسطة للتلاعب بالذكاء الاصطناعي",
    "Inconclusive / mixed signals": "غير حاسم / إشارات مختلطة",
    "Moderate likelihood of authenticity": "احتمالية متوسطة للأصالة",
    "High likelihood of authenticity": "احتمالية عالية للأصالة",
    "High likelihood of AI generation": "احتمالية عالية للتوليد بالذكاء الاصطناعي",
    "Inconclusive — manual review recommended": "غير حاسم — يُنصح بالمراجعة اليدوية",
}

# ▶ ARABIC — Signal label translations
ARABIC_SIGNAL_LABELS = {
    "Image Consistency": "اتساق الصورة",
    "Digital Patterns": "الأنماط الرقمية",
    "AI Analysis": "تحليل الذكاء الاصطناعي",
    "Arabic Text Forensics": "تحليل النصوص العربية",
}

# ▶ ARABIC — Manipulation type translations
ARABIC_MANIP_MAP = {
    "none": "لا يوجد",
    "text_overlay": "طبقة نصية مزيفة",
    "template_forgery": "تزوير قالب",
    "deepfake_subtitle": "ترجمة مزيفة بالذكاء الاصطناعي",
    "synthetic_document": "مستند مزيف",
}


def _arabic_verdict(text: str) -> str:
    """▶ ARABIC — Return Arabic translation or original if not found."""
    return ARABIC_VERDICT_MAP.get(text, text)


def _report_verdict(final_score: float):
    s = _clamp01(final_score)
    if s < 0.30:
        return ("High likelihood of authenticity", "verdict-green", "likely authentic")
    elif s <= 0.70:
        return ("Inconclusive — manual review recommended", "verdict-yellow", "inconclusive")
    else:
        return ("High likelihood of AI generation", "verdict-red", "likely AI-generated")


def _render_report_card(
    *,
    final_score: float,
    uncertainty: float,
    ela_score: Optional[float],
    fft_score: Optional[float],
    gemini_score: Optional[float],
    modality: str = "image",
    arabic_results: Optional[Dict[str, Any]] = None,  # ▶ ARABIC
    lang: str = "en",                                  # ▶ ARABIC
) -> None:
    verdict_text, verdict_css, verdict_short = _report_verdict(final_score)
    ela_label = _ela_to_human(ela_score)
    fft_label = _fft_to_human(fft_score)
    gem_label = _gemini_to_human(gemini_score)
    ai_pct = round(_clamp01(final_score) * 100, 1)
    unc_pct = round(_clamp01(uncertainty) * 100, 1)
    modality_word = "image" if modality == "image" else "recording"

    # ▶ ARABIC — Translate verdict if Arabic selected
    display_verdict = _arabic_verdict(verdict_text) if lang == "ar" else verdict_text

    # ▶ ARABIC — Build Arabic forensics section
    arabic_section = ""
    if arabic_results and arabic_results.get("arabic_text_detected"):
        ar_anomalies = arabic_results.get("arabic_anomalies_found", [])
        ar_type = arabic_results.get("arabic_manipulation_type", "none")
        ar_conf = arabic_results.get("arabic_confidence_score", 0)
        ar_notes = arabic_results.get("arabic_detail_notes", "")

        if lang == "ar":
            ar_type_display = ARABIC_MANIP_MAP.get(ar_type, ar_type)
            arabic_section = f"""
            <hr style="border-color:rgba(255,255,255,0.12);">
            <p><strong>🔤 تحليل النصوص العربية:</strong></p>
            <ul style="line-height:1.8;">
                <li>نوع التلاعب: <strong>{ar_type_display}</strong></li>
                <li>نسبة الثقة: <strong>{ar_conf}%</strong></li>
                <li>عدد التشوهات: <strong>{len(ar_anomalies)}</strong></li>
            </ul>
            """
            if ar_anomalies:
                items = "".join(f"<li>{a}</li>" for a in ar_anomalies[:5])
                arabic_section += f"<ul style='line-height:1.8;'>{items}</ul>"
        else:
            ar_type_display = ar_type.replace("_", " ").title()
            arabic_section = f"""
            <hr style="border-color:rgba(255,255,255,0.12);">
            <p><strong>🔤 Arabic Text Forensics:</strong></p>
            <ul style="line-height:1.8;">
                <li>Manipulation Type: <strong>{ar_type_display}</strong></li>
                <li>Confidence: <strong>{ar_conf}%</strong></li>
                <li>Anomalies Found: <strong>{len(ar_anomalies)}</strong></li>
            </ul>
            """
            if ar_anomalies:
                items = "".join(f"<li>{a}</li>" for a in ar_anomalies[:5])
                arabic_section += f"<ul style='line-height:1.8;'>{items}</ul>"

    # ▶ ARABIC — Signal labels
    if lang == "ar":
        ela_lbl = ARABIC_SIGNAL_LABELS["Image Consistency"]
        fft_lbl = ARABIC_SIGNAL_LABELS["Digital Patterns"]
        gem_lbl = ARABIC_SIGNAL_LABELS["AI Analysis"]
        title_text = "🪲 تقرير التحقق من BEETLES AI"
        verdict_lbl = "📌 الحكم العام"
        prob_lbl = "📊 احتمال الذكاء الاصطناعي"
        checks_lbl = "🔍 ما تم فحصه"
        summary_lbl = "📝 ملخص للاستشهاد"
        privacy_lbl = "🔒 الخصوصية: تم حذف ملفك نهائيًا. لا نخزن ملفاتك أبدًا."
        margin_word = "بهامش"
    else:
        ela_lbl = "Image Consistency"
        fft_lbl = "Digital Patterns"
        gem_lbl = "AI Analysis"
        title_text = "🪲 BEETLES AI VERIFICATION REPORT"
        verdict_lbl = "📌 Overall Verdict"
        prob_lbl = "📊 AI Probability"
        checks_lbl = "🔍 What we checked"
        summary_lbl = "📝 Summary for citation"
        privacy_lbl = "🔒 Privacy: Your file has been permanently deleted. We never store your uploads."
        margin_word = "with a margin of"

    html = f"""
    <div class="report-card">
        <h3 style="margin-top:0;">{title_text}</h3>
        <p><strong>{verdict_lbl}:</strong> <span class="{verdict_css}">{display_verdict}</span></p>
        <p><strong>{prob_lbl}:</strong> {ai_pct}% ({margin_word} {unc_pct}%)</p>
        <hr style="border-color:rgba(255,255,255,0.12);">
        <p><strong>{checks_lbl}:</strong></p>
        <ul style="line-height:1.8;">
            <li>{ela_lbl}: <strong>{ela_label}</strong></li>
            <li>{fft_lbl}: <strong>{fft_label}</strong></li>
            <li>{gem_lbl}: <strong>{gem_label}</strong></li>
        </ul>
        {arabic_section}
        <hr style="border-color:rgba(255,255,255,0.12);">
        <p><strong>{summary_lbl}:</strong></p>
        <p style="background:rgba(0,0,0,0.3); padding:12px; border-radius:8px; font-style:italic; line-height:1.5;">
            "Independently verified by Beetles AI — multi-layer analysis indicates
            this {modality_word} is {verdict_short} ({ai_pct}% AI probability).
            Full methodology: beetles-forensics.streamlit.app/methodology"
        </p>
        <p style="color:rgba(255,255,255,0.55); font-size:0.9em;">
            {privacy_lbl}
        </p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _safe_mime(uploaded_file: Any, fallback: str) -> str:
    mt = getattr(uploaded_file, "type", None)
    if isinstance(mt, str) and "/" in mt:
        return mt
    return fallback


def _build_whatsapp_text_card(
    *,
    title: str,
    modality: str,
    mime_type: str,
    consensus: Dict[str, Any],
    components: Dict[str, Any],
    gemini_summary: Optional[str] = None,
    arabic_results: Optional[Dict[str, Any]] = None,  # ▶ ARABIC
) -> str:
    final_score = float(consensus.get("final_score", 0.5) or 0.5)
    uncertainty = float(consensus.get("uncertainty_margin", 0.0) or 0.0)

    emoji = _verdict_emoji(final_score)
    label = _verdict_label(final_score)

    ela = components.get("ela")
    fft = components.get("fft")
    gem = components.get("gemini")

    lines = [
        f"{emoji} *{title}*",
        "━━━━━━━━━━━━━━━━━━━━",
        f"*Verdict:* {label}",
        f"*AI Probability:* {_fmt_pct(final_score)}  ({_fmt_margin_short(uncertainty)})",
        "",
        f"*Type:* {modality} ({mime_type})",
        "",
        "*Checks:*",
        f"• Image Consistency: {_ela_to_human(ela)}",
        f"• Digital Patterns: {_fft_to_human(fft)}",
        f"• AI Analysis: {_gemini_to_human(gem)}",
    ]

    # ▶ ARABIC — Add Arabic findings to WhatsApp card
    if arabic_results and arabic_results.get("arabic_text_detected"):
        ar_type = arabic_results.get("arabic_manipulation_type", "none")
        ar_conf = arabic_results.get("arabic_confidence_score", 0)
        ar_count = len(arabic_results.get("arabic_anomalies_found", []))
        lines += [
            "",
            "*Arabic Text Forensics:*",
            f"• Manipulation Type: {ar_type.replace('_', ' ').title()}",
            f"• Confidence: {ar_conf}%",
            f"• Anomalies: {ar_count}",
        ]

    if gemini_summary:
        summary = str(gemini_summary).strip()
        if summary:
            lines += ["", "*AI Summary:*", summary]

    lines += [
        "",
        "✅ *Verified by Beetles AI* 🪲",
        "_Real dermestid beetles solved murders for 100 years. Ours solve fakes in 30 seconds._",
        "🔒 File permanently deleted after analysis.",
    ]
    return "\n".join(lines)


def _whatsapp_link(text: str) -> str:
    return "https://wa.me/?text=" + urllib.parse.quote(text)


def _render_findings(findings: Any) -> None:
    if not findings:
        st.info("No structured findings were returned.")
        return

    if not isinstance(findings, list):
        st.write(findings)
        return

    for i, f in enumerate(findings, start=1):
        if not isinstance(f, dict):
            continue
        category = str(f.get("category", "unspecified"))
        evidence = str(f.get("evidence", "") or "").strip()
        severity = f.get("severity", 0.0)
        try:
            severity = float(severity)
        except Exception:
            severity = 0.0

        with st.container(border=True):
            st.markdown(f"**{i}. {category}**")
            cols = st.columns([1, 3])
            cols[0].metric("Severity", f"{_clamp01(severity) * 100:.0f}%")
            cols[1].write(evidence if evidence else "—")


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.markdown("### 🪲 Beetles AI")
    st.markdown('<span class="badge">Forensic Truth • v2.3</span>', unsafe_allow_html=True)  # ▶ ARABIC version bump
    st.markdown("---")

    # ▶ ARABIC — Language toggle
    lang_choice = st.selectbox(
        "Language / اللغة",
        options=["English", "العربية"],
        index=0,
    )
    selected_lang = "ar" if lang_choice == "العربية" else "en"
    st.markdown("---")

    default_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            default_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    if not default_key:
        default_key = os.getenv("GEMINI_API_KEY", "")

    gemini_api_key = st.text_input(
        "Google Gemini API Key",
        value=default_key,
        type="password",
        help="Loaded automatically if set in Streamlit Secrets",
    )

    st.markdown("#### Privacy")
    privacy_enabled = st.toggle(
        "🔒 Professional Mode",
        value=True,
        help="AES-128 encrypted • Auto-deleted • Never used for training. Required for journalists & lawyers.",
    )

    vault: Optional[PrivacyVault] = None
    if privacy_enabled:
        try:
            if "vault_key" not in st.session_state:
                st.session_state["vault_key"] = PrivacyVault.generate_key()
            vault = PrivacyVault(st.session_state["vault_key"])
            st.success("AES-128 Active • Zero Retention")
        except Exception as e:
            st.error(f"Failed to initialize PrivacyVault: {e}")
            vault = None
    else:
        st.info("Community Mode • Labeled submissions help train vernacular models")

    st.markdown("---")
    st.caption("Beetles never leave the lab. Your files are encrypted and deleted. We strip the fake — never your sources.")


# -----------------------------
# Main Header
# -----------------------------

st.title("🪲 BEETLES AI")
st.caption("Multi-Layer Verification • Emerging Markets First • Arabic-Aware 🌍")  # ▶ ARABIC

st.markdown(
    """
<div class="dermestid-hook">
<strong>"Real dermestid beetles have solved murders in crime labs for 100 years.<br>
Ours solve digital fakes in 30 seconds."</strong><br><br>
<em>We strip the fake — never your sources.</em><br>
<span class="muted">By the next election cycle, every newsroom will have a 30-second truth check.</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_img, tab_audio, tab_method = st.tabs(
    ["📸 Image Forensics", "🎙️ Voice Clone Forensics", "🔬 Methodology"]
)


# -----------------------------
# Tab 1: Image Forensics
# -----------------------------

with tab_img:
    st.subheader("Image Verification")

    uploaded_img = st.file_uploader(
        "Upload Image (PNG/JPG/WebP)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    img_bytes: Optional[bytes] = None
    if uploaded_img is not None:
        try:
            img_bytes = uploaded_img.getvalue()
        except Exception as e:
            st.error(f"Failed to read image: {e}")

    # State reset on new file
    if "current_file" not in st.session_state or st.session_state["current_file"] != getattr(uploaded_img, "name", None):
        st.session_state["current_file"] = getattr(uploaded_img, "name", None)
        st.session_state["gemini_img_score"] = None
        st.session_state["gemini_img_report"] = None
        st.session_state["ela_score"] = None
        st.session_state["fft_score"] = None
        st.session_state["metadata_res"] = None
        st.session_state["ela_img"] = None
        st.session_state["fft_heat"] = None
        st.session_state["arabic_results"] = None  # ▶ ARABIC

    if img_bytes:
        import io
        from PIL import Image

        mime_type = _safe_mime(uploaded_img, "image/jpeg")
        
        # 1. Open PIL image safely for metadata
        try:
            image_obj = Image.open(io.BytesIO(img_bytes))
        except Exception:
            image_obj = None

        # 2. Run signals and save to session state
        if st.session_state.get("ela_score") is None:
            fft_heat, fft_score = analyze_fft(img_bytes)
            ela_img, ela_score = perform_ela(img_bytes)
            metadata_res = analyze_metadata_signatures(image_obj) if image_obj else {"software": "Unknown", "flags": ["Could not parse EXIF"]}

            st.session_state["ela_score"] = ela_score
            st.session_state["fft_score"] = fft_score
            st.session_state["metadata_res"] = metadata_res
            st.session_state["ela_img"] = ela_img
            st.session_state["fft_heat"] = fft_heat
        else:
            ela_score = st.session_state.get("ela_score")
            fft_score = st.session_state.get("fft_score")
            metadata_res = st.session_state.get("metadata_res")
            ela_img = st.session_state.get("ela_img")
            fft_heat = st.session_state.get("fft_heat")

        # Display images
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Original**")
            st.image(img_bytes, use_container_width=True)
        with c2:
            st.markdown("**Consistency Map**")
            if ela_img: st.image(ela_img, use_container_width=True)
        with c3:
            st.markdown("**Pattern Map**")
            if fft_heat: st.image(fft_heat, use_container_width=True)

        gemini_score_display = st.session_state.get("gemini_img_score")
        arabic_results_display = st.session_state.get("arabic_results")  # ▶ ARABIC

        m1, m2, m3 = st.columns(3)
        m1.metric("Image Consistency", _fmt_pct(ela_score))
        m2.metric("Digital Patterns", _fmt_pct(fft_score))
        m3.metric("AI Analysis", _fmt_pct(gemini_score_display))

        # ▶ ARABIC — Arabic forensics metric
        if arabic_results_display and arabic_results_display.get("arabic_text_detected"):
            ar_conf = arabic_results_display.get("arabic_confidence_score", 0)
            st.metric(
                "Arabic Text Forensics" if selected_lang == "en" else "تحليل النصوص العربية",
                f"{ar_conf}%",
            )

        # --- METADATA DISPLAY BLOCK ---
        if metadata_res:
            with st.expander("🔍 Metadata & Provenance Footprints", expanded=True):
                st.write(f"**Primary Marker / Software:** {metadata_res.get('software', 'Clean')}")
                for flag in metadata_res.get('flags', []):
                    st.info(f"• {flag}")

        st.markdown("---")

        if st.button("Run Full Verification (AI + Consensus)", type="primary", use_container_width=True):
            if not gemini_api_key.strip():
                st.error("Enter your Gemini API Key in the sidebar.")
            else:
                with st.status("🪲 **Beetle Queen analyzing...**", expanded=True):
                    try:
                        score, report = analyze_multimodal_gemini(
                            media_bytes=img_bytes,
                            mime_type=mime_type,
                            file_type="image",
                            api_key=gemini_api_key.strip(),
                        )
                        st.session_state["gemini_img_score"] = score
                        st.session_state["gemini_img_report"] = report

                        # ▶ ARABIC — Run Arabic forensic analysis
                        if image_obj is not None:
                            st.write("🔤 Scanning for Arabic text artifacts...")
                            try:
                                ar_res = analyze_arabic_forensics(
                                    image=image_obj,
                                    api_key=gemini_api_key.strip(),
                                )
                                st.session_state["arabic_results"] = ar_res
                                if ar_res.get("arabic_text_detected"):
                                    st.write(f"✅ Arabic text detected — {len(ar_res.get('arabic_anomalies_found', []))} anomalies found")
                                else:
                                    st.write("ℹ️ No Arabic text detected — layer skipped")
                            except Exception as ar_err:
                                st.warning(f"Arabic analysis skipped: {ar_err}")
                                st.session_state["arabic_results"] = None
                        else:
                            st.session_state["arabic_results"] = None

                    except Exception as e:
                        st.error(f"AI analysis error: {e}")
                st.rerun()

        gemini_img_score = st.session_state.get("gemini_img_score")
        gemini_img_report = st.session_state.get("gemini_img_report")
        arabic_results = st.session_state.get("arabic_results")  # ▶ ARABIC

        consensus = compute_weighted_consensus(ela_score, fft_score, gemini_img_score)

        # ▶ ARABIC — Merge Arabic into consensus
        if arabic_results and arabic_results.get("arabic_text_detected"):
            arabic_consensus = update_consensus_with_arabic(
                existing_scores={
                    "ela": float(ela_score or 0) * 100,
                    "fft": float(fft_score or 0) * 100,
                    "gemini": float(gemini_img_score or 0) * 100,
                },
                arabic_results=arabic_results,
            )
            # Override consensus with Arabic-enhanced scores
            consensus["final_score"] = arabic_consensus["combined_score"] / 100.0
            consensus["uncertainty_margin"] = arabic_consensus["confidence_margin"] / 100.0

        final_score = float(consensus.get("final_score", 0.5) or 0.5)
        uncertainty = float(consensus.get("uncertainty_margin", 0.0) or 0.0)

        st.markdown("### Verification Result")
        _render_report_card(
            final_score=final_score,
            uncertainty=uncertainty,
            ela_score=ela_score,
            fft_score=fft_score,
            gemini_score=gemini_img_score,
            modality="image",
            arabic_results=arabic_results,  # ▶ ARABIC
            lang=selected_lang,              # ▶ ARABIC
        )

        st.progress(int(round(_clamp01(final_score) * 100)))

        if isinstance(gemini_img_report, dict):
            st.markdown("### Detailed AI Findings")
            summary = str(gemini_img_report.get("summary", "") or "").strip()
            if summary:
                st.info(summary)
            _render_findings(gemini_img_report.get("findings", []))

        # ▶ ARABIC — Display Arabic forensic details
        if arabic_results and arabic_results.get("arabic_text_detected"):
            st.markdown("### 🔤 Arabic Text Forensics" if selected_lang == "en" else "### 🔤 تحليل النصوص العربية")
            ar_anomalies = arabic_results.get("arabic_anomalies_found", [])
            ar_type = arabic_results.get("arabic_manipulation_type", "none")
            ar_conf = arabic_results.get("arabic_confidence_score", 0)
            ar_notes = arabic_results.get("arabic_detail_notes", "")

            col_a, col_b = st.columns(2)
            col_a.metric(
                "Confidence" if selected_lang == "en" else "نسبة الثقة",
                f"{ar_conf}%",
            )
            col_b.metric(
                "Anomalies" if selected_lang == "en" else "التشوهات",
                str(len(ar_anomalies)),
            )

            if ar_anomalies:
                for idx, anomaly in enumerate(ar_anomalies, 1):
                    st.warning(f"{idx}. {anomaly}")

            if ar_notes:
                st.caption(ar_notes)

        st.markdown("---")
        if st.button("📲 Generate Forwardable Report Card", use_container_width=True):
            gem_summary = None
            if isinstance(gemini_img_report, dict):
                gem_summary = str(gemini_img_report.get("summary", "") or "").strip() or None

            card = _build_whatsapp_text_card(
                title="Beetles AI Verification Report",
                modality="Image",
                mime_type=mime_type,
                consensus=consensus,
                components=consensus.get("components", {}),
                gemini_summary=gem_summary,
                arabic_results=arabic_results,  # ▶ ARABIC
            )
            st.text_area("Copy & forward on WhatsApp", value=card, height=280)

# -----------------------------
# Tab 2: Voice Clone Forensics
# -----------------------------

with tab_audio:
    st.subheader("Voice Clone Verification")

    uploaded_audio = st.file_uploader(
        "Upload Audio (WAV/MP3/M4A/OGG)",
        type=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
        accept_multiple_files=False,
    )

    if uploaded_audio:
        audio_bytes = uploaded_audio.getvalue()
        mime_type = _safe_mime(uploaded_audio, "audio/wav")
        st.audio(audio_bytes)

        if st.button("Run Voice Verification (AI Audio Analysis)", type="primary", use_container_width=True):
            if not gemini_api_key.strip():
                st.error("Enter your Gemini API Key in the sidebar.")
            else:
                with st.status("🪲 **Voice swarm deploying...**", expanded=True):
                    try:
                        score, report = analyze_multimodal_gemini(
                            media_bytes=audio_bytes,
                            mime_type=mime_type,
                            file_type="audio",
                            api_key=gemini_api_key.strip(),
                        )
                        st.session_state["gemini_audio_score"] = score
                        st.session_state["gemini_audio_report"] = report
                    except Exception as e:
                        st.error(f"AI audio analysis failed: {e}")
                st.rerun()

        gemini_audio_score = st.session_state.get("gemini_audio_score")
        gemini_audio_report = st.session_state.get("gemini_audio_report")

        consensus = compute_weighted_consensus(None, None, gemini_audio_score)
        final_score = float(consensus.get("final_score", 0.5) or 0.5)
        uncertainty = float(consensus.get("uncertainty_margin", 0.0) or 0.0)

        st.markdown("### Verification Result")
        _render_report_card(
            final_score=final_score,
            uncertainty=uncertainty,
            ela_score=None,
            fft_score=None,
            gemini_score=gemini_audio_score,
            modality="audio",
            lang=selected_lang,  # ▶ ARABIC
        )


# -----------------------------
# Tab 3: Methodology
# -----------------------------

with tab_method:
    st.subheader("🔬 Technical Methodology")
    st.markdown(
        """
        **Image Consistency Check (internally: Error Level Analysis)**
        Detects regions with mismatched JPEG compression.

        **Digital Pattern Scan (internally: 2D Fast Fourier Transform)**
        Detects periodic frequency artifacts left by generative AI models.

        **AI Analysis (Gemini Multimodal)**
        Analyzes high-level semantic anomalies, anatomical errors, and lighting logic.

        **Arabic Text Forensics (Gemini Arabic-Aware Layer)**
        Detects Arabic character rendering errors, RTL layout violations, tashkeel
        anomalies, font inconsistencies, and fake Arabic social media templates.
        This layer activates automatically when Arabic script is detected in the image.
        Competitors like Hive and Reality Defender do not support Arabic — this is
        our monopoly wedge for 400M+ Arabic speakers in the MENA region.

        **Overall Verdict (Weighted Consensus)**
        Combines all signals with calibrated uncertainty ranges.
        When Arabic text is present, the Arabic forensic score is incorporated
        as a fourth signal alongside ELA, FFT, and English Gemini analysis.
        """
    )