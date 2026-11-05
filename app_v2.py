"""
Beetles AI - Deepfake & Voice Clone Forensics (Streamlit Frontend) v2.2
Dermestid-branded • Staged Beetle Progress • Professional Privacy Mode
Jargon-free UI for journalists, lawyers & media companies

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
) -> None:
    verdict_text, verdict_css, verdict_short = _report_verdict(final_score)
    ela_label = _ela_to_human(ela_score)
    fft_label = _fft_to_human(fft_score)
    gem_label = _gemini_to_human(gemini_score)
    ai_pct = round(_clamp01(final_score) * 100, 1)
    unc_pct = round(_clamp01(uncertainty) * 100, 1)
    modality_word = "image" if modality == "image" else "recording"

    html = f"""
    <div class="report-card">
        <h3 style="margin-top:0;">🪲 BEETLES AI VERIFICATION REPORT</h3>
        <p><strong>📌 Overall Verdict:</strong> <span class="{verdict_css}">{verdict_text}</span></p>
        <p><strong>📊 AI Probability:</strong> {ai_pct}% (with a margin of {unc_pct}%)</p>
        <hr style="border-color:rgba(255,255,255,0.12);">
        <p><strong>🔍 What we checked:</strong></p>
        <ul style="line-height:1.8;">
            <li>Image Consistency: <strong>{ela_label}</strong></li>
            <li>Digital Patterns: <strong>{fft_label}</strong></li>
            <li>AI Analysis: <strong>{gem_label}</strong></li>
        </ul>
        <hr style="border-color:rgba(255,255,255,0.12);">
        <p><strong>📝 Summary for citation:</strong></p>
        <p style="background:rgba(0,0,0,0.3); padding:12px; border-radius:8px; font-style:italic; line-height:1.5;">
            "Independently verified by Beetles AI — multi-layer analysis indicates
            this {modality_word} is {verdict_short} ({ai_pct}% AI probability).
            Full methodology: beetles-forensics.streamlit.app/methodology"
        </p>
        <p style="color:rgba(255,255,255,0.55); font-size:0.9em;">
            🔒 Privacy: Your file has been permanently deleted. We never store your uploads.
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
    st.markdown('<span class="badge">Forensic Truth • v2.2</span>', unsafe_allow_html=True)
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
st.caption("Multi-Layer Verification • Emerging Markets First")

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

    if img_bytes:
        mime_type = _safe_mime(uploaded_img, "image/jpeg")

        # Run base signal extraction
        fft_heat, fft_score = analyze_fft(img_bytes)
        ela_img, ela_score = perform_ela(img_bytes)

        st.session_state["ela_score"] = ela_score
        st.session_state["fft_score"] = fft_score

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

        m1, m2, m3 = st.columns(3)
        m1.metric("Image Consistency", _fmt_pct(ela_score))
        m2.metric("Digital Patterns", _fmt_pct(fft_score))
        m3.metric("AI Analysis", _fmt_pct(gemini_score_display))

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
                    except Exception as e:
                        st.error(f"AI analysis error: {e}")
                st.rerun()

        gemini_img_score = st.session_state.get("gemini_img_score")
        gemini_img_report = st.session_state.get("gemini_img_report")

        consensus = compute_weighted_consensus(ela_score, fft_score, gemini_img_score)
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
        )

        st.progress(int(round(_clamp01(final_score) * 100)))

        if isinstance(gemini_img_report, dict):
            st.markdown("### Detailed AI Findings")
            summary = str(gemini_img_report.get("summary", "") or "").strip()
            if summary:
                st.info(summary)
            _render_findings(gemini_img_report.get("findings", []))

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

        **Overall Verdict (Weighted Consensus)**
        Combines all signals with calibrated uncertainty ranges.
        """
    )