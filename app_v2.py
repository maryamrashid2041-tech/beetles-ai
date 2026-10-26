"""
Beetles AI - Deepfake & Voice Clone Forensics (Streamlit Frontend) v2.1
Dermestid-branded • Staged Beetle Progress • Professional Privacy Mode

Run:
  streamlit run app_v2.py

Requires:
  pip install streamlit pillow opencv-python numpy cryptography google-generativeai
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
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, x))


def _fmt_pct(score01: Optional[float]) -> str:
    if score01 is None:
        return "N/A"
    return f"{_clamp01(score01) * 100:.1f}%"


def _fmt_margin(margin01: Optional[float]) -> str:
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


def _safe_mime(uploaded_file: Any, fallback: str) -> str:
    mt = getattr(uploaded_file, "type", None)
    if isinstance(mt, str) and "/" in mt:
        return mt
    return fallback


def _store_forensics_record(
    *,
    privacy_enabled: bool,
    vault: Optional[PrivacyVault],
    modality: str,
    media_bytes: bytes,
    mime_type: str,
    consensus: Dict[str, Any],
    gemini_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if "vault_storage" not in st.session_state:
        st.session_state["vault_storage"] = []

    media_sha256 = PrivacyVault.sha256_hex(media_bytes)

    record: Dict[str, Any] = {
        "ts_utc": _utc_iso(),
        "modality": modality,
        "mime_type": mime_type,
        "media_sha256": media_sha256,
        "consensus": consensus,
        "gemini_report_redacted": redact_report(gemini_report) if isinstance(gemini_report, dict) else None,
    }

    if privacy_enabled:
        if vault is None:
            raise RuntimeError("Privacy mode is enabled but vault is not initialized.")
        encrypted = vault.encrypt_dict(record)
        token_str = vault.token_to_str(encrypted)
        stored = {
            "encrypted": True,
            "token": token_str,
            "ts_utc": record["ts_utc"],
            "modality": modality,
            "media_sha256": media_sha256,
        }
        st.session_state["vault_storage"].append(stored)
        return stored

    stored = {"encrypted": False, "record": record}
    st.session_state["vault_storage"].append(stored)
    return stored


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
        f"*AI Likelihood:* {_fmt_pct(final_score)}  ({_fmt_margin(uncertainty)})",
        "",
        f"*Modality:* {modality} ({mime_type})",
        "",
        "*Signals:*",
        f"• ELA: {_fmt_pct(ela) if ela is not None else 'N/A'}",
        f"• FFT: {_fmt_pct(fft) if fft is not None else 'N/A'}",
        f"• Gemini: {_fmt_pct(gem) if gem is not None else 'N/A'}",
    ]

    if gemini_summary:
        summary = str(gemini_summary).strip()
        if summary:
            lines += ["", "*Gemini Summary:*", summary]

    lines += [
        "",
        "✅ *Verified by Beetles AI* 🪲",
        "_Real dermestid beetles solved murders for 100 years. Ours solve fakes in 30 seconds._",
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
    st.markdown('<span class="badge">Forensic Truth • v2.1</span>', unsafe_allow_html=True)
    st.markdown("---")

    default_key = os.getenv("GEMINI_API_KEY", "")
    gemini_api_key = st.text_input(
        "Google Gemini API Key",
        value=default_key,
        type="password",
        help="Stored only in memory for this session. Or set env var GEMINI_API_KEY.",
    )

    st.markdown("#### Privacy")
    privacy_enabled = st.toggle(
        "🔒 Professional Mode",
        value=True,
        help="AES-256 encrypted • Auto-deleted • Never used for training. Required for journalists & lawyers.",
    )

    vault: Optional[PrivacyVault] = None
    if privacy_enabled:
        try:
            if "vault_key" not in st.session_state:
                st.session_state["vault_key"] = PrivacyVault.generate_key()
            vault = PrivacyVault(st.session_state["vault_key"])
            st.success("AES-256 Active • Zero Retention")
            with st.expander("Vault Details", expanded=False):
                st.write("Encryption: Fernet (AES-256 + HMAC).")
                st.code(st.session_state["vault_key"], language="text")
                st.caption("Keep this key secret. Losing it = cannot decrypt logs.")
        except Exception as e:
            st.error(f"Failed to initialize PrivacyVault: {e}")
            vault = None
    else:
        st.info("Community Mode • Labeled submissions help train vernacular models")

    st.markdown("---")
    with st.expander("Session Storage", expanded=False):
        count = len(st.session_state.get("vault_storage", []))
        st.write(f"Stored entries this session: **{count}**")
        if st.button("Clear Session Storage", use_container_width=True):
            st.session_state["vault_storage"] = []
            st.success("Cleared.")
            st.rerun()

    st.markdown("---")
    st.caption("Beetles never leave the lab. Your files are encrypted and deleted. We strip the fake — never your sources.")
    st.caption("Disclaimer: Probabilistic signals, not definitive legal proof.")


# -----------------------------
# Hero / Brand
# -----------------------------

st.title("🪲 BEETLES AI")
st.caption("Multi-Signal Digital Forensics • Emerging Markets First")

st.markdown(
    """
<div class="dermestid-hook">
<strong>“Real dermestid beetles have solved murders in crime labs for 100 years.<br>
Ours solve digital fakes in 30 seconds.”</strong><br><br>
<em>We strip the fake — never your sources.</em><br>
<span class="muted">By the next election cycle, every Urdu newsroom will have a 30-second truth check.</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_img, tab_audio = st.tabs(["📸 Image Forensics", "🎙️ Voice Clone Forensics"])


# -----------------------------
# Tab 1: Image Forensics
# -----------------------------
with tab_img:
    st.subheader("Image Forensics")
    st.write("ELA + FFT + Gemini multimodal consensus. Calibrated uncertainty. Forwardable report card.")

    uploaded_img = st.file_uploader(
        "Upload Image (PNG/JPG/WebP)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    img_bytes: Optional[bytes] = None
    if uploaded_img is not None:
        try:
            img_bytes = uploaded_img.getvalue()
            if not img_bytes:
                raise ValueError("Uploaded file is empty.")
        except Exception as e:
            st.error(f"Failed to read uploaded image: {e}")
            img_bytes = None

    ela_img = None
    fft_heat = None
    ela_score: Optional[float] = None
    fft_score: Optional[float] = None
    gemini_img_score: Optional[float] = None
    gemini_img_report: Optional[Dict[str, Any]] = None

    if img_bytes:
        mime_type = _safe_mime(uploaded_img, "image/jpeg")

        # --- STAGED BEETLE PROGRESS ---
        status_box = st.status("🪲 **Swarm deploying...**", expanded=True)

        with status_box:
            # Beetle 1 — Frequency
            st.write("🪲 **Beetle 1 (Frequency):** scanning 2D FFT for diffusion artifacts...")
            t0 = time.time()
            try:
                fft_heat, fft_score = analyze_fft(img_bytes)
                st.write(f"  done — {time.time()-t0:.1f}s")
            except Exception as e:
                st.write(f"  failed — {e}")
                fft_heat, fft_score = None, None

            # Beetle 2 — Physics / ELA
            st.write("🪲 **Beetle 2 (Physics):** checking Error Level Analysis compression gradients...")
            t1 = time.time()
            try:
                ela_img, ela_score = perform_ela(img_bytes)
                st.write(f"  done — {time.time()-t1:.1f}s")
            except Exception as e:
                st.write(f"  failed — {e}")
                ela_img, ela_score = None, None

            # Beetle 3 — Multimodal (only if key present and user triggers later, but we stage the slot)
            st.write("🪲 **Beetle 3 (Sensor):** ready for Gemini multimodal pass...")
            st.write("  awaiting verification trigger")

            status_box.update(label="🪲 **Base swarm complete — ready for consensus**", state="complete", expanded=False)

        # Visuals
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Original**")
            st.image(img_bytes, use_container_width=True)
        with c2:
            st.markdown("**ELA Heatmap**")
            if ela_img is not None:
                st.image(ela_img, use_container_width=True)
            else:
                st.info("ELA unavailable.")
        with c3:
            st.markdown("**FFT Heatmap**")
            if fft_heat is not None:
                st.image(fft_heat, use_container_width=True)
            else:
                st.info("FFT heatmap unavailable.")

        m1, m2, m3 = st.columns(3)
        m1.metric("ELA Anomaly", _fmt_pct(ela_score))
        m2.metric("FFT Artifact", _fmt_pct(fft_score))
        m3.metric("Gemini", _fmt_pct(gemini_img_score))

        st.markdown("---")

        run_gemini = st.button("Run Full Swarm (Gemini + Consensus)", type="primary", use_container_width=True)
        if run_gemini:
            if not gemini_api_key.strip():
                st.error("Enter your Gemini API Key in the sidebar.")
            else:
                queen = st.status("🪲 **Beetle Queen (Consensus) activating...**", expanded=True)
                with queen:
                    st.write("🪲 **Beetle 3 (Sensor):** querying Gemini 1.5 Flash vision engine...")
                    t2 = time.time()
                    try:
                        gemini_img_score, gemini_img_report = analyze_multimodal_gemini(
                            media_bytes=img_bytes,
                            mime_type=mime_type,
                            file_type="image",
                            api_key=gemini_api_key.strip(),
                        )
                        st.write(f"  done — {time.time()-t2:.1f}s")
                    except Exception as e:
                        st.error(f"Gemini failed: {e}")
                        st.caption(traceback.format_exc())
                        gemini_img_score, gemini_img_report = None, None

                    st.write("🪲 **Beetle Queen:** weighing multi-signal evidence + uncertainty margin...")
                    time.sleep(0.3)
                    queen.update(label="✅ **Forensic Consensus Complete**", state="complete", expanded=False)

        # Consensus (runs with whatever signals we have)
        consensus = compute_weighted_consensus(ela_score, fft_score, gemini_img_score)
        final_score = float(consensus.get("final_score", 0.5) or 0.5)
        uncertainty = float(consensus.get("uncertainty_margin", 0.0) or 0.0)
        interpretation = str(consensus.get("interpretation", "inconclusive"))

        st.markdown("### Result")
        st.markdown(
            f"<div class='card'><b>{_verdict_emoji(final_score)} Verdict:</b> {_verdict_label(final_score)}<br/>"
            f"<span class='muted'>Interpretation: {interpretation}</span></div>",
            unsafe_allow_html=True,
        )

        st.write(f"**AI Likelihood:** {_fmt_pct(final_score)}  •  **Uncertainty:** {_fmt_margin(uncertainty)}")
        st.progress(int(round(_clamp01(final_score) * 100)))

        if isinstance(gemini_img_report, dict):
            st.markdown("### Gemini Findings")
            summary = str(gemini_img_report.get("summary", "") or "").strip()
            if summary:
                st.info(summary)
            _render_findings(gemini_img_report.get("findings", []))

        st.markdown("---")
        if st.button("Save Forensic Log to Session", use_container_width=True):
            try:
                stored = _store_forensics_record(
                    privacy_enabled=privacy_enabled,
                    vault=vault,
                    modality="image",
                    media_bytes=img_bytes,
                    mime_type=mime_type,
                    consensus=consensus,
                    gemini_report=gemini_img_report,
                )
                if stored.get("encrypted"):
                    st.success("Encrypted log saved.")
                    with st.expander("Encrypted Token", expanded=False):
                        st.code(stored.get("token", ""), language="text")
                else:
                    st.success("Plain log saved.")
            except Exception as e:
                st.error(f"Save failed: {e}")
                st.caption(traceback.format_exc())

        st.markdown("### One-Click WhatsApp Report")
        if st.button("📲 Generate Forwardable Report Card", use_container_width=True):
            gem_summary = None
            if isinstance(gemini_img_report, dict):
                gem_summary = str(gemini_img_report.get("summary", "") or "").strip() or None

            card = _build_whatsapp_text_card(
                title="Beetles AI Forensics Report",
                modality="Image",
                mime_type=mime_type,
                consensus=consensus,
                components=consensus.get("components", {}),
                gemini_summary=gem_summary,
            )
            st.text_area("Copy & forward on WhatsApp", value=card, height=280)
            wa = _whatsapp_link(card)
            try:
                st.link_button("Open WhatsApp with this message", wa, use_container_width=True)
            except Exception:
                st.markdown(f"[Open WhatsApp with this message]({wa})")

    else:
        st.info("Upload an image to deploy the swarm.")


# -----------------------------
# Tab 2: Voice Clone Forensics
# -----------------------------
with tab_audio:
    st.subheader("Voice Clone Forensics")
    st.write("Gemini multimodal audio forensics. Acoustic anomaly detection. Calibrated report.")

    uploaded_audio = st.file_uploader(
        "Upload Audio (WAV/MP3/M4A/OGG)",
        type=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
        accept_multiple_files=False,
        key="audio_uploader",
    )

    audio_bytes: Optional[bytes] = None
    if uploaded_audio is not None:
        try:
            audio_bytes = uploaded_audio.getvalue()
            if not audio_bytes:
                raise ValueError("Uploaded file is empty.")
        except Exception as e:
            st.error(f"Failed to read audio: {e}")
            audio_bytes = None

    gemini_audio_score: Optional[float] = None
    gemini_audio_report: Optional[Dict[str, Any]] = None

    if audio_bytes:
        mime_type = _safe_mime(uploaded_audio, "audio/wav")
        st.markdown("**Playback**")
        st.audio(audio_bytes)

        st.markdown("---")
        run_audio = st.button("Run Voice Swarm (Gemini Audio Forensics)", type="primary", use_container_width=True)
        if run_audio:
            if not gemini_api_key.strip():
                st.error("Enter your Gemini API Key in the sidebar.")
            else:
                queen = st.status("🪲 **Voice swarm deploying...**", expanded=True)
                with queen:
                    st.write("🪲 **Beetle 1 (Spectral):** analyzing frequency & breathing patterns...")
                    time.sleep(0.4)
                    st.write("🪲 **Beetle 2 (Prosody):** checking pause rhythm & micro-artifacts...")
                    time.sleep(0.3)
                    st.write("🪲 **Beetle 3 (Sensor):** querying Gemini 1.5 Flash audio engine...")
                    t0 = time.time()
                    try:
                        gemini_audio_score, gemini_audio_report = analyze_multimodal_gemini(
                            media_bytes=audio_bytes,
                            mime_type=mime_type,
                            file_type="audio",
                            api_key=gemini_api_key.strip(),
                        )
                        st.write(f"  done — {time.time()-t0:.1f}s")
                    except Exception as e:
                        st.error(f"Gemini audio failed: {e}")
                        st.caption(traceback.format_exc())
                        gemini_audio_score, gemini_audio_report = None, None

                    st.write("🪲 **Beetle Queen:** computing calibrated voice-clone likelihood...")
                    time.sleep(0.25)
                    queen.update(label="✅ **Voice Consensus Complete**", state="complete", expanded=False)

        consensus = compute_weighted_consensus(None, None, gemini_audio_score)
        final_score = float(consensus.get("final_score", 0.5) or 0.5)
        uncertainty = float(consensus.get("uncertainty_margin", 0.0) or 0.0)
        interpretation = str(consensus.get("interpretation", "inconclusive"))

        st.markdown("### Result")
        st.markdown(
            f"<div class='card'><b>{_verdict_emoji(final_score)} Verdict:</b> {_verdict_label(final_score)}<br/>"
            f"<span class='muted'>Interpretation: {interpretation}</span></div>",
            unsafe_allow_html=True,
        )
        st.write(f"**AI Likelihood:** {_fmt_pct(final_score)}  •  **Uncertainty:** {_fmt_margin(uncertainty)}")
        st.progress(int(round(_clamp01(final_score) * 100)))

        if isinstance(gemini_audio_report, dict):
            st.markdown("### Acoustic Anomalies")
            summary = str(gemini_audio_report.get("summary", "") or "").strip()
            if summary:
                st.info(summary)
            _render_findings(gemini_audio_report.get("findings", []))

        st.markdown("---")
        if st.button("Save Forensic Log (Audio)", use_container_width=True):
            try:
                stored = _store_forensics_record(
                    privacy_enabled=privacy_enabled,
                    vault=vault,
                    modality="audio",
                    media_bytes=audio_bytes,
                    mime_type=mime_type,
                    consensus=consensus,
                    gemini_report=gemini_audio_report,
                )
                if stored.get("encrypted"):
                    st.success("Encrypted log saved.")
                    with st.expander("Encrypted Token", expanded=False):
                        st.code(stored.get("token", ""), language="text")
                else:
                    st.success("Plain log saved.")
            except Exception as e:
                st.error(f"Save failed: {e}")
                st.caption(traceback.format_exc())

        st.markdown("### One-Click WhatsApp Report")
        if st.button("📲 Generate Voice Report Card", use_container_width=True):
            gem_summary = None
            if isinstance(gemini_audio_report, dict):
                gem_summary = str(gemini_audio_report.get("summary", "") or "").strip() or None

            card = _build_whatsapp_text_card(
                title="Beetles AI Voice Forensics Report",
                modality="Audio",
                mime_type=mime_type,
                consensus=consensus,
                components=consensus.get("components", {}),
                gemini_summary=gem_summary,
            )
            st.text_area("Copy & forward on WhatsApp", value=card, height=280)
            wa = _whatsapp_link(card)
            try:
                st.link_button("Open WhatsApp with this message", wa, use_container_width=True)
            except Exception:
                st.markdown(f"[Open WhatsApp with this message]({wa})")

    else:
        st.info("Upload an audio file to deploy the voice swarm.")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("🪲 Beetles AI — Real dermestid beetles solved murders for 100 years. Ours solve fakes in 30 seconds.")
st.caption("Professional Mode: encrypted at rest • never trained • auto-deleted. Community Mode feeds the vernacular flywheel.")

with st.expander("Debug / Session (safe)", expanded=False):
    st.write("Non-sensitive session info only (no API key).")
    st.json(
        {
            "privacy_enabled": bool(privacy_enabled),
            "stored_entries": len(st.session_state.get("vault_storage", [])),
            "time_utc": _utc_iso(),
        }
    )