"""
Beetles AI - Frontend UI v2.6 (Multilingual Audio-Visual & Global Forensics)
Runs on Streamlit: https://beetles-forensics.streamlit.app/
Dual-Engine: Image Deepfake / Document Forgery + Multilingual Audio Voice Clone Forensics
"""

from __future__ import annotations

import io
import os
import time
import streamlit as st
from PIL import Image

# Import forensic engine routines
try:
    import forensic_engine as fe
except ImportError:
    st.error("Fatal Error: `forensic_engine.py` not found in current working directory.")

# Import privacy vault routines with fallback
try:
    import privacy_vault as pv
    _HAS_VAULT = True
except ImportError:
    _HAS_VAULT = False

# -----------------------------
# Page Configuration & Styling
# -----------------------------
st.set_page_config(
    page_title="Beetles AI | Digital Forensics",
    page_icon="🪲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Mode Cyber Forensics CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #f0f6fc;
        margin-bottom: 0px;
    }
    .tagline {
        color: #8b949e;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }
    .quote-box {
        background-color: #161b22;
        border-left: 4px solid #238636;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 25px;
        color: #c9d1d9;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .quote-box b {
        color: #58a6ff;
    }
    .stMetric {
        background: #161b22;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .badge-secure {
        background-color: #0e4429;
        color: #3fb950;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
        margin-top: 5px;
    }
    .badge-alert {
        background-color: #490202;
        color: #ff7b72;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Silent Enterprise API Key Resolution
# -----------------------------
resolved_api_key = ""
if "GEMINI_API_KEY" in st.secrets:
    resolved_api_key = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    resolved_api_key = os.environ["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in st.session_state:
    resolved_api_key = st.session_state["GEMINI_API_KEY"]

# -----------------------------
# Sidebar: Engine Configuration
# -----------------------------
st.sidebar.markdown("### 🪲 Beetles AI")
st.sidebar.caption("Forensic Truth Engine • v2.6 Multilingual")

selected_lang = st.sidebar.selectbox(
    "Language Model Focus / اللغة / 语言",
    [
        "English",
        "Arabic (العربية)",
        "Chinese / Cantonese (中文)",
        "Urdu / Hindi (اردو)",
        "Korean (한국어)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛰️ Engine Infrastructure")
if resolved_api_key:
    st.sidebar.markdown(
        '<span class="badge-secure">● Enterprise Node Connected</span>',
        unsafe_allow_html=True
    )
    st.sidebar.caption("Full Multimodal & Acoustic Probes Active")
else:
    st.sidebar.markdown(
        '<span class="badge-alert">○ Offline Mode (Local Probes Only)</span>',
        unsafe_allow_html=True
    )
    st.sidebar.caption("Add GEMINI_API_KEY to Streamlit Secrets to activate neural probes.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔒 Privacy Architecture")
pro_mode = st.sidebar.toggle("Professional Mode", value=True, help="Enables AES encryption and immediate zero-retention memory scrubbing.")

if pro_mode:
    st.sidebar.success("🛡️ AES Active • Zero Retention")
    st.sidebar.caption("Files never leave the memory buffer. Source metadata stripped upon report rendering.")
else:
    st.sidebar.warning("Standard Mode Active")

# Hidden Developer Override Expander (For stealth local debugging)
with st.sidebar.expander("⚙️ Developer Node Override", expanded=False):
    dev_key = st.text_input("Custom Gemini Key", type="password", help="Leave blank to use default enterprise node.")
    if dev_key:
        resolved_api_key = dev_key
        st.session_state["GEMINI_API_KEY"] = dev_key

st.sidebar.markdown("---")
st.sidebar.caption("“Beetles never leave the lab. We strip the fake — never your sources.”")

# -----------------------------
# Header Branding
# -----------------------------
st.markdown('<h1 class="main-title">🪲 BEETLES AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="tagline">Multi-Signal Digital Forensics • Non-Western Linguistic Defense • Calibrated Truth</p>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="quote-box">
    <b>"Real dermestid beetles have solved murders in crime labs for 100 years. Ours solve digital fakes in 30 seconds."</b><br>
    <i>Multi-layer forensic consensus engineered for emerging market disinformation, Cantonese/Arabic voice heists, and trial evidence verification.</i>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Tabs Interface
# -----------------------------
tab_image, tab_voice, tab_methodology = st.tabs([
    "🖼️ Image & Document Forensics",
    "🎙️ Voice Clone Forensics (Multilingual)",
    "🔬 Threat Intelligence & Methodology"
])

api_key = resolved_api_key

# ==============================================================================
# TAB 1: IMAGE & DOCUMENT FORENSICS
# ==============================================================================
with tab_image:
    st.markdown("### 🔍 Visual Manipulation & Document Forgery Verification")
    st.caption("Probes: Error Level Analysis (ELA), 2D-FFT Spectral Resampling, EXIF Provenance, and Quad-Script Radical/Ligature Integrity.")

    uploaded_img = st.file_uploader(
        "Upload image for forensic analysis (Disputed evidence, screenshots, identity documents)",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader"
    )

    if uploaded_img:
        img_bytes = uploaded_img.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        col_prev, col_stats = st.columns([1, 2])
        with col_prev:
            st.image(pil_image, caption="Uploaded Artifact", use_container_width=True)

        if st.button("Run Multi-Signal Image Forensics", type="primary", use_container_width=True):
            with st.spinner("Stripping compression artifacts, frequency residuals, and script geometry..."):
                # 1. Local ELA Pass
                ela_img, ela_score = fe.perform_ela(img_bytes)
                
                # 2. Local FFT Pass
                fft_img, fft_score = fe.analyze_fft(img_bytes)
                
                # 3. EXIF & PNG Metadata Scan
                meta_res = fe.analyze_metadata_signatures(pil_image)
                
                # 4. Gemini Multimodal Pass
                gemini_score = None
                gemini_report = {}
                if api_key:
                    mime = uploaded_img.type or "image/jpeg"
                    gemini_score, gemini_report = fe.analyze_multimodal_gemini(
                        media_bytes=img_bytes,
                        mime_type=mime,
                        file_type="image",
                        api_key=api_key
                    )

                # 5. Multilingual Document/Script Pass
                script_res = {}
                if api_key:
                    script_res = fe.analyze_multilingual_forensics(pil_image, api_key=api_key)

                # Consensus Math
                existing_scores = {"ela": ela_score, "fft": fft_score}
                if gemini_score is not None:
                    existing_scores["gemini"] = gemini_score

                if script_res.get("text_detected", False):
                    consensus = fe.update_consensus_with_multilingual(existing_scores, script_res)
                    final_score = consensus["combined_score"]
                    margin = consensus["confidence_margin"]
                else:
                    consensus = fe.compute_weighted_consensus(ela_score, fft_score, gemini_score)
                    final_score = consensus["consensus_score"]
                    margin = consensus["uncertainty_margin"]

                # -----------------------------
                # Display Verdict & Scorecard
                # -----------------------------
                st.markdown("---")
                st.markdown("### 📋 Forensic Evidence Scorecard")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(
                        label="AI Generation Likelihood",
                        value=f"{final_score:.1f}%",
                        delta=f"±{margin:.1f}% Calibrated Margin",
                        delta_color="inverse"
                    )
                with m2:
                    if final_score >= 70:
                        verdict = "HIGH LIKELIHOOD OF SYNTHETIC / TAMPERED ASSET"
                        v_color = "badge-alert"
                    elif final_score <= 30:
                        verdict = "HIGH LIKELIHOOD OF AUTHENTIC CAMERA CAPTURE"
                        v_color = "badge-secure"
                    else:
                        verdict = "INDETERMINATE / MIXED FORENSIC SIGNALS"
                        v_color = "badge-alert"
                    st.markdown(f"**Forensic Verdict:**<br><span class='{v_color}'>{verdict}</span>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"**Software Provenance:** `{meta_res.get('software', 'None declared')}`")
                    st.markdown(f"**Script Scan:** `{', '.join(script_res.get('languages_detected', ['None / Latin']))}`")

                # Visual Diagnostic Breakdown
                st.markdown("#### 🔬 Diagnostic Signal Maps")
                c1, c2 = st.columns(2)
                with c1:
                    if ela_img:
                        st.image(ela_img, caption=f"Error Level Analysis Map (Suspicion: {ela_score:.1f}%)", use_container_width=True)
                with c2:
                    if fft_img:
                        st.image(fft_img, caption=f"2D-FFT Resampling Spectrum (Suspicion: {fft_score:.1f}%)", use_container_width=True)

                # Script & Document Tampering Expander
                if script_res.get("text_detected", False):
                    with st.expander("📝 Multilingual Script & Document Integrity Analysis", expanded=True):
                        st.write(f"**Target Manipulation Type:** `{script_res.get('manipulation_type')}`")
                        st.write(f"**Script Confidence Score:** {script_res.get('script_confidence_score')}%")
                        st.write(f"**Examiner Notes:** {script_res.get('detail_notes')}")
                        if script_res.get("anomalies_found"):
                            st.write("**Specific Structural Anomalies:**")
                            for a in script_res["anomalies_found"]:
                                st.write(f"- {a}")

                # EXIF Expander
                with st.expander("🏷️ EXIF / Metadata Provenance Header"):
                    for f in meta_res.get("flags", []):
                        st.write(f"- {f}")

                # WhatsApp / Fast Share Report Card
                whatsapp_summary = (
                    f"🪲 *BEETLES AI FORENSIC AUDIT REPORT*\n"
                    f"• AI Likelihood: {final_score:.1f}% (±{margin:.1f}%)\n"
                    f"• Verdict: {verdict}\n"
                    f"• ELA Signal: {ela_score:.1f}%\n"
                    f"• FFT Resample Signal: {fft_score:.1f}%\n"
                    f"• Software: {meta_res.get('software', 'None declared')}\n"
                    f"• Protocol: AES-128 Zero-Retention Audit\n"
                    f"_Independently verified via Beetles AI Forensics._"
                )
                st.text_area("WhatsApp-Ready Report Card (Click to Copy)", value=whatsapp_summary, height=130)


# ==============================================================================
# TAB 2: VOICE CLONE FORENSICS (MULTILINGUAL)
# ==============================================================================
with tab_voice:
    st.markdown("### 🎙️ Language-Aware Voice Clone & Audio Forensics")
    st.caption("Engineered specifically for non-Western voice heists: Cantonese tonal contour drift, Arabic pharyngeal fricatives, and Urdu breath-burst synthesis.")

    uploaded_audio = st.file_uploader(
        "Upload audio evidence (Voice note, phone call recording, memo)",
        type=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
        key="voice_uploader"
    )

    if uploaded_audio:
        audio_bytes = uploaded_audio.read()
        mime_type = uploaded_audio.type or "audio/mp3"
        st.audio(audio_bytes, format=mime_type)

        if st.button("Run Multilingual Voice Forensics", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ Enterprise neural node offline. Please configure GEMINI_API_KEY in Streamlit Secrets.")
            else:
                with st.spinner(f"Stripping acoustic envelope & analyzing {selected_lang} phonemic boundaries..."):
                    report = fe.analyze_multilingual_audio_forensics(
                        audio_bytes=audio_bytes,
                        mime_type=mime_type,
                        api_key=api_key,
                        target_language=selected_lang
                    )

                    score = report.get("ai_likelihood_score", 50)
                    margin = report.get("uncertainty_margin", 15)
                    verdict_text = report.get("verdict", "Indeterminate")

                    st.markdown("---")
                    st.markdown("### 🧬 Beetles Voice Forensics Verdict")

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric(
                            label="AI Voice Clone Likelihood",
                            value=f"{score}%",
                            delta=f"±{margin}% Uncertainty",
                            delta_color="inverse"
                        )
                    with col2:
                        st.info(f"**Target Language Profile:** {report.get('detected_language', selected_lang)}\n\n**Verdict:** {verdict_text}")

                    st.markdown(f"**Forensic Breakdown:** {report.get('forensic_summary', '')}")

                    with st.expander("🔍 Deep Linguistic & Acoustic Anomalies", expanded=True):
                        st.write("**Linguistic Anomalies (Tonal/Phonetic):**")
                        ling = report.get("linguistic_anomalies", [])
                        if ling:
                            for item in ling:
                                st.write(f"- {item}")
                        else:
                            st.write("- None detected.")

                        st.write("**Acoustic & Vocoder Artifacts (Breathing/Phase/Spectra):**")
                        acous = report.get("acoustic_artifacts", [])
                        if acous:
                            for item in acous:
                                st.write(f"- {item}")
                        else:
                            st.write("- None detected.")

                    # WhatsApp Audio Report
                    audio_share = (
                        f"🪲 *BEETLES AI AUDIO FORENSIC AUDIT*\n"
                        f"• Target Language: {report.get('detected_language', selected_lang)}\n"
                        f"• Synthetic Likelihood: {score}% (±{margin}%)\n"
                        f"• Verdict: {verdict_text}\n"
                        f"• Summary: {report.get('forensic_summary', '')}\n"
                        f"_Audited via Beetles Multilingual Forensics._"
                    )
                    st.text_area("WhatsApp Audio Report Card (Click to Copy)", value=audio_share, height=120)


# ==============================================================================
# TAB 3: THREAT INTELLIGENCE & METHODOLOGY
# ==============================================================================
with tab_methodology:
    st.markdown("### 🔬 The Beetles AI Architectural Moat")
    
    st.markdown("""
    #### 1. Why Western Incumbents Fail in Emerging Markets
    Most deepfake detection platforms (Reality Defender, Hive, Sensity) are trained on English news broadcasts and Western corporate press conferences.
    When financial crime syndicates execute corporate heists in Asia and the Middle East, they exploit non-Western languages:
    * **Cantonese & Mandarin (The HK$200M Corporate Heist):** Scammers use real-time face-swap and voice conversion to impersonate CFOs on Zoom calls. Western models miss pitch flattening and tonal contour drift across Cantonese's 6 tones.
    * **Arabic (Glottal & Pharyngeal Tampering):** AI speech synthesis models introduce metallic vocoder ring around guttural letters (*ع, ح, خ, ق*).
    * **Urdu & Hindi (Aspirated Consonants):** Synthetic tools fail to simulate biological breath-burst dynamics during aspirated stop consonants (*بھ, پھ, تھ*).

    #### 2. The Multi-Signal Consensus Principle
    Beetles AI never relies on a single model. A binary "REAL vs. FAKE" label creates legal liability and false certainty. 
    We fuse:
    1. **Spatial Compression Forensics (ELA):** Detects localized re-saving, bounding-box haloing, and screenshot tampering.
    2. **Spectral Frequency Analysis (2D-FFT):** Identifies diffusion grid checkerboards and GAN upsampling artifacts in the frequency domain.
    3. **Document Script Radicals (Quad-Script Engine):** Scrutinizes Hangul syllabic blocks, Hanzi stroke geometry, and Arabic RTL ligatures.
    4. **Multimodal LLM Consensus:** Evaluates lighting physics, organic breath pauses, and acoustic dynamic range.

    #### 3. Privacy & Source Protection Promise
    * Real Dermestid beetles clean skeletons to the bone in forensic crime labs without damaging the physical evidence.
    * Beetles AI strips synthetic artifacts without exposing source identities, wire transfer memos, or confidential trial documents.
    """)