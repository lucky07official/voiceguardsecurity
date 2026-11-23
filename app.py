import io
import os
import json
import hashlib
import hmac
import time
import tempfile
import mimetypes
import sqlite3
import re
import secrets
from pathlib import Path
from datetime import datetime
from PIL import Image

import numpy as np
import pandas as pd
import librosa
import librosa.display
import streamlit as st
import matplotlib.pyplot as plt
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

TITLE = "VoiceGuard AI · v3.0"
APP_VERSION = "3.0.0"
DETECTOR_ID = "garystafford/wav2vec2-deepfake-voice-detector"
ASR_ID = "openai/whisper-tiny"
TARGET_SR = 16000

st.set_page_config(page_title=TITLE, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# ---------- Visual design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#eaf2ff; --muted:#9fb0c9; --panel:#111d31; --line:rgba(148,163,184,.18); --cyan:#42d9e8; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #153252 0%, #0a1323 42%, #080f1b 100%); color:var(--ink); }
.block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#f4f8ff !important; }
.hero { padding:28px 30px; border:1px solid rgba(66,217,232,.28); border-radius:22px; background:linear-gradient(115deg,rgba(20,54,82,.94),rgba(16,29,49,.92)); box-shadow:0 18px 55px rgba(0,0,0,.22); margin-bottom:22px; }
.hero-kicker { color:var(--cyan); font-size:.78rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; }
.hero-title { font:700 2.25rem 'Space Grotesk',sans-serif; margin:8px 0 5px; color:#fff; }
.hero-sub { color:#b7c7dc; font-size:1rem; margin:0; }
.audio-preview { background:rgba(17,29,49,.88); border:1px solid rgba(66,217,232,.28); border-radius:16px; padding:16px 18px; margin:12px 0 18px; }
.section-kicker { color:#42d9e8; font-size:.76rem; letter-spacing:1.5px; font-weight:700; text-transform:uppercase; }
div[data-testid="stAudio"] { width:100%; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:#14243a; border-radius:10px 10px 0 0; padding:10px 16px; }
.pill { display:inline-block; border:1px solid rgba(66,217,232,.35); color:#9df3f4; border-radius:99px; padding:5px 11px; font-size:.75rem; margin:14px 7px 0 0; background:rgba(66,217,232,.07); }
.feature { background:rgba(17,29,49,.86); border:1px solid var(--line); padding:15px 16px; border-radius:14px; min-height:100px; }
.feature b { color:#f4f8ff; display:block; margin-bottom:5px; }
.feature span { color:var(--muted); font-size:.84rem; }
.metric-card { background:linear-gradient(145deg,rgba(22,43,67,.95),rgba(15,27,45,.95)); border:1px solid var(--line); border-radius:16px; padding:17px 18px; }
.metric-label { color:var(--muted); font-size:.8rem; }
.metric-value { color:#fff; font:700 1.55rem 'Space Grotesk',sans-serif; margin-top:5px; }
div[data-testid="stFileUploader"], div[data-testid="stAudioInput"] { background:rgba(17,29,49,.65); border:1px dashed rgba(66,217,232,.35); border-radius:14px; padding:12px; }
.stButton>button { border-radius:10px; border:1px solid rgba(66,217,232,.45); background:linear-gradient(90deg,#087f91,#13a7ad); color:white; font-weight:700; min-height:42px; }
.stButton>button:hover { border-color:#8af4f4; color:white; }
section[data-testid="stSidebar"] { background:#0b1627; border-right:1px solid var(--line); }
small, .muted { color:var(--muted); }
hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)

# ---------- Public user accounts (SQLite + PBKDF2) ----------
DB_PATH = Path(os.getenv("VOICEGUARD_DB_PATH", str(Path(__file__).resolve().parent / "voiceguard_users.db")))
MAX_PROFILE_BYTES = 2 * 1024 * 1024


def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db():
    with db_connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            username TEXT NOT NULL COLLATE NOCASE UNIQUE,
            email TEXT NOT NULL COLLATE NOCASE UNIQUE,
            mobile TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            profile_image BLOB,
            created_at TEXT NOT NULL
        )""")


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 310_000).hex()
    return digest, salt


def create_account(full_name, username, email, mobile, password, profile_image):
    digest, salt = hash_password(password)
    try:
        with db_connect() as conn:
            conn.execute("""INSERT INTO users
                (full_name, username, email, mobile, password_hash, password_salt, profile_image, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (full_name.strip(), username.strip(), email.strip().lower(), mobile.strip(), digest, salt,
                 profile_image, datetime.now().isoformat(timespec="seconds")))
        return True, "Account created. You can now sign in."
    except sqlite3.IntegrityError:
        return False, "That username or Gmail is already registered. Try signing in or use another one."


def authenticate_user(identifier, password):
    with db_connect() as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ? OR email = ?",
                            (identifier.strip(), identifier.strip().lower())).fetchone()
    if not user:
        # Keep a similar password-hash workload for unknown usernames.
        hash_password(password, "voiceguard_unknown_user_salt")
        return None
    candidate, _ = hash_password(password, user["password_salt"])
    if hmac.compare_digest(candidate, user["password_hash"]):
        return {"id": user["id"], "full_name": user["full_name"], "username": user["username"],
                "email": user["email"], "mobile": user["mobile"], "profile_image": user["profile_image"]}
    return None


init_user_db()

if "vg_user" not in st.session_state:
    st.session_state.vg_user = None
if "vg_login_failures" not in st.session_state:
    st.session_state.vg_login_failures = 0
if "vg_lock_until" not in st.session_state:
    st.session_state.vg_lock_until = 0.0

if not st.session_state.vg_user:
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">Create your account · Secure access</div>
      <div class="hero-title">🛡️ VoiceGuard AI</div>
      <p class="hero-sub">Create a personal account or sign in to access AI voice screening.</p>
      <span class="pill">Personal account</span><span class="pill">Password protected</span><span class="pill">Version 3.0.0</span>
    </div>
    """, unsafe_allow_html=True)
    register_tab, login_tab = st.tabs(["✨ Create account", "🔐 Sign in"])
    with register_tab:
        st.markdown("### Join VoiceGuard AI")
        st.caption("Your profile details are stored in this app's user database. Email and mobile are not verified in this version.")
        with st.form("voiceguard_register_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                full_name = st.text_input("Full name", max_chars=80, placeholder="Your name")
                username = st.text_input("Username", max_chars=24, placeholder="Choose a username")
                email = st.text_input("Gmail address", max_chars=254, placeholder="name@gmail.com")
            with c2:
                mobile = st.text_input("Mobile number", max_chars=16, placeholder="10-digit mobile number")
                new_password = st.text_input("Password", type="password", help="Use at least 12 characters.")
                confirm_password = st.text_input("Confirm password", type="password")
            profile_file = st.file_uploader("Profile picture (optional, JPG/PNG/WebP; max 2 MB)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False)
            agree = st.checkbox("I understand that my account details are stored by this app and I will only upload a picture I have permission to use.")
            create_submitted = st.form_submit_button("Create my account", type="primary", use_container_width=True)
        if create_submitted:
            errors = []
            if len(full_name.strip()) < 2: errors.append("Enter your full name.")
            if not re.fullmatch(r"[A-Za-z0-9_.]{3,24}", username.strip()): errors.append("Username must be 3–24 characters and use only letters, numbers, dot or underscore.")
            if not re.fullmatch(r"[A-Za-z0-9._%+-]+@gmail\.com", email.strip(), re.I): errors.append("Enter a valid @gmail.com address.")
            if not re.fullmatch(r"\+?[0-9]{10,15}", mobile.strip()): errors.append("Enter a valid mobile number with 10–15 digits (optional + at the start).")
            if len(new_password) < 12: errors.append("Password must be at least 12 characters.")
            if new_password != confirm_password: errors.append("Passwords do not match.")
            if not agree: errors.append("Please accept the account data notice to continue.")
            photo_bytes = None
            if profile_file is not None:
                photo_bytes = profile_file.getvalue()
                if len(photo_bytes) > MAX_PROFILE_BYTES:
                    errors.append("Profile picture must be 2 MB or smaller.")
                else:
                    try:
                        img = Image.open(io.BytesIO(photo_bytes))
                        img.verify()
                        if img.format not in ("JPEG", "PNG", "WEBP"):
                            errors.append("Use a JPG, PNG or WebP profile picture.")
                    except Exception:
                        errors.append("The selected profile picture is not a valid image.")
            if errors:
                for error in errors: st.error(error)
            else:
                ok, message = create_account(full_name, username, email, mobile, new_password, photo_bytes)
                if ok:
                    st.success(message + " Open the Sign in tab.")
                else:
                    st.error(message)
    with login_tab:
        st.markdown("### Welcome back")
        remaining = st.session_state.vg_lock_until - time.time()
        with st.form("voiceguard_login_form", clear_on_submit=True):
            identifier = st.text_input("Username or Gmail")
            login_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Sign in to VoiceGuard AI", type="primary", use_container_width=True)
        if remaining > 0:
            st.warning(f"Too many failed attempts. Try again in {int(remaining) + 1} seconds.")
        elif login_submitted:
            user = authenticate_user(identifier, login_password)
            if user:
                st.session_state.vg_user = user
                st.session_state.vg_login_failures = 0
                st.session_state.vg_lock_until = 0.0
                st.rerun()
            else:
                st.session_state.vg_login_failures += 1
                if st.session_state.vg_login_failures >= 5:
                    st.session_state.vg_lock_until = time.time() + 60
                    st.session_state.vg_login_failures = 0
                    st.error("Too many incorrect attempts. Sign in is paused for 60 seconds.")
                else:
                    st.error(f"Incorrect username/Gmail or password. {5 - st.session_state.vg_login_failures} attempts remaining.")
        st.caption("You must create an account and sign in before the dashboard or audio tools become available.")
    st.stop()

FEATURES = [
    ("🎙️", "Live recording & upload", "Record audio or upload common audio formats."),
    ("📚", "Batch audio analysis", "Analyze multiple recordings in one session."),
    ("🧠", "AI deepfake screening", "Run an existing pretrained audio-classification model."),
    ("📊", "Real / fake score breakdown", "Review the model's raw class scores."),
    ("〰️", "Waveform & spectrogram", "Explore audio patterns and acoustic metrics."),
    ("💬", "Speech transcription", "Optional Whisper Tiny transcript."),
    ("📄", "Export & session history", "Download reports and review this session's results."),
]

st.markdown("""
<div class="hero">
  <div class="hero-kicker">Audio authenticity screening platform · Version 3.0.0</div>
  <div class="hero-title">🛡️ VoiceGuard AI</div>
  <p class="hero-sub">AI-powered voice clone and deepfake audio screening assistant.</p>
  <span class="pill">Wav2Vec2 detector</span><span class="pill">Audio forensics</span><span class="pill">Privacy-aware workflow</span>
</div>
""", unsafe_allow_html=True)

with st.expander("✨ Explore the 7 project features", expanded=False):
    cols = st.columns(2)
    for i, (icon, name, desc) in enumerate(FEATURES):
        cols[i % 2].markdown(f'<div class="feature"><b>{icon} {i+1}. {name}</b><span>{desc}</span></div>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "results" not in st.session_state:
    st.session_state.results = {}

@st.cache_resource(show_spinner="Loading the AI detector (first run may take a few minutes)...")
def load_detector():
    extractor = AutoFeatureExtractor.from_pretrained(DETECTOR_ID)
    model = AutoModelForAudioClassification.from_pretrained(DETECTOR_ID)
    model.eval()
    return extractor, model

@st.cache_resource(show_spinner="Loading Whisper Tiny...")
def load_asr():
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=ASR_ID, device=-1)

def decode_audio(raw_bytes):
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(raw_bytes)
        path = tmp.name
    try:
        audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if audio.size == 0:
        raise ValueError("The audio file appears to be empty or unsupported.")
    return audio.astype(np.float32), sr

def get_label_map(model):
    mapping = getattr(model.config, "id2label", {}) or {}
    return {int(k): str(v) for k, v in mapping.items()}

def classify_audio(audio, sr):
    extractor, model = load_detector()
    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    labels = get_label_map(model)
    scores = {labels.get(i, f"class_{i}"): float(p) for i, p in enumerate(probs)}
    # The checkpoint commonly labels classes as REAL/FAKE; preserve the model's own labels.
    fake_key = next((k for k in scores if "fake" in k.lower() or "spoof" in k.lower()), None)
    real_key = next((k for k in scores if "real" in k.lower() or "bonafide" in k.lower() or "bonafide" in k.lower()), None)
    if fake_key is None or real_key is None:
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        predicted = ordered[0][0]
        return scores, predicted, None, None
    predicted = max(scores, key=scores.get)
    return scores, predicted, scores[real_key], scores[fake_key]

def audio_metrics(audio, sr):
    duration = len(audio) / sr
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    return {"Duration (seconds)": round(duration, 2), "Sample rate (Hz)": sr,
            "RMS energy": round(rms, 5), "Peak amplitude": round(peak, 5),
            "Zero-crossing rate": round(zcr, 5), "Spectral centroid (Hz)": round(centroid, 2)}

def make_pdf(result):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Subtle", parent=styles["Normal"], textColor=colors.HexColor("#526174"), leading=15))
    story = [Paragraph("VoiceGuard AI — Audio Screening Report", styles["Title"]), Spacer(1, 8),
             Paragraph(f"File: {result['filename']}<br/>Generated: {result['timestamp']}", styles["Subtle"]), Spacer(1, 18)]
    rows = [["Model output", result["prediction"]]]
    if result.get("real_score") is not None:
        rows += [["Real score (raw)", f"{result['real_score']*100:.2f}%"], ["Fake score (raw)", f"{result['fake_score']*100:.2f}%"]]
    rows += [[k, str(v)] for k, v in result["metrics"].items()]
    table = Table(rows, colWidths=[190, 290])
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8f3f7")), ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#cbd5e1")), ("PADDING",(0,0),(-1,-1),8), ("VALIGN",(0,0),(-1,-1),"TOP")]))
    story += [table, Spacer(1, 18), Paragraph("Important limitation", styles["Heading2"]),
              Paragraph("Scores are raw model outputs, not calibrated probabilities. This screening result is not proof of authenticity, identity, or fraud. False positives and false negatives are possible. Review the audio and use independent verification.", styles["BodyText"])]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def render_result(result, key):
    st.markdown(f"### 🔎 Analysis: {result['filename']}")
    st.markdown('<div class="section-kicker">Listen to analyzed recording</div>', unsafe_allow_html=True)
    st.audio(result["audio"], sample_rate=result["sr"])
    st.caption("Use the player above to hear the exact audio that was analyzed.")
    if result.get("real_score") is not None:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">REAL score · raw</div><div class="metric-value">{result["real_score"]*100:.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">FAKE score · raw</div><div class="metric-value">{result["fake_score"]*100:.2f}%</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">Top model label</div><div class="metric-value">{result["prediction"]}</div></div>', unsafe_allow_html=True)
    else:
        st.info(f"Top model label: **{result['prediction']}**. The checkpoint's class labels could not be mapped confidently to Real/Fake.")
    if result.get("scores"):
        st.write("**Model class scores (raw)**")
        st.bar_chart(pd.DataFrame({"Score": result["scores"]}))
    with st.expander("🎚️ Audio waveform and spectrogram", expanded=True):
        a = np.asarray(result["audio"])
        sr = result["sr"]
        fig, ax = plt.subplots(figsize=(11, 2.6))
        librosa.display.waveshow(a, sr=sr, ax=ax)
        ax.set_title("Waveform"); ax.set_xlabel("Time (seconds)")
        st.pyplot(fig, clear_figure=True)
        spec = librosa.feature.melspectrogram(y=a, sr=sr, n_mels=64)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        fig2, ax2 = plt.subplots(figsize=(11, 3.4))
        img = librosa.display.specshow(spec_db, sr=sr, x_axis="time", y_axis="mel", ax=ax2)
        ax2.set_title("Mel spectrogram"); fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        st.pyplot(fig2, clear_figure=True)
    st.write("**Acoustic metrics**")
    st.dataframe(pd.DataFrame([result["metrics"]]), use_container_width=True, hide_index=True)
    d1, d2, d3 = st.columns(3)
    safe = result["filename"].rsplit(".", 1)[0].replace(" ", "_")
    d1.download_button("⬇️ Download PDF", make_pdf(result), file_name=f"{safe}_voiceguard_report.pdf", mime="application/pdf", key=f"pdf_{key}")
    d2.download_button("⬇️ Download JSON", json.dumps({k:v for k,v in result.items() if k not in ("audio", "raw_bytes")}, indent=2, default=str), file_name=f"{safe}_report.json", mime="application/json", key=f"json_{key}")
    d3.download_button("⬇️ Download CSV", pd.DataFrame([{**{k:v for k,v in result.items() if k not in ("audio", "raw_bytes", "scores")}, "scores": json.dumps(result["scores"])}]).to_csv(index=False), file_name=f"{safe}_report.csv", mime="text/csv", key=f"csv_{key}")
    if st.button("💬 Generate optional transcript", key=f"asr_{key}"):
        try:
            with st.spinner("Transcribing audio with Whisper Tiny..."):
                asr = load_asr()
                transcript = asr(a)["text"]
            st.session_state.results[key]["transcript"] = transcript
            st.success("Transcript generated")
            st.write(transcript)
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
    if result.get("transcript"):
        st.info("Transcript (may contain recognition errors): " + result["transcript"])

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🛡️ VoiceGuard AI")
    st.caption("Audio screening workspace")
    st.markdown("---")
    st.markdown("**Detector**")
    st.caption("garystafford / wav2vec2-deepfake-voice-detector")
    user = st.session_state.vg_user
    if user.get("profile_image"):
        st.image(user["profile_image"], width=76)
    st.markdown(f"**{user['full_name']}**")
    st.caption(f"@{user['username']} · {user['email']}")
    st.markdown("**Session**")
    st.metric("Analyzed recordings", len(st.session_state.history))
    if st.button("🗑️ Clear session history", use_container_width=True):
        st.session_state.history = []
        st.session_state.results = {}
        st.rerun()
    st.markdown("---")
    st.caption(f"Version {APP_VERSION}")
    if st.button("🔒 Log out", use_container_width=True):
        st.session_state.vg_user = None
        st.session_state.history = []
        st.session_state.results = {}
        st.rerun()
    st.caption("Audio is processed in this app session. Avoid uploading recordings unless you have permission.")

# ---------- Input ----------
st.markdown("## 🎙️ Upload, record & listen")
st.write("Add a recording, listen to it immediately, then run the AI screening. Playback is available both before and after analysis.")
left, right = st.columns([1.2, 1])
with left:
    uploads = st.file_uploader("Choose audio files", type=["wav", "mp3", "m4a", "flac", "ogg", "aac"], accept_multiple_files=True)
with right:
    recording = None
    if hasattr(st, "audio_input"):
        recording = st.audio_input("Record a voice sample")
    else:
        st.info("Live recording needs a newer Streamlit version. File upload is available.")

# Immediate playback preview: users can listen before running AI analysis.
selected_count = len(uploads or []) + (1 if recording is not None else 0)
if selected_count:
    st.markdown("<div class='audio-preview'><div class='section-kicker'>Audio preview · Listen before analysis</div></div>", unsafe_allow_html=True)
    preview_tabs = st.tabs([f"🎧 {getattr(f, 'name', f'Audio {i+1}')[:28]}" for i, f in enumerate(list(uploads or []) + ([recording] if recording is not None else []))])
    preview_items = list(uploads or []) + ([recording] if recording is not None else [])
    for tab, audio_item in zip(preview_tabs, preview_items):
        with tab:
            item_name = getattr(audio_item, "name", "recording.wav")
            audio_mime = mimetypes.guess_type(item_name)[0] or "audio/wav"
            st.audio(audio_item.getvalue(), format=audio_mime)
            st.caption("Playback preview — this is the recording that will be sent for analysis.")

if st.button("🚀 Analyze selected audio", type="primary", use_container_width=True, disabled=not (uploads or recording)):
    items = list(uploads or [])
    if recording is not None:
        items.append(recording)
    progress = st.progress(0, text="Preparing audio...")
    for idx, item in enumerate(items):
        raw = item.getvalue()
        filename = getattr(item, "name", None) or f"recording_{idx+1}.wav"
        digest = hashlib.sha256(raw).hexdigest()
        key = digest[:12]
        try:
            progress.progress((idx) / len(items), text=f"Analyzing {filename}...")
            audio, sr = decode_audio(raw)
            scores, prediction, real_score, fake_score = classify_audio(audio, sr)
            result = {"filename": filename, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      "prediction": prediction, "real_score": real_score, "fake_score": fake_score,
                      "scores": scores, "metrics": audio_metrics(audio, sr), "audio": audio,
                      "sr": sr, "raw_bytes": raw, "sha256": digest}
            st.session_state.results[key] = result
            if not any(x.get("sha256") == digest for x in st.session_state.history):
                st.session_state.history.append({"filename": filename, "timestamp": result["timestamp"],
                    "prediction": prediction, "real_score": real_score, "fake_score": fake_score, "sha256": digest})
        except Exception as exc:
            st.error(f"Could not analyze {filename}: {exc}")
    progress.progress(1.0, text="Analysis complete")
    st.success(f"Finished processing {len(items)} recording(s).")

# ---------- Results and history ----------
if st.session_state.results:
    st.markdown("---")
    st.markdown("## Analysis results")
    for key, result in list(st.session_state.results.items())[::-1]:
        with st.container(border=True):
            render_result(result, key)

if st.session_state.history:
    st.markdown("---")
    st.markdown("## Recent session history")
    hist = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(hist.drop(columns=["sha256"], errors="ignore"), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Export history CSV", hist.to_csv(index=False), "voiceguard_session_history.csv", "text/csv")

with st.expander("🔐 Privacy, interpretation & model limitations"):
    st.markdown("""
    - The displayed scores are raw softmax outputs, **not calibrated probabilities**.
    - Results are not proof of authenticity, identity, or fraud. False positives and false negatives are possible.
    - Performance may vary by language, speaker, microphone, codec, recording quality, and synthesis method.
    - Account name, username, Gmail, mobile number, password hash and optional profile picture are stored in the local SQLite user database. Passwords are PBKDF2-hashed; raw passwords are not stored.
    - This prototype does not verify email addresses or mobile numbers. SQLite on a local disk is suitable for a local demo, not a multi-instance public production service. Use a managed persistent database, verified email flow, HTTPS, backups and privacy policy before public launch.
    - Session history is stored in Streamlit session state and is not written to a database by this app. Hosting providers, browsers, operating systems, and networks may have their own retention behavior.
    - Whisper transcripts can contain recognition errors and are not evidence of authenticity.
    - Only analyze recordings you have permission to use. Review the model card and license before public or commercial deployment.
    """)
    st.caption(f"Detector model: {DETECTOR_ID} · Transcript model: {ASR_ID}")

st.markdown("<hr><p style='text-align:center;color:#8193ad;font-size:.8rem'>VoiceGuard AI · Account-enabled academic project prototype · Use results as screening signals only</p>", unsafe_allow_html=True)
