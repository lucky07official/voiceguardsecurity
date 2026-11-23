# VoiceGuard AI — v3.0.0 Account Edition

## What's new
- Self-service account registration and sign-in.
- Registration fields: full name, username, Gmail address, mobile number, password, and optional profile picture.
- Passwords are stored as PBKDF2-HMAC-SHA256 hashes with unique random salts.
- Usernames and Gmail addresses are unique (case-insensitive).
- Five failed sign-in attempts trigger a 60-second session lock.
- Dashboard and audio tools remain hidden until sign-in.
- Existing audio playback, deepfake screening, charts, transcription, reports and session history remain.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Open the local URL shown by Streamlit, choose **Create account**, register, then sign in.

## Data and security notes
- The app creates `voiceguard_users.db` beside `app.py`. This database contains account profile data and password hashes. Keep it private and back it up securely.
- Profile pictures are optional and limited to 2 MB.
- Gmail and mobile numbers are collected but **not verified** in this prototype. Do not describe them as verified.
- SQLite on a local disk is for a local demo/single-server prototype. Streamlit Cloud's local filesystem may be ephemeral and multiple app instances may not share the same database. For a real public launch, connect a managed persistent database (for example Supabase/Postgres), add email verification, HTTPS, rate limiting shared across sessions, account recovery, privacy policy, retention/deletion controls and backups.
- Do not commit `voiceguard_users.db`, uploaded private profile photos, secrets, or audio recordings to a public Git repository.
- The detector scores are raw model outputs, not calibrated probabilities or proof of authenticity.
