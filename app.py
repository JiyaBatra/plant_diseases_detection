import os
import re
import json
import base64
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import certifi
import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from bson import ObjectId
from PIL import Image, ImageOps
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError


LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="{s}" height="{s}">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#9BE0BC"/><stop offset="1" stop-color="#4FAE80"/></linearGradient></defs>'
    '<rect width="48" height="48" rx="12" fill="url(#g)"/>'
    '<path d="M24 9 C37 10 40 23 34 32 C30 38 21 40 13 36 C9 26 13 13 24 9 Z" fill="#08261A"/>'
    '<path d="M15 35 C20 28 25 22 31 16" stroke="#9BE0BC" stroke-width="2.4" '
    'stroke-linecap="round" fill="none"/>'
    '<path d="M22 27 L28 27 M26 22 L31 22" stroke="#9BE0BC" stroke-width="1.8" stroke-linecap="round"/>'
    '</svg>'
)


def logo(size=40):
    return LOGO_SVG.format(s=size)


def make_favicon():
    """Draw the same leaf mark as a PNG for the browser tab."""
    from PIL import ImageDraw
    s = 256
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, s, s), radius=64, fill=(116, 198, 157, 255))
    # leaf outline approximated with a polygon (two curved edges)
    pts = []
    for t in np.linspace(0, 1, 40):          # right edge
        pts.append((128 + 82 * np.sin(np.pi * t) * 0.9 + 20 * t, 48 + 160 * t))
    for t in np.linspace(1, 0, 40):          # left edge
        pts.append((128 - 70 * np.sin(np.pi * t) * 0.9 + 20 * t, 48 + 160 * t))
    d.polygon(pts, fill=(8, 38, 26, 255))
    d.line([(88, 190), (168, 90)], fill=(155, 224, 188, 255), width=12)
    return img


# ==================================================
# 1. PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="PlantCare AI - Plant Disease Detection",
    page_icon=make_favicon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "plant_disease_cnn.keras")
CLASS_PATH = os.path.join(BASE_DIR, "saved_model", "class_names.json")
BACKGROUND_PATH = os.path.join(BASE_DIR, "assets", "plant_app.png")
IMG_SIZE = 128
DEFAULT_LOW_CONFIDENCE = 0.60

PAGE_DASH = "Dashboard"
PAGE_DETECT = "Detect Disease"
PAGE_PLAN = "30-Day Care Plan"
PAGE_HISTORY = "History & Reports"
PAGE_LIBRARY = "Disease Library"
PAGE_TIPS = "Plant Care Guide"
PAGE_PROFILE = "Profile & Settings"
PAGES = [PAGE_DASH, PAGE_DETECT, PAGE_PLAN, PAGE_HISTORY, PAGE_LIBRARY, PAGE_TIPS, PAGE_PROFILE]

# Palette (kept in one place so the CSS and charts stay consistent)
C_BG = "#0A1B14"
C_PANEL = "#122E22"
C_PRIMARY = "#74C69D"
C_PRIMARY_DARK = "#08261A"
C_TEXT = "#EAF4EC"
C_MUTED = "#A9C4B5"
C_WARN = "#F4B860"
C_DANGER = "#F07167"


# ==================================================
# 2. STYLING
# ==================================================
@st.cache_data
def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

html, body, [class*="css"], .stMarkdown, .stApp { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.01em; color: #FFFFFF; }
h1 { font-size: 2.3rem !important; }
h4, h5, h6 { color: #FFFFFF; font-weight: 600; }
p, li, label, span, .stMarkdown { color: #EAF4EC; }
.stCaption, small, [data-testid="stCaptionContainer"] { color: #A9C4B5 !important; }

/* App background: photo (if present) under a strong dark overlay for readable text */
.stApp {
    __BG__
    background-size: cover; background-position: center; background-attachment: fixed;
}
header[data-testid="stHeader"] { background: rgba(10,27,20,.85); backdrop-filter: blur(6px); }
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }

/* Hero banner at the top of each page */
.hero {
    background: linear-gradient(120deg, rgba(18,46,34,.96) 0%, rgba(11,36,26,.92) 100%);
    border: 1px solid rgba(116,198,157,.30); border-left: 6px solid #74C69D;
    border-radius: 16px; padding: 1.4rem 1.7rem; margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(0,0,0,.35);
}
.hero h1 { margin: 0 0 .3rem 0; padding: 0; line-height: 1.15; }
.hero p { margin: 0; color: #C9E0D2; font-size: 1.05rem; }

/* Generic panel */
.panel {
    background: rgba(18,46,34,.85); border: 1px solid rgba(116,198,157,.22);
    border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: .8rem;
}
.panel b { color: #FFFFFF; }
.pill {
    display: inline-block; padding: 2px 12px; border-radius: 999px; font-size: .82rem;
    font-weight: 600; margin-right: 6px; border: 1px solid rgba(255,255,255,.18);
}
.pill-high { background: rgba(240,113,103,.18); color: #FFB4AC; }
.pill-medium { background: rgba(244,184,96,.18); color: #FFD9A0; }
.pill-none { background: rgba(116,198,157,.18); color: #B7E4C7; }
.pill-cat { background: rgba(255,255,255,.08); color: #EAF4EC; }

/* Metric tiles */
div[data-testid="stMetric"] {
    background: rgba(18,46,34,.88); border: 1px solid rgba(116,198,157,.25);
    border-top: 3px solid #74C69D; border-radius: 12px; padding: 14px 16px;
}
div[data-testid="stMetricLabel"] p { color: #B7E4C7 !important; font-weight: 500; }
div[data-testid="stMetricValue"] { color: #FFFFFF; font-weight: 700; }

/* Buttons: dark text on mint so it is always readable */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(180deg, #8AD6AE 0%, #74C69D 100%);
    border: none; border-radius: 10px; min-height: 46px; width: 100%;
    box-shadow: 0 3px 10px rgba(0,0,0,.30); transition: transform .08s ease, filter .15s ease;
}
div.stButton > button p, div[data-testid="stFormSubmitButton"] > button p,
div[data-testid="stDownloadButton"] > button p {
    color: #08261A !important; font-weight: 700; font-size: 1rem;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover { filter: brightness(1.08); transform: translateY(-1px); }
div.stButton > button:focus-visible { outline: 3px solid #FFFFFF; outline-offset: 2px; }

/* Inputs and uploaders */
div[data-testid="stFileUploader"], div[data-testid="stCameraInput"] {
    background: rgba(18,46,34,.85); border: 1px dashed rgba(116,198,157,.55);
    border-radius: 14px; padding: 14px;
}
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, textarea {
    background-color: rgba(8,24,17,.9) !important; border-color: rgba(116,198,157,.35) !important;
}
input, textarea { color: #FFFFFF !important; }

/* Tabs, expanders */
button[data-baseweb="tab"] p { font-weight: 600; font-size: 1rem; }
div[data-testid="stExpander"] {
    background: rgba(18,46,34,.80); border: 1px solid rgba(116,198,157,.22); border-radius: 12px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071811 0%, #0C2419 100%);
    border-right: 1px solid rgba(116,198,157,.22);
}
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 10px 12px; border-radius: 10px; width: 100%; cursor: pointer;
    border: 1px solid transparent; transition: background .15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(116,198,157,.12); }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(116,198,157,.22); border-color: rgba(116,198,157,.55);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 1rem; font-weight: 500; }
.brand { font-family: 'Fraunces', Georgia, serif; font-size: 1.6rem; font-weight: 700; color: #FFFFFF; }
.brand-row { display: flex; align-items: center; gap: 12px; }
.brand-row .brand { line-height: 1.1; font-size: 1.35rem; }
.brand-sub { color: #A9C4B5; font-size: .9rem; margin-top: -4px; }

/* Progress bars */
div[data-testid="stProgress"] > div > div > div > div { background-color: #74C69D; }

/* Footer */
.footer { text-align: center; color: #8FAE9D; font-size: .88rem; padding-top: .5rem; }
</style>
"""


def inject_css():
    if os.path.exists(BACKGROUND_PATH):
        bg = (
            "background-image: linear-gradient(rgba(6,18,12,.86), rgba(6,18,12,.93)),"
            f'url("data:image/png;base64,{_encode_image(BACKGROUND_PATH)}");'
        )
    else:
        bg = f"background: linear-gradient(160deg, {C_BG}, #0F2A1F);"
    st.markdown(CSS.replace("__BG__", bg), unsafe_allow_html=True)


def hero(title, subtitle, with_logo=False):
    mark = f'<div style="margin-bottom:.7rem">{logo(56)}</div>' if with_logo else ""
    st.markdown(f'<div class="hero">{mark}<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def pill(text, kind="cat"):
    return f'<span class="pill pill-{kind}">{text}</span>'


def show_df(df):
    """dataframe() with full width, compatible with old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def goto(page):
    st.session_state["nav"] = page


# ==================================================
# 3. CONFIG HELPER + DATABASE (MongoDB)
# ==================================================
def get_secret(key, default=None):
    """Render env var first, then Streamlit secrets, then default. Never crashes."""
    value = os.environ.get(key)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        return default


@st.cache_resource(show_spinner="Connecting to database...")
def get_db():
    """Connect to MongoDB using Render environment variables or Streamlit secrets."""
    mongo_uri = get_secret("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI is missing. Add it in Render Environment Variables.")

    db_name = get_secret("DB_NAME", "plant_disease_app")

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=8000,
        tlsCAFile=certifi.where(),
        tz_aware=True,
    )
    client.admin.command("ping")
    db = client[db_name]

    db.users.create_index("email", unique=True)
    db.scans.create_index([("user_id", 1), ("created_at", -1)])
    db.plans.create_index([("user_id", 1), ("status", 1)])
    db.checkins.create_index([("plan_id", 1), ("created_at", 1)])

    # password reset tokens (only a SHA-256 hash of each token is stored)
    db.password_resets.create_index("token_hash", unique=True)
    db.password_resets.create_index([("user_id", 1), ("created_at", -1)])
    db.password_resets.create_index("expires_at", expireAfterSeconds=0)  # auto-delete when expired

    return db


def now():
    return datetime.now(timezone.utc)


def fmt(dt):
    return dt.strftime("%d %b %Y, %H:%M UTC")


def uid():
    return st.session_state["user"]["id"]


# ==================================================
# 4. AUTHENTICATION
# ==================================================
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def signup(name, email, password, confirm):
    name, email = name.strip(), email.strip().lower()
    if len(name) < 2:
        return False, "Please enter your name."
    if not EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if password != confirm:
        return False, "Passwords do not match."
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        get_db().users.insert_one({
            "name": name, "email": email,
            "password_hash": hashed, "created_at": now(),
        })
    except DuplicateKeyError:
        return False, "This email is already registered. Please log in."
    return True, "Account created. You can log in now."


def login(email, password):
    user = get_db().users.find_one({"email": email.strip().lower()})
    if not user or not bcrypt.checkpw(password.encode(), user["password_hash"]):
        return False, "Invalid email or password."
    st.session_state["user"] = {
        "id": str(user["_id"]), "name": user["name"], "email": user["email"],
    }
    return True, "Welcome back!"


def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# ---------- Forgot password (Resend email) ----------
RESEND_URL = "https://api.resend.com/emails"
RESET_TTL_MINUTES = 30
RESET_COOLDOWN_SECONDS = 60
GENERIC_RESET_MSG = ("If an account exists for that email, a password reset link has been sent. "
                     "Check your inbox (and spam folder). The link is valid for "
                     f"{RESET_TTL_MINUTES} minutes.")


def _hash_token(token):
    return hashlib.sha256(token.encode()).hexdigest()


def _reset_email_html(name, link):
    return f"""
    <div style="font-family:Arial,Helvetica,sans-serif;background:#f3f7f4;padding:24px;">
      <div style="max-width:520px;margin:auto;background:#ffffff;border-radius:12px;
                  padding:32px;border-top:6px solid #74C69D;">
        <h2 style="margin:0 0 12px;color:#0A1B14;">Reset your password</h2>
        <p style="color:#333;font-size:15px;line-height:1.6;">Hi {name},</p>
        <p style="color:#333;font-size:15px;line-height:1.6;">
          We received a request to reset your PlantCare AI password. Click the button below to
          choose a new one. This link expires in {RESET_TTL_MINUTES} minutes and works only once.
        </p>
        <p style="text-align:center;margin:28px 0;">
          <a href="{link}" style="background:#74C69D;color:#08261A;text-decoration:none;
             font-weight:bold;padding:13px 28px;border-radius:8px;display:inline-block;">Reset password</a>
        </p>
        <p style="color:#666;font-size:13px;line-height:1.6;">
          If the button does not work, copy this link into your browser:<br>
          <span style="word-break:break-all;">{link}</span>
        </p>
        <p style="color:#666;font-size:13px;">
          If you did not request this, you can safely ignore this email. Your password will not change.
        </p>
      </div>
    </div>
    """


def send_reset_email(to_email, name, link):
    api_key = get_secret("RESEND_API_KEY")
    if not api_key:
        return False, "RESEND_API_KEY is missing. Add it in Render Environment Variables."
    sender = get_secret("RESEND_FROM", "PlantCare AI <onboarding@resend.dev>")
    try:
        resp = requests.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from": sender, "to": [to_email],
                "subject": "Reset your PlantCare AI password",
                "html": _reset_email_html(name, link),
                "text": f"Hi {name},\n\nReset your PlantCare AI password here (valid for "
                        f"{RESET_TTL_MINUTES} minutes):\n{link}\n\nIf you did not request this, ignore this email.",
            },
            timeout=15,
        )
    except requests.RequestException as err:
        return False, f"Could not reach Resend: {err}"
    if resp.status_code >= 400:
        return False, f"Resend error {resp.status_code}: {resp.text[:300]}"
    return True, "sent"


def request_password_reset(email):
    """Always returns the same public message so nobody can discover which emails are registered.
    Real errors are printed to the server console (Render Logs).
    For testing only: set env var DEBUG_RESET=1 on Render to also see the error on screen."""
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    try:
        db = get_db()
        user = db.users.find_one({"email": email})
        if not user:
            return True, GENERIC_RESET_MSG
        recent = db.password_resets.find_one({
            "user_id": str(user["_id"]),
            "created_at": {"$gt": now() - timedelta(seconds=RESET_COOLDOWN_SECONDS)},
        })
        if recent:
            return True, GENERIC_RESET_MSG

        token = secrets.token_urlsafe(32)
        db.password_resets.insert_one({
            "user_id": str(user["_id"]), "token_hash": _hash_token(token),
            "created_at": now(), "expires_at": now() + timedelta(minutes=RESET_TTL_MINUTES),
        })
        app_url = str(get_secret("APP_URL", "https://plant-diseases-detection-h1s8.onrender.com")).rstrip("/")
        ok, detail = send_reset_email(email, user["name"], f"{app_url}/?reset_token={token}")
        if not ok:
            print(f"[password reset] email failed for {email}: {detail}", flush=True)
            st.session_state["reset_debug"] = detail
            db.password_resets.delete_one({"token_hash": _hash_token(token)})
    except Exception as err:
        import traceback
        print("[password reset] unexpected error:\n" + traceback.format_exc(), flush=True)
        st.session_state["reset_debug"] = f"{type(err).__name__}: {err}"
    return True, GENERIC_RESET_MSG


def _find_valid_reset(token):
    return get_db().password_resets.find_one({
        "token_hash": _hash_token(token), "expires_at": {"$gt": now()},
    })


def reset_password(token, password, confirm):
    record = _find_valid_reset(token)
    if not record:
        return False, "This reset link is invalid or has expired. Please request a new one."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if password != confirm:
        return False, "Passwords do not match."
    db = get_db()
    db.users.update_one(
        {"_id": ObjectId(record["user_id"])},
        {"$set": {"password_hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt())}},
    )
    db.password_resets.delete_many({"user_id": record["user_id"]})  # one-time use
    return True, "Password updated. You can now log in with your new password."


def render_reset_form(token):
    st.subheader("Choose a new password")
    if not _find_valid_reset(token):
        st.error("This reset link is invalid or has expired. Please request a new one.")
        if st.button("Back to login"):
            st.query_params.clear()
            st.rerun()
        return
    with st.form("reset_form"):
        pwd = st.text_input("New password (min 8 characters)", type="password")
        conf = st.text_input("Confirm new password", type="password")
        if st.form_submit_button("Update password"):
            ok, msg = reset_password(token, pwd, conf)
            if ok:
                st.session_state["reset_done"] = msg
                st.query_params.clear()
                st.rerun()
            st.error(msg)
    if st.button("Cancel"):
        st.query_params.clear()
        st.rerun()


def render_auth():
    hero("PlantCare AI", "Upload a leaf photo, get a diagnosis and treatment steps, then track recovery over 30 days.", with_logo=True)
    f1, f2, f3 = st.columns(3)
    f1.markdown('<div class="panel"><b>Instant diagnosis</b><br>A CNN trained on leaf images '
                'identifies the plant and its disease in seconds.</div>', unsafe_allow_html=True)
    f2.markdown('<div class="panel"><b>Clear treatment steps</b><br>Get practical treatment and '
                'prevention advice for each condition.</div>', unsafe_allow_html=True)
    f3.markdown('<div class="panel"><b>30-day recovery plan</b><br>Daily tasks and check-ins show '
                'whether your plant is improving.</div>', unsafe_allow_html=True)
    st.write("")
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        token = st.query_params.get("reset_token")
        if token:
            render_reset_form(token)
            return

        if st.session_state.get("reset_done"):
            st.success(st.session_state.pop("reset_done"))

        tab_login, tab_signup, tab_forgot = st.tabs(["Log in", "Create account", "Forgot password"])
        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Log in"):
                    ok, msg = login(email, pwd)
                    if ok:
                        st.rerun()
                    st.error(msg)
            st.caption("Forgot your password? Open the 'Forgot password' tab.")
        with tab_signup:
            with st.form("signup_form"):
                name = st.text_input("Full name")
                email = st.text_input("Email", key="su_email")
                pwd = st.text_input("Password (min 8 characters)", type="password", key="su_pwd")
                conf = st.text_input("Confirm password", type="password")
                if st.form_submit_button("Create account"):
                    ok, msg = signup(name, email, pwd, conf)
                    (st.success if ok else st.error)(msg)
        with tab_forgot:
            st.write("Enter your account email and we will send you a link to reset your password.")
            with st.form("forgot_form"):
                email = st.text_input("Email", key="fp_email")
                if st.form_submit_button("Send reset link"):
                    with st.spinner("Sending..."):
                        ok, msg = request_password_reset(email)
                    (st.success if ok else st.error)(msg)
            # Testing only: shows the real reason when DEBUG_RESET=1 is set on Render
            debug_msg = st.session_state.pop("reset_debug", None)
            if debug_msg and get_secret("DEBUG_RESET") == "1":
                st.warning(f"DEBUG: {debug_msg}")


# ==================================================
# 5. MODEL + PREDICTION
# ==================================================
@st.cache_resource(show_spinner="Loading AI model...")
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def model_files_exist():
    return os.path.exists(MODEL_PATH) and os.path.exists(CLASS_PATH)


def preprocess_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def plant_key(class_name):
    return class_name.split("___")[0]


def compute_health(probs, class_name):
    """0-100 score: probability of the 'healthy' class of the same plant.
    Falls back to (1 - P(class)) if that plant has no healthy class."""
    names = load_class_names()
    plant = plant_key(class_name).lower()
    for i, n in enumerate(names):
        if n.lower().startswith(plant) and "healthy" in n.lower():
            return float(probs[i]) * 100
    return (1 - float(probs[names.index(class_name)])) * 100


def predict(image):
    names = load_class_names()
    probs = load_trained_model().predict(preprocess_image(image), verbose=0)[0]
    if len(probs) != len(names):
        raise ValueError("Model output does not match class_names.json")
    top = np.argsort(probs)[::-1][:3]
    best = int(top[0])
    return {
        "class_name": names[best],
        "confidence": float(probs[best]),
        "top3": [{"class_name": names[int(i)], "score": float(probs[int(i)])} for i in top],
        "probs": [float(p) for p in probs],
        "health_score": compute_health(probs, names[best]),
    }


# ==================================================
# 6. TREATMENT KNOWLEDGE BASE + 30-DAY SCHEDULE
# ==================================================
HEALTHY_TIPS = [
    "Water at the base in the morning; avoid wetting the leaves.",
    "Give 6-8 hours of light and keep good air circulation between plants.",
    "Check leaves (top and underside) once a week for early spots or pests.",
    "Remove fallen leaves and debris from the soil surface.",
    "Feed with balanced fertilizer as per the plant's season; avoid over-fertilizing.",
]

GENERIC_NOTE = "Use any fungicide/pesticide only as per label and local agriculture guidelines."

RULES = [
    (("mosaic", "virus", "curl"), {
        "category": "Viral", "severity": "High",
        "summary": "A viral infection spread by insects (aphids, whiteflies), tools and hands. It cannot be cured, only controlled.",
        "treatment": [
            "Isolate the plant immediately from other plants.",
            "Remove and destroy (do not compost) heavily infected leaves/plants.",
            "Control vectors: spray neem oil or insecticidal soap on aphids/whiteflies.",
            "Disinfect tools and wash hands after handling the plant.",
            "Keep the plant well-watered and fed so new growth stays strong.",
        ],
        "prevention": ["Use virus-free seeds/seedlings.", "Use yellow sticky traps or nets.", "Control weeds that host insects."],
    }),
    (("mite",), {
        "category": "Pest", "severity": "Medium",
        "summary": "Tiny spider mites suck sap from leaves causing speckling, yellowing and webbing.",
        "treatment": [
            "Spray the plant with a strong jet of water, including leaf undersides.",
            "Apply neem oil or insecticidal soap every 5-7 days for 3 weeks.",
            "Remove badly damaged leaves.",
            "Raise humidity; mites thrive in hot, dry conditions.",
        ],
        "prevention": ["Inspect leaf undersides weekly.", "Avoid water stress and dust build-up."],
    }),
    (("greening", "haunglongbing"), {
        "category": "Bacterial (insect-spread)", "severity": "High",
        "summary": "Citrus greening is a serious bacterial disease spread by psyllids; there is no known cure.",
        "treatment": [
            "Report to your local agriculture office - it is regulated in many regions.",
            "Remove and destroy infected trees to protect others.",
            "Control psyllids with recommended insecticides/oils.",
            "Support the tree with balanced nutrition while awaiting expert advice.",
        ],
        "prevention": ["Buy certified disease-free plants.", "Monitor new flushes for psyllids."],
    }),
    (("powdery",), {
        "category": "Fungal", "severity": "Medium",
        "summary": "A white powdery fungal coating on leaves that reduces photosynthesis.",
        "treatment": [
            "Remove heavily coated leaves and dispose of them in the trash.",
            "Spray a baking-soda solution (1 tsp/litre + few drops of soap) or sulfur/potassium bicarbonate fungicide.",
            "Improve airflow by pruning crowded growth.",
            "Repeat every 7 days until new growth stays clean.",
        ],
        "prevention": ["Avoid overhead watering.", "Give plants full sun and spacing."],
    }),
    (("rust",), {
        "category": "Fungal", "severity": "Medium",
        "summary": "Orange/brown pustules on leaves caused by rust fungi.",
        "treatment": [
            "Pick off and destroy affected leaves.",
            "Apply a copper- or sulfur-based fungicide every 7-10 days.",
            "Water only at the base and keep foliage dry.",
            "Clear fallen leaves around the plant.",
        ],
        "prevention": ["Choose resistant varieties.", "Avoid dense planting."],
    }),
    (("scab",), {
        "category": "Fungal", "severity": "Medium",
        "summary": "Dark, scabby lesions on leaves and fruit caused by a fungus.",
        "treatment": [
            "Remove infected leaves and fruit; rake fallen leaves.",
            "Apply a protectant fungicide (captan/copper) as per label during wet weather.",
            "Prune to open up the canopy for drying.",
        ],
        "prevention": ["Clean up leaf litter every autumn.", "Plant scab-resistant varieties."],
    }),
    (("rot", "esca"), {
        "category": "Fungal", "severity": "High",
        "summary": "Rot-causing fungi damage leaves, fruit and wood, spreading fast in warm, humid weather.",
        "treatment": [
            "Prune out infected wood/leaves and mummified fruit; destroy them.",
            "Apply copper or a labelled fungicide on a 10-14 day cycle.",
            "Improve drainage and canopy airflow.",
            "Sterilize pruning tools between cuts.",
        ],
        "prevention": ["Remove old fruit and debris.", "Avoid wounding the plant."],
    }),
    (("scorch",), {
        "category": "Environmental / Fungal", "severity": "Medium",
        "summary": "Brown, dry leaf edges due to water stress, heat, salt build-up or infection.",
        "treatment": [
            "Water deeply and regularly; avoid drought stress.",
            "Mulch to keep roots cool and moist.",
            "Trim scorched leaves and flush soil if salts have built up.",
            "Provide afternoon shade during heat waves.",
        ],
        "prevention": ["Avoid over-fertilizing.", "Maintain consistent watering."],
    }),
    (("mold",), {
        "category": "Fungal", "severity": "Medium",
        "summary": "Leaf mold thrives in high humidity: yellow patches on top, mold underneath.",
        "treatment": [
            "Remove affected leaves; lower humidity and increase ventilation.",
            "Space plants and prune lower foliage.",
            "Apply a copper or labelled fungicide every 7-10 days.",
            "Water at soil level in the morning.",
        ],
        "prevention": ["Keep humidity below ~85%.", "Stake plants to improve airflow."],
    }),
    (("blight",), {
        "category": "Fungal / Oomycete", "severity": "High",
        "summary": "Blight causes fast-spreading brown lesions and leaf death, especially in cool, wet weather.",
        "treatment": [
            "Remove and destroy infected leaves immediately; do not compost them.",
            "Apply copper-based or labelled fungicide every 7 days during wet weather.",
            "Stop overhead watering and keep foliage dry.",
            "Stake/prune for airflow and mulch to prevent soil splash.",
        ],
        "prevention": ["Rotate crops yearly.", "Use resistant varieties and certified seed."],
    }),
    (("spot", "target", "septoria", "cercospora"), {
        "category": "Fungal / Bacterial", "severity": "Medium",
        "summary": "Circular spots on leaves that can merge and cause leaf drop.",
        "treatment": [
            "Remove spotted leaves and keep them away from the plant.",
            "Apply copper-based spray (works for bacterial and fungal spots) every 7-10 days.",
            "Avoid working with the plant when leaves are wet.",
            "Mulch to prevent soil splash.",
        ],
        "prevention": ["Rotate crops.", "Water at the base; keep leaves dry."],
    }),
]

DEFAULT_INFO = {
    "category": "Unspecified", "severity": "Medium",
    "summary": "A plant disease or stress was detected. Treat conservatively and confirm with an expert.",
    "treatment": [
        "Isolate the plant and remove visibly affected leaves.",
        "Water at the base and improve airflow and light.",
        "Consult a local agriculture expert with a clear photo for a confirmed diagnosis.",
    ],
    "prevention": ["Inspect plants weekly.", "Keep tools and pots clean."],
}


def clean(text):
    return text.replace("_", " ").replace("(", "").replace(")", "").strip().title()


def get_condition(class_name):
    plant, _, cond = class_name.partition("___")
    base = {"plant": clean(plant), "condition": clean(cond or "Unknown")}
    low = cond.lower()
    if "healthy" in low:
        return {**base, "condition": "Healthy", "is_healthy": True, "category": "None",
                "severity": "None", "summary": "No disease detected.",
                "treatment": [], "prevention": HEALTHY_TIPS}
    data = dict(DEFAULT_INFO)
    for keys, info in RULES:
        if any(k in low for k in keys):
            data = dict(info)
            break
    data["treatment"] = data["treatment"] + [GENERIC_NOTE]
    return {**base, **data, "is_healthy": False}


def severity_pill(sev):
    return pill(f"Severity: {sev}", sev.lower() if sev.lower() in ("high", "medium") else "none")


def build_schedule(info):
    treat_days = sorted({2, 6, 10, 16, 23})
    review_days = {7, 14, 21, 28}
    steps = info["treatment"]
    schedule = []
    for d in range(1, 31):
        phase = ("Phase 1 - Intensive treatment" if d <= 7 else
                 "Phase 2 - Control" if d <= 14 else
                 "Phase 3 - Recovery" if d <= 21 else "Phase 4 - Prevention")
        if d == 1:
            tasks = ["Isolate the plant and remove all visibly infected leaves.",
                     "Take a clear baseline photo of the affected leaves.",
                     "Clean and disinfect your tools."]
        elif d in treat_days:
            step = steps[treat_days.index(d) % max(1, len(steps) - 1)]
            tasks = [f"Treatment round: {step}", "Check leaf undersides for new spots."]
        elif d in review_days:
            tasks = ["Weekly review: compare today's leaves with your day-1 photo.",
                     "Remove any newly infected leaves."]
        elif d == 30:
            tasks = ["Final assessment: upload a photo and record the final health score.",
                     "Decide whether to continue prevention care."]
        elif d <= 14:
            tasks = ["Inspect all leaves and remove new infected parts.",
                     "Water at the base only; keep leaves dry."]
        elif d <= 21:
            tasks = ["Check for new growth and note any new symptoms.",
                     "Maintain light, airflow and balanced watering."]
        else:
            tasks = ["Quick leaf inspection.",
                     "Keep soil moist but not soggy; no overhead watering."]
        schedule.append({"day": d, "phase": phase, "tasks": tasks})
    return schedule


# ==================================================
# 7. PAGE: DETECT DISEASE
# ==================================================
def start_plan(scan):
    info = get_condition(scan["class_name"])
    nick = st.session_state.get("plan_nick", "").strip() or f"My {info['plant']}"
    res = get_db().plans.insert_one({
        "user_id": uid(), "nickname": nick, "class_name": scan["class_name"],
        "plant": info["plant"], "condition": info["condition"],
        "baseline_health": scan["health_score"], "scan_id": scan["scan_id"],
        "start_date": now(), "status": "active", "done": [],
    })
    st.session_state["active_plan"] = str(res.inserted_id)
    st.session_state["nav"] = PAGE_PLAN


def show_result(scan):
    info = get_condition(scan["class_name"])
    conf = scan["confidence"]
    threshold = st.session_state.get("low_conf", DEFAULT_LOW_CONFIDENCE)
    st.divider()
    st.subheader("Prediction results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plant", info["plant"])
    c2.metric("Condition", info["condition"])
    c3.metric("Model confidence", f"{conf * 100:.1f}%")
    c4.metric("Leaf health score", f"{scan['health_score']:.0f}/100")

    if conf < threshold:
        st.warning("The model is not confident about this result. Retake the photo in good light "
                   "with one leaf clearly visible against a plain background.")

    if info["is_healthy"]:
        st.success("Your plant looks healthy. No disease detected.")
        st.markdown("#### Keep it healthy")
        for tip in HEALTHY_TIPS:
            st.markdown(f"- {tip}")
    else:
        st.error(f"Disease detected: **{info['condition']}**")
        st.markdown(pill(info["category"]) + severity_pill(info["severity"]), unsafe_allow_html=True)
        st.write("")
        st.write(info["summary"])
        t_col, p_col = st.columns(2, gap="large")
        with t_col:
            st.markdown("#### How to treat it")
            for i, step in enumerate(info["treatment"], 1):
                st.markdown(f"{i}. {step}")
        with p_col:
            st.markdown("#### Prevention")
            for tip in info["prevention"]:
                st.markdown(f"- {tip}")
        st.divider()
        st.markdown("#### Track your recovery")
        st.caption("Start a 30-day plan with daily tasks and a check-in every 24 hours to see how much the plant improves.")
        st.text_input("Plant nickname (optional)", key="plan_nick", placeholder=f"My {info['plant']}")
        st.button("Start 30-day care plan", type="primary", on_click=start_plan, args=(scan,))

    with st.expander("Top 3 predictions"):
        for t in scan["top3"]:
            st.write(f"**{clean(t['class_name'].replace('___', ' - '))}** - {t['score'] * 100:.1f}%")
            st.progress(min(1.0, t["score"]))
    st.caption("This is an AI prediction, not a confirmed diagnosis. Consult an agriculture expert for serious cases.")


def render_detect():
    hero("Detect plant disease", "Upload a leaf photo or capture one with your camera.")
    if not model_files_exist():
        st.error("Model files not found. Place plant_disease_cnn.keras and class_names.json in the saved_model folder.")
        return

    with st.expander("Tips for an accurate result", expanded=False):
        st.markdown(
            "- Photograph **one leaf** filling most of the frame.\n"
            "- Use natural daylight and avoid harsh shadows.\n"
            "- Keep the leaf in focus against a plain background.\n"
            "- Capture the most affected leaf, not a healthy one."
        )

    left, right = st.columns(2, gap="large")
    with left:
        up = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"], key="leaf_uploader")
    with right:
        cam = st.camera_input("Capture a leaf image", key="leaf_camera")
    selected = up if up is not None else cam
    if selected is None:
        st.info("Upload or capture a leaf image to get started.")
        return

    try:
        image = ImageOps.exif_transpose(Image.open(selected)).convert("RGB")
    except Exception as err:
        st.error(f"Could not read this image: {err}")
        return
    st.image(image, caption="Selected image", width=320)

    if st.button("Analyze leaf", type="primary"):
        with st.spinner("Analyzing the leaf..."):
            try:
                res = predict(image)
            except Exception as err:
                st.error(f"Prediction failed: {err}")
                return
        info = get_condition(res["class_name"])
        doc = {
            "user_id": uid(), "class_name": res["class_name"], "plant": info["plant"],
            "condition": info["condition"], "is_healthy": info["is_healthy"],
            "category": info["category"], "severity": info["severity"],
            "confidence": res["confidence"], "health_score": res["health_score"],
            "created_at": now(),
        }
        scan_id = str(get_db().scans.insert_one(doc).inserted_id)
        st.session_state["scan"] = {**res, "scan_id": scan_id, "file_id": selected.file_id}

    scan = st.session_state.get("scan")
    if scan and scan["file_id"] == selected.file_id:
        show_result(scan)


# ==================================================
# 8. PAGE: 30-DAY CARE PLAN
# ==================================================
def toggle_task(plan_id, token, key):
    op = "$addToSet" if st.session_state[key] else "$pull"
    get_db().plans.update_one({"_id": ObjectId(plan_id)}, {op: {"done": token}})


def status_text(delta):
    if delta >= 5:
        return "Improving"
    if delta <= -5:
        return "Getting worse - consider consulting an expert"
    return "Stable"


def render_care_plan():
    db = get_db()
    hero("30-day care plan", "Follow the daily tasks and check in every 24 hours to measure recovery.")
    plans = list(db.plans.find({"user_id": uid(), "status": "active"}).sort("start_date", -1))
    if not plans:
        st.info("No active care plan. Detect a disease first, then start a plan from the result page.")
        st.button("Go to Detect Disease", on_click=goto, args=(PAGE_DETECT,))
        return

    labels = {str(p["_id"]): f"{p['nickname']} - {p['condition']} (started {p['start_date']:%d %b})" for p in plans}
    ids = list(labels)
    default = ids.index(st.session_state["active_plan"]) if st.session_state.get("active_plan") in ids else 0
    pid = st.selectbox("Select plan", ids, index=default, format_func=labels.get)
    st.session_state["active_plan"] = pid
    plan = next(p for p in plans if str(p["_id"]) == pid)
    info = get_condition(plan["class_name"])
    schedule = build_schedule(info)

    elapsed = (now().date() - plan["start_date"].date()).days + 1
    day = max(1, min(elapsed, 30))
    checkins = list(db.checkins.find({"plan_id": pid}).sort("created_at", 1))
    last = checkins[-1] if checkins else None
    current_health = last["health_score"] if last else plan["baseline_health"]
    total_tasks = sum(len(s["tasks"]) for s in schedule)

    st.subheader(f"{plan['nickname']} - {plan['plant']} ({plan['condition']})")
    st.progress(day / 30, text=f"Day {day} of 30")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Day", f"{day}/30")
    m2.metric("Check-ins", len(checkins))
    m3.metric("Tasks done", f"{len(plan.get('done', [])) / total_tasks * 100:.0f}%")
    m4.metric("Health score", f"{current_health:.0f}/100",
              delta=f"{current_health - plan['baseline_health']:+.0f} vs start")

    if elapsed > 30:
        st.success("The 30-day period is over. Review your progress below and mark the plan as completed.")

    tab_today, tab_checkin, tab_progress, tab_sched = st.tabs(
        ["Today's tasks", "Daily check-in", "Improvement", "Full schedule"])

    with tab_today:
        st.markdown(f"#### {schedule[day - 1]['phase']}")
        for i, task in enumerate(schedule[day - 1]["tasks"]):
            token, key = f"d{day}t{i}", f"{pid}-d{day}t{i}"
            st.checkbox(task, value=token in plan.get("done", []), key=key,
                        on_change=toggle_task, args=(pid, token, key))

    with tab_checkin:
        next_time = last["created_at"] + timedelta(hours=24) if last else None
        if next_time and now() < next_time:
            remaining = next_time - now()
            st.info(f"Next check-in available in {remaining.seconds // 3600}h "
                    f"{(remaining.seconds % 3600) // 60}m (at {fmt(next_time)}).")
        else:
            with st.form("checkin_form", clear_on_submit=True):
                photo = st.file_uploader("Leaf photo (optional but recommended)", type=["jpg", "jpeg", "png"])
                affected = st.slider("How much of the plant/leaf is still affected? (%)", 0, 100, 50, 5)
                notes = st.text_area("Notes (optional)", placeholder="e.g. new spots stopped appearing")
                if st.form_submit_button("Save check-in"):
                    model_h, photo_class = None, None
                    if photo is not None:
                        try:
                            img = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
                            res = predict(img)
                            model_h = compute_health(res["probs"], plan["class_name"])
                            photo_class = res["class_name"]
                        except Exception as err:
                            st.error(f"Could not analyze photo: {err}")
                            st.stop()
                    self_h = 100 - affected
                    health = round(0.6 * model_h + 0.4 * self_h, 1) if model_h is not None else float(self_h)
                    prev = last["health_score"] if last else plan["baseline_health"]
                    db.checkins.insert_one({
                        "plan_id": pid, "user_id": uid(), "day": day, "health_score": health,
                        "model_health": model_h, "self_health": self_h, "affected_pct": affected,
                        "delta": round(health - prev, 1), "photo_class": photo_class,
                        "notes": notes.strip(), "created_at": now(),
                    })
                    st.rerun()

    with tab_progress:
        if not checkins:
            st.info("Save your first check-in to see your improvement chart.")
        else:
            delta = last["delta"]
            c1, c2 = st.columns(2)
            c1.metric("Change since last check-in (24h)", f"{delta:+.1f} pts", delta=f"{delta:+.1f}")
            c2.metric("Total change since start", f"{last['health_score'] - plan['baseline_health']:+.1f} pts")
            st.write(f"**Status:** {status_text(delta)}")
            df = pd.DataFrame({
                "Check-in": [f"#{i + 1} (Day {c['day']})" for i, c in enumerate(checkins)],
                "Health score": [c["health_score"] for c in checkins],
            }).set_index("Check-in")
            st.line_chart(df, color=C_PRIMARY)
            with st.expander("Check-in history"):
                show_df(pd.DataFrame([{
                    "Date": fmt(c["created_at"]), "Day": c["day"], "Health": c["health_score"],
                    "Change": c["delta"], "Affected %": c["affected_pct"], "Notes": c["notes"],
                } for c in reversed(checkins)]))

    with tab_sched:
        show_df(pd.DataFrame([{
            "Day": s["day"], "Phase": s["phase"], "Tasks": " • ".join(s["tasks"]),
        } for s in schedule]))

    st.divider()
    a, b = st.columns(2)
    if a.button("Mark plan as completed"):
        db.plans.update_one({"_id": plan["_id"]}, {"$set": {"status": "completed", "end_date": now()}})
        st.rerun()
    if b.button("Stop this plan"):
        db.plans.update_one({"_id": plan["_id"]}, {"$set": {"status": "stopped", "end_date": now()}})
        st.rerun()


# ==================================================
# 9. PAGE: DASHBOARD
# ==================================================
def render_dashboard():
    db = get_db()
    name = st.session_state["user"]["name"]
    hero(f"Welcome back, {name.split()[0].title()}", "Your plant health overview at a glance.")
    scans = list(db.scans.find({"user_id": uid()}).sort("created_at", -1))
    plans = list(db.plans.find({"user_id": uid()}))
    active = [p for p in plans if p["status"] == "active"]

    healthy = sum(1 for s in scans if s["is_healthy"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total scans", len(scans))
    c2.metric("Healthy plants", healthy)
    c3.metric("Diseases found", len(scans) - healthy)
    c4.metric("Active care plans", len(active))

    st.write("")
    q1, q2, q3 = st.columns(3)
    q1.button("Scan a new leaf", on_click=goto, args=(PAGE_DETECT,), key="q_detect")
    q2.button("Open my care plan", on_click=goto, args=(PAGE_PLAN,), key="q_plan")
    q3.button("Browse disease library", on_click=goto, args=(PAGE_LIBRARY,), key="q_lib")

    if not scans:
        st.info("No scans yet. Use 'Detect Disease' to analyze your first leaf.")
        return

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Disease distribution")
        diseased = [f"{s['plant']} - {s['condition']}" for s in scans if not s["is_healthy"]]
        if diseased:
            counts = pd.Series(diseased).value_counts().rename("Scans")
            try:
                st.bar_chart(counts, horizontal=True, color=C_PRIMARY)
            except TypeError:
                st.bar_chart(counts, color=C_PRIMARY)
        else:
            st.success("No diseases detected so far ")
    with right:
        st.markdown("#### Active plan progress")
        if active:
            for p in active:
                cks = list(db.checkins.find({"plan_id": str(p["_id"])}).sort("created_at", 1))
                day = max(1, min((now().date() - p["start_date"].date()).days + 1, 30))
                health = cks[-1]["health_score"] if cks else p["baseline_health"]
                st.markdown(
                    f'<div class="panel"><b>{p["nickname"]}</b> ({p["condition"]})<br>'
                    f'Day {day}/30 &nbsp;|&nbsp; Health {health:.0f}/100</div>',
                    unsafe_allow_html=True)
                st.progress(day / 30)
        else:
            st.caption("No active plans.")

    st.markdown("#### Scans over time")
    by_day = pd.Series([s["created_at"].date() for s in scans]).value_counts().sort_index().rename("Scans")
    st.line_chart(by_day, color=C_PRIMARY)

    st.markdown("#### Recent scans")
    show_df(pd.DataFrame([{
        "Date": fmt(s["created_at"]), "Plant": s["plant"], "Condition": s["condition"],
        "Confidence %": round(s["confidence"] * 100, 1),
    } for s in scans[:10]]))


# ==================================================
# 10. PAGE: HISTORY & REPORTS
# ==================================================
def render_history():
    db = get_db()
    hero("History & reports", "Review every scan and care plan, and download your data.")
    tab_scans, tab_plans = st.tabs(["Scan history", "Care plans"])

    with tab_scans:
        scans = list(db.scans.find({"user_id": uid()}).sort("created_at", -1))
        if not scans:
            st.info("No scans yet.")
        else:
            df = pd.DataFrame([{
                "Date": s["created_at"], "Plant": s["plant"], "Condition": s["condition"],
                "Category": s.get("category", "-"), "Severity": s.get("severity", "-"),
                "Confidence %": round(s["confidence"] * 100, 1),
                "Health score": round(s.get("health_score", 0), 1),
                "Healthy": s["is_healthy"],
            } for s in scans])

            f1, f2, f3 = st.columns(3)
            plant_f = f1.multiselect("Plant", sorted(df["Plant"].unique()))
            cond_f = f2.selectbox("Result", ["All", "Healthy only", "Diseased only"])
            search = f3.text_input("Search condition", placeholder="e.g. rust")
            if plant_f:
                df = df[df["Plant"].isin(plant_f)]
            if cond_f == "Healthy only":
                df = df[df["Healthy"]]
            elif cond_f == "Diseased only":
                df = df[~df["Healthy"]]
            if search.strip():
                df = df[df["Condition"].str.contains(search.strip(), case=False, na=False)]

            st.caption(f"{len(df)} scan(s) shown")
            view = df.copy()
            view["Date"] = view["Date"].apply(fmt)
            show_df(view.drop(columns=["Healthy"]))
            st.download_button("Download scans as CSV", view.to_csv(index=False).encode("utf-8"),
                               file_name="plantcare_scans.csv", mime="text/csv")

    with tab_plans:
        plans = list(db.plans.find({"user_id": uid()}).sort("start_date", -1))
        if not plans:
            st.info("No care plans yet.")
        else:
            rows = []
            for p in plans:
                cks = list(db.checkins.find({"plan_id": str(p["_id"])}).sort("created_at", 1))
                final = cks[-1]["health_score"] if cks else p["baseline_health"]
                rows.append({
                    "Plan": p["nickname"], "Plant": p["plant"], "Condition": p["condition"],
                    "Status": p["status"].title(), "Started": fmt(p["start_date"]),
                    "Check-ins": len(cks), "Start health": round(p["baseline_health"], 1),
                    "Latest health": round(final, 1),
                    "Change": round(final - p["baseline_health"], 1),
                })
            pdf = pd.DataFrame(rows)
            show_df(pdf)
            st.download_button("Download plans as CSV", pdf.to_csv(index=False).encode("utf-8"),
                               file_name="plantcare_plans.csv", mime="text/csv")


# ==================================================
# 11. PAGE: DISEASE LIBRARY
# ==================================================
def render_library():
    hero("Disease library", "Browse every plant and condition the model can recognise, with treatment advice.")
    if not model_files_exist():
        st.error("class_names.json not found in the saved_model folder.")
        return
    names = load_class_names()
    plants = sorted({clean(plant_key(n)) for n in names})

    c1, c2 = st.columns([1, 2])
    plant_sel = c1.selectbox("Plant", ["All plants"] + plants)
    query = c2.text_input("Search condition", placeholder="e.g. blight, rust, mildew")

    shown = 0
    for n in sorted(names):
        info = get_condition(n)
        if plant_sel != "All plants" and info["plant"] != plant_sel:
            continue
        if query.strip() and query.strip().lower() not in info["condition"].lower():
            continue
        shown += 1
        with st.expander(f"{info['plant']} - {info['condition']}"):
            st.markdown(pill(info["category"]) + severity_pill(info["severity"]), unsafe_allow_html=True)
            st.write("")
            st.write(info["summary"])
            if not info["is_healthy"]:
                t, p = st.columns(2, gap="large")
                with t:
                    st.markdown("**Treatment**")
                    for i, step in enumerate(info["treatment"], 1):
                        st.markdown(f"{i}. {step}")
                with p:
                    st.markdown("**Prevention**")
                    for tip in info["prevention"]:
                        st.markdown(f"- {tip}")
            else:
                st.markdown("**Keep it healthy**")
                for tip in HEALTHY_TIPS:
                    st.markdown(f"- {tip}")
    st.caption(f"{shown} condition(s) shown out of {len(names)}.")
    if shown == 0:
        st.info("No conditions match your filters.")


# ==================================================
# 12. PAGE: PLANT CARE GUIDE
# ==================================================
def render_tips():
    hero("Plant care guide", "Simple habits that prevent most leaf diseases.")
    t1, t2, t3, t4 = st.tabs(["Watering", "Light & soil", "Pests & hygiene", "Safe spraying"])
    with t1:
        st.markdown("""
- **Water the soil, not the leaves.** Wet foliage is the most common trigger for fungal disease.
- **Water in the morning** so leaves and soil surface dry before night.
- **Water deeply, less often.** Shallow daily watering encourages weak roots.
- **Check before watering:** push a finger 2-3 cm into the soil; water only if it feels dry.
- **Mulch** to reduce soil splash, which carries spores onto lower leaves.
""")
    with t2:
        st.markdown("""
- **Light:** most vegetables and fruit plants need 6-8 hours of direct sun.
- **Spacing:** crowded plants trap humidity. Follow the spacing on the seed packet.
- **Soil:** well-drained, compost-rich soil lowers the risk of root rot.
- **Rotate crops** each season so soil-borne diseases do not build up.
- **Feed in moderation.** Too much nitrogen gives soft, disease-prone growth.
""")
    with t3:
        st.markdown("""
- **Inspect weekly**, including the underside of leaves.
- **Remove sick leaves early** and bin them; do not compost infected material.
- **Disinfect tools** with rubbing alcohol or diluted bleach between plants.
- **Use sticky traps or nets** against aphids and whiteflies, which spread viruses.
- **Keep the area tidy:** fallen leaves and old fruit shelter pests and spores.
""")
    with t4:
        st.markdown("""
- **Read the label** and follow dose, interval and safety period before harvest.
- **Spray in the cool part of the day**, when it is not windy.
- **Wear gloves, a mask and eye protection.**
- **Cover leaf undersides** as well as tops.
- **Rotate products** so pests and fungi do not become resistant.
- **Ask your local agriculture office** for approved products in your region.
""")

    st.markdown("#### Quick answers")
    with st.expander("How often should I scan my plant?"):
        st.write("Once a week is enough for routine checks. Scan sooner if you notice spots, curling, "
                 "yellowing or a powdery coating.")
    with st.expander("Why does the app say the result is low confidence?"):
        st.write("The model is unsure, usually because of poor light, blur, several leaves in one photo "
                 " or a plant it was not trained on. Retake the photo and try again.")
    with st.expander("What does the health score mean?"):
        st.write("It estimates how close the leaf is to a healthy leaf of the same plant, from 0 (very "
                 "diseased) to 100 (healthy). In care-plan check-ins it also uses your own estimate of how much "
                 "of the plant is still affected.")
    with st.expander("Is the diagnosis guaranteed?"):
        st.write("No. It is an AI prediction. For valuable crops or serious outbreaks, confirm with an "
                 "agriculture expert.")


# ==================================================
# 13. PAGE: PROFILE & SETTINGS
# ==================================================
def render_profile():
    db = get_db()
    hero("Profile & settings", "Manage your account, preferences and data.")
    user_doc = db.users.find_one({"_id": ObjectId(uid())})
    if not user_doc:
        st.error("Account not found.")
        return

    n_scans = db.scans.count_documents({"user_id": uid()})
    n_plans = db.plans.count_documents({"user_id": uid()})
    c1, c2, c3 = st.columns(3)
    c1.metric("Member since", f"{user_doc['created_at']:%d %b %Y}")
    c2.metric("Scans", n_scans)
    c3.metric("Care plans", n_plans)

    tab_acc, tab_pwd, tab_pref, tab_data = st.tabs(
        ["Account", "Password", "Preferences", "Data"])

    with tab_acc:
        with st.form("profile_form"):
            new_name = st.text_input("Full name", value=user_doc["name"])
            st.text_input("Email (cannot be changed)", value=user_doc["email"], disabled=True)
            if st.form_submit_button("Save changes"):
                if len(new_name.strip()) < 2:
                    st.error("Please enter your name.")
                else:
                    db.users.update_one({"_id": user_doc["_id"]}, {"$set": {"name": new_name.strip()}})
                    st.session_state["user"]["name"] = new_name.strip()
                    st.success("Profile updated.")

    with tab_pwd:
        with st.form("password_form", clear_on_submit=True):
            old = st.text_input("Current password", type="password")
            new = st.text_input("New password (min 8 characters)", type="password")
            conf = st.text_input("Confirm new password", type="password")
            if st.form_submit_button("Update password"):
                if not bcrypt.checkpw(old.encode(), user_doc["password_hash"]):
                    st.error("Current password is incorrect.")
                elif len(new) < 8:
                    st.error("New password must be at least 8 characters.")
                elif new != conf:
                    st.error("New passwords do not match.")
                else:
                    db.users.update_one(
                        {"_id": user_doc["_id"]},
                        {"$set": {"password_hash": bcrypt.hashpw(new.encode(), bcrypt.gensalt())}})
                    st.success("Password updated.")

    with tab_pref:
        st.slider("Warn me when model confidence is below (%)", 30, 90,
                  int(st.session_state.get("low_conf", DEFAULT_LOW_CONFIDENCE) * 100), 5, key="low_conf_pct")
        st.session_state["low_conf"] = st.session_state["low_conf_pct"] / 100
        st.caption("Applies to the Detect Disease page for this session.")

    with tab_data:
        st.markdown("**Clear scan history**")
        st.caption("Permanently deletes all of your saved scans. Care plans are kept.")
        if st.checkbox("I understand this cannot be undone", key="confirm_clear"):
            if st.button("Delete all scans"):
                db.scans.delete_many({"user_id": uid()})
                st.session_state.pop("scan", None)
                st.success("Scan history cleared.")
        st.divider()
        st.markdown("**Delete account**")
        st.caption("Permanently deletes your account, scans, care plans and check-ins.")
        if st.checkbox("I want to permanently delete my account", key="confirm_delete"):
            if st.button("Delete my account"):
                for coll in (db.scans, db.plans, db.checkins):
                    coll.delete_many({"user_id": uid()})
                db.users.delete_one({"_id": user_doc["_id"]})
                logout()
                st.rerun()


# ==================================================
# 14. MAIN
# ==================================================
inject_css()

try:
    get_db()
except ValueError as err:
    st.error(str(err))
    st.stop()
except Exception as err:
    st.error(f"Could not connect to MongoDB: {err}")
    st.info("Check your credentials and that your IP is allowed in Atlas -> Network Access.")
    st.stop()

if "user" not in st.session_state or st.query_params.get("reset_token"):
    render_auth()
    st.stop()

if st.session_state.get("nav") not in PAGES:
    st.session_state["nav"] = PAGE_DASH

with st.sidebar:
    st.markdown(
        f'<div class="brand-row">{logo(44)}<div><div class="brand">PlantCare AI</div>'
        '<div class="brand-sub">Plant disease detection</div></div></div>',
        unsafe_allow_html=True)
    st.caption(f"Signed in as **{st.session_state['user']['name']}**")
    st.write("")
    st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
    st.divider()
    if st.button("Log out"):
        logout()
        st.rerun()

page = st.session_state["nav"]
if page == PAGE_DASH:
    render_dashboard()
elif page == PAGE_DETECT:
    render_detect()
elif page == PAGE_PLAN:
    render_care_plan()
elif page == PAGE_HISTORY:
    render_history()
elif page == PAGE_LIBRARY:
    render_library()
elif page == PAGE_TIPS:
    render_tips()
elif page == PAGE_PROFILE:
    render_profile()

st.divider()
st.markdown('<div class="footer">PlantCare AI | Deep Learning (CNN) | '
            'Predictions are not a confirmed diagnosis.</div>', unsafe_allow_html=True)
