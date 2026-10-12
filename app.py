import os
import re
import json
import base64
from datetime import datetime, timedelta, timezone

import bcrypt
import certifi
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from bson import ObjectId
from PIL import Image, ImageOps
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# ==================================================
# 1. PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "plant_disease_cnn.keras")
CLASS_PATH = os.path.join(BASE_DIR, "saved_model", "class_names.json")
BACKGROUND_PATH = os.path.join(BASE_DIR, "assets", "plant_app.png")
IMG_SIZE = 128
LOW_CONFIDENCE = 0.60

PAGE_DASH = " Dashboard"
PAGE_DETECT = " Detect Disease"
PAGE_PLAN = " 30-Day Care Plan"
PAGES = [PAGE_DASH, PAGE_DETECT, PAGE_PLAN]


# ==================================================
# 2. STYLING
# ==================================================
@st.cache_data
def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def inject_css():
    if os.path.exists(BACKGROUND_PATH):
        bg = (
            "background-image: linear-gradient(rgba(8,25,17,.78), rgba(8,25,17,.88)),"
            f'url("data:image/png;base64,{_encode_image(BACKGROUND_PATH)}");'
        )
    else:
        bg = "background: linear-gradient(160deg,#081911,#12372a);"

    st.markdown(
        f"""
        <style>
        .stApp {{ {bg} background-size: cover; background-position: center;
                  background-attachment: fixed; }}
        .block-container {{ max-width: 1150px; padding-top: 2rem; padding-bottom: 3rem; }}
        h1, h2, h3, h4, p, label, .stMarkdown {{ color: #F4F8F2; }}
        div[data-testid="stFileUploader"], div[data-testid="stCameraInput"] {{
            background-color: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.2);
            border-radius: 14px; padding: 14px; }}
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
            background-color: #B7E4C7; color: #153D2B; border: none; border-radius: 10px;
            min-height: 45px; font-weight: 600; width: 100%; }}
        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{
            background-color: #95D5B2; color: #153D2B; }}
        div[data-testid="stMetric"] {{ background-color: rgba(255,255,255,.10);
            padding: 16px; border-radius: 12px; }}
        section[data-testid="stSidebar"] {{ background-color: rgba(8,25,17,.92); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# 3. DATABASE (MongoDB)
# ==================================================
@st.cache_resource(show_spinner="Connecting to database...")
def get_db():
    client = MongoClient(
        st.secrets["MONGO_URI"],
        serverSelectionTimeoutMS=8000,
        tlsCAFile=certifi.where(),
        tz_aware=True,
    )
    client.admin.command("ping")
    db = client[st.secrets.get("DB_NAME", "plant_disease_app")]
    db.users.create_index("email", unique=True)
    db.scans.create_index([("user_id", 1), ("created_at", -1)])
    db.plans.create_index([("user_id", 1), ("status", 1)])
    db.checkins.create_index([("plan_id", 1), ("created_at", 1)])
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


def render_auth():
    st.title("🌿 Plant Disease Detection")
    st.markdown("##### AI-powered leaf analysis, treatment guidance and 30-day recovery tracking")
    st.divider()
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])
        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    ok, msg = login(email, pwd)
                    if ok:
                        st.rerun()
                    st.error(msg)
        with tab_signup:
            with st.form("signup_form"):
                name = st.text_input("Full name")
                email = st.text_input("Email", key="su_email")
                pwd = st.text_input("Password (min 8 characters)", type="password", key="su_pwd")
                conf = st.text_input("Confirm password", type="password")
                if st.form_submit_button("Create account"):
                    ok, msg = signup(name, email, pwd, conf)
                    (st.success if ok else st.error)(msg)


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
    st.divider()
    st.subheader("🔍 Prediction Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Plant", info["plant"])
    c2.metric("Condition", info["condition"])
    c3.metric("Model confidence", f"{conf * 100:.1f}%")

    if conf < LOW_CONFIDENCE:
        st.warning("The model is not confident about this result. Retake the photo in good light with one leaf clearly visible.")

    if info["is_healthy"]:
        st.success(" All good - your plant looks healthy. No disease detected.")
        st.markdown("####  Keep it healthy")
        for tip in HEALTHY_TIPS:
            st.markdown(f"- {tip}")
    else:
        st.error(f"⚠️ Disease detected: **{info['condition']}** · {info['category']} · Severity: {info['severity']}")
        st.write(info["summary"])
        t_col, p_col = st.columns(2, gap="large")
        with t_col:
            st.markdown("####  How to treat it")
            for i, step in enumerate(info["treatment"], 1):
                st.markdown(f"{i}. {step}")
        with p_col:
            st.markdown("####  Prevention")
            for tip in info["prevention"]:
                st.markdown(f"- {tip}")
        st.divider()
        st.markdown("####  Track your recovery")
        st.caption("Start a 30-day plan with daily tasks and a check-in every 24 hours to see how much the plant improves.")
        st.text_input("Plant nickname (optional)", key="plan_nick", placeholder=f"My {info['plant']}")
        st.button("Start 30-day care plan", type="primary", on_click=start_plan, args=(scan,))

    with st.expander("Top 3 predictions"):
        for t in scan["top3"]:
            st.write(f"**{clean(t['class_name'].replace('___', ' - '))}** - {t['score'] * 100:.1f}%")
            st.progress(min(1.0, t["score"]))
    st.caption("This is an AI prediction, not a confirmed diagnosis. Consult an agriculture expert for serious cases.")


def render_detect():
    st.title("🔍 Detect Plant Disease")
    st.caption("Upload a leaf photo or capture one with your camera.")
    if not model_files_exist():
        st.error("Model files not found. Place plant_disease_cnn.keras and class_names.json in the saved_model folder.")
        return

    left, right = st.columns(2, gap="large")
    with left:
        up = st.file_uploader("📁 Upload a leaf image", type=["jpg", "jpeg", "png"], key="leaf_uploader")
    with right:
        cam = st.camera_input("📷 Capture a leaf image", key="leaf_camera")
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

    if st.button("Analyze Plant Leaf", type="primary"):
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
        return " Improving"
    if delta <= -5:
        return " Getting worse - consider consulting an expert"
    return "➖ Stable"


def render_care_plan():
    db = get_db()
    st.title(" 30-Day Care Plan")
    plans = list(db.plans.find({"user_id": uid(), "status": "active"}).sort("start_date", -1))
    if not plans:
        st.info("No active care plan. Detect a disease first, then start a plan from the result page.")
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

    st.subheader(f"{plan['nickname']} · {plan['plant']} · {plan['condition']}")
    st.progress(day / 30, text=f"Day {day} of 30")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Day", f"{day}/30")
    m2.metric("Check-ins", len(checkins))
    m3.metric("Tasks done", f"{len(plan.get('done', [])) / total_tasks * 100:.0f}%")
    m4.metric("Health score", f"{current_health:.0f}/100",
              delta=f"{current_health - plan['baseline_health']:+.0f} vs start")

    if elapsed > 30:
        st.success("🎉 The 30-day period is over. Review your progress below and mark the plan as completed.")

    # Today's tasks
    st.markdown(f"### Today's tasks - {schedule[day - 1]['phase']}")
    for i, task in enumerate(schedule[day - 1]["tasks"]):
        token, key = f"d{day}t{i}", f"{pid}-d{day}t{i}"
        st.checkbox(task, value=token in plan.get("done", []), key=key,
                    on_change=toggle_task, args=(pid, token, key))

    # 24-hour check-in
    st.markdown("### 📸 Daily check-in (every 24 hours)")
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

    # Improvement
    if checkins:
        st.markdown("###  Improvement")
        delta = last["delta"]
        c1, c2 = st.columns(2)
        c1.metric("Change since last check-in (24h)", f"{delta:+.1f} pts", delta=f"{delta:+.1f}")
        c2.metric("Total change since start", f"{last['health_score'] - plan['baseline_health']:+.1f} pts")
        st.write(f"**Status:** {status_text(delta)}")
        df = pd.DataFrame({
            "Check-in": [f"#{i + 1} (Day {c['day']})" for i, c in enumerate(checkins)],
            "Health score": [c["health_score"] for c in checkins],
        }).set_index("Check-in")
        st.line_chart(df)
        with st.expander("Check-in history"):
            st.dataframe(pd.DataFrame([{
                "Date": fmt(c["created_at"]), "Day": c["day"], "Health": c["health_score"],
                "Change": c["delta"], "Affected %": c["affected_pct"], "Notes": c["notes"],
            } for c in reversed(checkins)]), use_container_width=True, hide_index=True)

    with st.expander(" Full 30-day schedule"):
        st.dataframe(pd.DataFrame([{
            "Day": s["day"], "Phase": s["phase"], "Tasks": " • ".join(s["tasks"]),
        } for s in schedule]), use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    if a.button(" Mark plan as completed"):
        db.plans.update_one({"_id": plan["_id"]}, {"$set": {"status": "completed", "end_date": now()}})
        st.rerun()
    if b.button(" Stop this plan"):
        db.plans.update_one({"_id": plan["_id"]}, {"$set": {"status": "stopped", "end_date": now()}})
        st.rerun()


# ==================================================
# 9. PAGE: DASHBOARD
# ==================================================
def render_dashboard():
    db = get_db()
    name = st.session_state["user"]["name"]
    st.title(f" Dashboard - welcome, {name.split()[0]}")
    scans = list(db.scans.find({"user_id": uid()}).sort("created_at", -1))
    plans = list(db.plans.find({"user_id": uid()}))
    active = [p for p in plans if p["status"] == "active"]

    healthy = sum(1 for s in scans if s["is_healthy"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total scans", len(scans))
    c2.metric("Healthy plants", healthy)
    c3.metric("Diseases found", len(scans) - healthy)
    c4.metric("Active care plans", len(active))

    if not scans:
        st.info("No scans yet. Go to 'Detect Disease' to analyze your first leaf.")
        return

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Disease distribution")
        diseased = [f"{s['plant']} - {s['condition']}" for s in scans if not s["is_healthy"]]
        if diseased:
            st.bar_chart(pd.Series(diseased).value_counts())
        else:
            st.success("No diseases detected so far 🎉")
    with right:
        st.markdown("#### Active plan progress")
        if active:
            for p in active:
                cks = list(db.checkins.find({"plan_id": str(p["_id"])}).sort("created_at", 1))
                day = max(1, min((now().date() - p["start_date"].date()).days + 1, 30))
                health = cks[-1]["health_score"] if cks else p["baseline_health"]
                st.write(f"**{p['nickname']}** ({p['condition']}) - Day {day}/30 · Health {health:.0f}/100")
                st.progress(day / 30)
        else:
            st.caption("No active plans.")

    st.markdown("#### Recent scans")
    st.dataframe(pd.DataFrame([{
        "Date": fmt(s["created_at"]), "Plant": s["plant"], "Condition": s["condition"],
        "Confidence %": round(s["confidence"] * 100, 1),
    } for s in scans[:15]]), use_container_width=True, hide_index=True)


# ==================================================
# 10. MAIN
# ==================================================
inject_css()

try:
    get_db()
except KeyError:
    st.error("MONGO_URI is missing. Add it to .streamlit/secrets.toml.")
    st.stop()
except Exception as err:
    st.error(f"Could not connect to MongoDB: {err}")
    st.info("Check your credentials and that your IP is allowed in Atlas -> Network Access.")
    st.stop()

if "user" not in st.session_state:
    render_auth()
    st.stop()

with st.sidebar:
    st.markdown("## 🌿 PlantCare AI")
    st.caption(f"Signed in as **{st.session_state['user']['name']}**")
    st.radio("Navigate", PAGES, key="nav")
    st.divider()
    if st.button("Logout"):
        logout()
        st.rerun()

page = st.session_state["nav"]
if page == PAGE_DASH:
    render_dashboard()
elif page == PAGE_DETECT:
    render_detect()
elif page == PAGE_PLAN:
    render_care_plan()

st.divider()
st.caption("Plant Disease Detection | Deep Learning | CNN · AI predictions are not a confirmed diagnosis.")