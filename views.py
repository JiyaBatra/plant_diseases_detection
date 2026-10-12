from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from bson import ObjectId
from PIL import Image, ImageOps

from db import get_db
from predictor import LOW_CONFIDENCE, health_score, model_files_exist, predict
from treatments import HEALTHY_TIPS, build_schedule, clean, get_condition

PAGE_DASH = " Dashboard"
PAGE_DETECT = " Detect Disease"
PAGE_PLAN = " 30-Day Care Plan"
PAGES = [PAGE_DASH, PAGE_DETECT, PAGE_PLAN]


def _now():
    return datetime.now(timezone.utc)


def _fmt(dt):
    return dt.strftime("%d %b %Y, %H:%M UTC")


def _uid():
    return st.session_state["user"]["id"]


# ------------------------------------------------------------------ DETECT
def _start_plan(scan: dict):
    info = get_condition(scan["class_name"])
    nick = st.session_state.get("plan_nick", "").strip() or f"My {info['plant']}"
    res = get_db().plans.insert_one({
        "user_id": _uid(), "nickname": nick, "class_name": scan["class_name"],
        "plant": info["plant"], "condition": info["condition"],
        "baseline_health": scan["health_score"], "scan_id": scan["scan_id"],
        "start_date": _now(), "status": "active", "done": [],
    })
    st.session_state["active_plan"] = str(res.inserted_id)
    st.session_state["nav"] = PAGE_PLAN


def _show_result(scan: dict):
    info = get_condition(scan["class_name"])
    conf = scan["confidence"]
    st.divider()
    st.subheader(" Prediction Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Plant", info["plant"])
    c2.metric("Condition", info["condition"])
    c3.metric("Model confidence", f"{conf * 100:.1f}%")

    if conf < LOW_CONFIDENCE:
        st.warning("The model is not confident about this result. Retake the photo in good light with one leaf clearly visible.")

    if info["is_healthy"]:
        st.success("All good - your plant looks healthy. No disease detected.")
        st.markdown("####  Keep it healthy")
        for tip in HEALTHY_TIPS:
            st.markdown(f"- {tip}")
    else:
        st.error(f" Disease detected: **{info['condition']}** · {info['category']} · Severity: {info['severity']}")
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
        st.button("Start 30-day care plan", type="primary", on_click=_start_plan, args=(scan,))

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
            "user_id": _uid(), "class_name": res["class_name"], "plant": info["plant"],
            "condition": info["condition"], "is_healthy": info["is_healthy"],
            "confidence": res["confidence"], "health_score": res["health_score"],
            "created_at": _now(),
        }
        scan_id = str(get_db().scans.insert_one(doc).inserted_id)
        st.session_state["scan"] = {**res, "scan_id": scan_id, "file_id": selected.file_id}

    scan = st.session_state.get("scan")
    if scan and scan["file_id"] == selected.file_id:
        _show_result(scan)


# --------------------------------------------------------------- CARE PLAN
def _toggle(plan_id: str, token: str, key: str):
    op = "$addToSet" if st.session_state[key] else "$pull"
    get_db().plans.update_one({"_id": ObjectId(plan_id)}, {op: {"done": token}})


def _status_text(delta: float) -> str:
    if delta >= 5:
        return " Improving"
    if delta <= -5:
        return " Getting worse - consider consulting an expert"
    return "➖ Stable"


def render_care_plan():
    db = get_db()
    st.title(" 30-Day Care Plan")
    plans = list(db.plans.find({"user_id": _uid(), "status": "active"}).sort("start_date", -1))
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

    elapsed = (_now().date() - plan["start_date"].date()).days + 1
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

    # ---- today's tasks
    st.markdown(f"### Today's tasks - {schedule[day - 1]['phase']}")
    for i, task in enumerate(schedule[day - 1]["tasks"]):
        token, key = f"d{day}t{i}", f"{pid}-d{day}t{i}"
        st.checkbox(task, value=token in plan.get("done", []), key=key,
                    on_change=_toggle, args=(pid, token, key))

    # ---- 24-hour check-in
    st.markdown("### 📸 Daily check-in (every 24 hours)")
    next_time = last["created_at"] + timedelta(hours=24) if last else None
    if next_time and _now() < next_time:
        left = next_time - _now()
        st.info(f"Next check-in available in {left.seconds // 3600}h {(left.seconds % 3600) // 60}m "
                f"(at {_fmt(next_time)}).")
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
                        model_h = health_score(res["probs"], plan["class_name"])
                        photo_class = res["class_name"]
                    except Exception as err:
                        st.error(f"Could not analyze photo: {err}")
                        st.stop()
                self_h = 100 - affected
                health = round(0.6 * model_h + 0.4 * self_h, 1) if model_h is not None else float(self_h)
                prev = last["health_score"] if last else plan["baseline_health"]
                db.checkins.insert_one({
                    "plan_id": pid, "user_id": _uid(), "day": day, "health_score": health,
                    "model_health": model_h, "self_health": self_h, "affected_pct": affected,
                    "delta": round(health - prev, 1), "photo_class": photo_class,
                    "notes": notes.strip(), "created_at": _now(),
                })
                st.rerun()

    # ---- progress
    if checkins:
        st.markdown("###  Improvement")
        delta = last["delta"]
        c1, c2 = st.columns(2)
        c1.metric("Change since last check-in (24h)", f"{delta:+.1f} pts", delta=f"{delta:+.1f}")
        c2.metric("Total change since start", f"{last['health_score'] - plan['baseline_health']:+.1f} pts")
        st.write(f"**Status:** {_status_text(delta)}")
        df = pd.DataFrame({
            "Check-in": [f"#{i + 1} (Day {c['day']})" for i, c in enumerate(checkins)],
            "Health score": [c["health_score"] for c in checkins],
        }).set_index("Check-in")
        st.line_chart(df)
        with st.expander("Check-in history"):
            st.dataframe(pd.DataFrame([{
                "Date": _fmt(c["created_at"]), "Day": c["day"], "Health": c["health_score"],
                "Change": c["delta"], "Affected %": c["affected_pct"], "Notes": c["notes"],
            } for c in reversed(checkins)]), use_container_width=True, hide_index=True)

    with st.expander(" Full 30-day schedule"):
        st.dataframe(pd.DataFrame([{
            "Day": s["day"], "Phase": s["phase"], "Tasks": " • ".join(s["tasks"]),
        } for s in schedule]), use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    if a.button(" Mark plan as completed"):
        db.plans.update_one({"_id": plan["_id"]}, {"$set": {"status": "completed", "end_date": _now()}})
        st.rerun()
    if b.button(" Stop this plan"):
        db.plans.update_one({"_id": plan["_id"]}, {"$set": {"status": "stopped", "end_date": _now()}})
        st.rerun()


# --------------------------------------------------------------- DASHBOARD
def render_dashboard():
    db = get_db()
    name = st.session_state["user"]["name"]
    st.title(f" Dashboard - welcome, {name.split()[0]}")
    scans = list(db.scans.find({"user_id": _uid()}).sort("created_at", -1))
    plans = list(db.plans.find({"user_id": _uid()}))
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
                day = max(1, min((_now().date() - p["start_date"].date()).days + 1, 30))
                health = cks[-1]["health_score"] if cks else p["baseline_health"]
                st.write(f"**{p['nickname']}** ({p['condition']}) - Day {day}/30 · Health {health:.0f}/100")
                st.progress(day / 30)
        else:
            st.caption("No active plans.")

    st.markdown("#### Recent scans")
    st.dataframe(pd.DataFrame([{
        "Date": _fmt(s["created_at"]), "Plant": s["plant"], "Condition": s["condition"],
        "Confidence %": round(s["confidence"] * 100, 1),
    } for s in scans[:15]]), use_container_width=True, hide_index=True)