import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import requests
import streamlit as st
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from db import get_db

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

RESEND_URL = "https://api.resend.com/emails"
RESET_TTL_MINUTES = 30          # how long a reset link stays valid
RESET_COOLDOWN_SECONDS = 60     # minimum gap between reset emails for one account
GENERIC_RESET_MSG = ("If an account exists for that email, a password reset link has been sent. "
                     "Check your inbox (and spam folder). The link is valid for "
                     f"{RESET_TTL_MINUTES} minutes.")


def _now():
    return datetime.now(timezone.utc)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ==================================================
# SIGN UP / LOGIN / LOGOUT
# ==================================================
def signup(name: str, email: str, password: str, confirm: str):
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
            "name": name, "email": email, "password_hash": hashed,
            "created_at": _now(),
        })
    except DuplicateKeyError:
        return False, "This email is already registered. Please log in."
    return True, "Account created. You can log in now."


def login(email: str, password: str):
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


# ==================================================
# FORGOT PASSWORD (Resend email)
# ==================================================
def _reset_email_html(name: str, link: str) -> str:
    return f"""
    <div style="font-family:Arial,Helvetica,sans-serif;background:#f3f7f4;padding:24px;">
      <div style="max-width:520px;margin:auto;background:#ffffff;border-radius:12px;
                  padding:32px;border-top:6px solid #74C69D;">
        <h2 style="margin:0 0 12px;color:#0A1B14;">Reset your password</h2>
        <p style="color:#333;font-size:15px;line-height:1.6;">Hi {name},</p>
        <p style="color:#333;font-size:15px;line-height:1.6;">
          We received a request to reset your PlantCare AI password.
          Click the button below to choose a new one. This link expires in
          {RESET_TTL_MINUTES} minutes and can be used only once.
        </p>
        <p style="text-align:center;margin:28px 0;">
          <a href="{link}" style="background:#74C69D;color:#08261A;text-decoration:none;
             font-weight:bold;padding:13px 28px;border-radius:8px;display:inline-block;">
            Reset password
          </a>
        </p>
        <p style="color:#666;font-size:13px;line-height:1.6;">
          If the button does not work, copy this link into your browser:<br>
          <span style="word-break:break-all;">{link}</span>
        </p>
        <p style="color:#666;font-size:13px;">
          If you did not request this, you can safely ignore this email.
          Your password will not change.
        </p>
      </div>
    </div>
    """


def send_reset_email(to_email: str, name: str, link: str):
    """Send the reset link through the Resend REST API."""
    api_key = st.secrets.get("RESEND_API_KEY")
    if not api_key:
        return False, "RESEND_API_KEY is missing in secrets."
    sender = st.secrets.get("RESEND_FROM", "PlantCare AI <onboarding@resend.dev>")
    try:
        resp = requests.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from": sender,
                "to": [to_email],
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


def request_password_reset(email: str):
    """Always returns the same public message so nobody can discover which emails are registered.
    Real errors are printed to the server console for the developer."""
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        return False, "Please enter a valid email address."

    db = get_db()
    user = db.users.find_one({"email": email})
    if not user:
        return True, GENERIC_RESET_MSG

    recent = db.password_resets.find_one({
        "user_id": str(user["_id"]),
        "created_at": {"$gt": _now() - timedelta(seconds=RESET_COOLDOWN_SECONDS)},
    })
    if recent:
        return True, GENERIC_RESET_MSG

    token = secrets.token_urlsafe(32)
    db.password_resets.insert_one({
        "user_id": str(user["_id"]),
        "token_hash": _hash_token(token),
        "created_at": _now(),
        "expires_at": _now() + timedelta(minutes=RESET_TTL_MINUTES),
    })

    app_url = str(st.secrets.get("APP_URL", "http://localhost:8501")).rstrip("/")
    ok, detail = send_reset_email(email, user["name"], f"{app_url}/?reset_token={token}")
    if not ok:
        print(f"[password reset] email failed for {email}: {detail}")
        db.password_resets.delete_one({"token_hash": _hash_token(token)})
    return True, GENERIC_RESET_MSG


def _find_valid_reset(token: str):
    return get_db().password_resets.find_one({
        "token_hash": _hash_token(token),
        "expires_at": {"$gt": _now()},
    })


def reset_password(token: str, password: str, confirm: str):
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
    # one-time use: remove this and any other outstanding links for the account
    db.password_resets.delete_many({"user_id": record["user_id"]})
    return True, "Password updated. You can now log in with your new password."


# ==================================================
# UI
# ==================================================
def _render_reset_form(token: str):
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
    st.title("PlantCare AI")
    st.markdown("##### AI-powered leaf analysis, treatment guidance and 30-day recovery tracking")
    st.divider()
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        token = st.query_params.get("reset_token")
        if token:
            _render_reset_form(token)
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