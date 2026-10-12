import re
from datetime import datetime, timezone

import bcrypt
import streamlit as st
from pymongo.errors import DuplicateKeyError

from db import get_db

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


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
            "created_at": datetime.now(timezone.utc),
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


def render_auth():
    st.title("🌿 Plant Disease Detection")
    st.markdown("##### AI-powered leaf analysis, treatment guidance and 30-day recovery tracking")
    st.divider()
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        tab_login, tab_signup = st.tabs([" Login", " Sign Up"])
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