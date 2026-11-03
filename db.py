import certifi
import streamlit as st
from pymongo import MongoClient


@st.cache_resource(show_spinner="Connecting to database...")
def get_db():
    """Single cached MongoDB connection. Credentials come from st.secrets only."""
    client = MongoClient(
        st.secrets["MONGO_URI"],
        serverSelectionTimeoutMS=8000,
        tlsCAFile=certifi.where(),
        tz_aware=True,
    )
    client.admin.command("ping")  # fail fast if unreachable
    db = client[st.secrets.get("DB_NAME", "plant_disease_app")]

    db.users.create_index("email", unique=True)
    db.scans.create_index([("user_id", 1), ("created_at", -1)])
    db.plans.create_index([("user_id", 1), ("status", 1)])
    db.checkins.create_index([("plan_id", 1), ("created_at", 1)])

    # Password reset tokens (only a SHA-256 hash of each token is stored)
    db.password_resets.create_index("token_hash", unique=True)
    db.password_resets.create_index([("user_id", 1), ("created_at", -1)])
    # MongoDB deletes each document automatically once expires_at has passed
    db.password_resets.create_index("expires_at", expireAfterSeconds=0)
    return db