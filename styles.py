import base64
import os
import streamlit as st


@st.cache_data
def _encode(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def inject_css(background_path: str) -> None:
    if os.path.exists(background_path):
        bg = (
            "background-image: linear-gradient(rgba(8,25,17,.78), rgba(8,25,17,.88)),"
            f'url("data:image/png;base64,{_encode(background_path)}");'
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