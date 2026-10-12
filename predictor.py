import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "plant_disease_cnn.keras")
CLASS_PATH = os.path.join(BASE_DIR, "saved_model", "class_names.json")
IMG_SIZE = 128
LOW_CONFIDENCE = 0.60


@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def model_files_exist() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(CLASS_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def plant_key(class_name: str) -> str:
    return class_name.split("___")[0]


def health_score(probs, class_name: str) -> float:
    """0-100. Probability of the 'healthy' class of the same plant;
    falls back to (1 - P(class)) if the plant has no healthy class."""
    names = load_class_names()
    plant = plant_key(class_name).lower()
    for i, n in enumerate(names):
        if n.lower().startswith(plant) and "healthy" in n.lower():
            return float(probs[i]) * 100
    return (1 - float(probs[names.index(class_name)])) * 100


def predict(image: Image.Image) -> dict:
    names = load_class_names()
    probs = load_model().predict(preprocess(image), verbose=0)[0]
    if len(probs) != len(names):
        raise ValueError("Model output does not match class_names.json")
    top = np.argsort(probs)[::-1][:3]
    best = int(top[0])
    return {
        "class_name": names[best],
        "confidence": float(probs[best]),
        "top3": [{"class_name": names[int(i)], "score": float(probs[int(i)])} for i in top],
        "probs": [float(p) for p in probs],
        "health_score": health_score(probs, names[best]),
    }