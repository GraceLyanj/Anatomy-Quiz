import cv2
import numpy as np
import os
import streamlit as st
import torch
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import parse_qs, urlparse
import gdown
from segment_anything import sam_model_registry, SamPredictor
from streamlit_image_coordinates import streamlit_image_coordinates

from btcv_loader import BTCVDataLoader


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
SAM_CHECKPOINT = APP_DIR / "ckpt" / "sam_vit_b_01ec64.pth"
BTCV_PATH = ROOT_DIR / "IMIS-Bench-main" / "dataset" / "BTCV"
DISPLAY_SIZE = 512


def is_google_drive_url(url):
    return "drive.google.com" in url


def normalize_google_drive_url(url):
    parsed = urlparse(url)
    # Handle /file/d/<id>/view links
    if "/file/d/" in parsed.path:
        file_id = parsed.path.split("/file/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?id={file_id}"

    # Handle query-style links with id=...
    query = parse_qs(parsed.query)
    file_id = query.get("id", [None])[0]
    if file_id:
        return f"https://drive.google.com/uc?id={file_id}"

    return url


def validate_checkpoint_file(path):
    if not path.exists():
        return False, "Checkpoint file was not created."

    # SAM ViT-B checkpoint is ~357MB; much smaller files are usually HTML error pages.
    if path.stat().st_size < 100 * 1024 * 1024:
        return False, "Downloaded file is too small and is likely not the model checkpoint."

    with open(path, "rb") as f:
        header = f.read(512).lower()
    if b"<html" in header or b"<!doctype" in header:
        return False, "Downloaded content appears to be an HTML page, not a .pth file."

    return True, ""


def download_checkpoint(checkpoint_url):
    SAM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if is_google_drive_url(checkpoint_url):
        normalized_url = normalize_google_drive_url(checkpoint_url)
        gdown.download(url=normalized_url, output=str(SAM_CHECKPOINT), quiet=False)
    else:
        urlretrieve(checkpoint_url, SAM_CHECKPOINT)

    is_valid, reason = validate_checkpoint_file(SAM_CHECKPOINT)
    if not is_valid:
        SAM_CHECKPOINT.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checkpoint download failed validation: {reason} "
            "Use a true direct-download URL to the .pth binary."
        )


def ensure_checkpoint(force_download=False):
    checkpoint_url = os.getenv("SAM_CHECKPOINT_URL", "").strip()

    if SAM_CHECKPOINT.exists() and not force_download:
        is_valid, reason = validate_checkpoint_file(SAM_CHECKPOINT)
        if is_valid:
            return
        SAM_CHECKPOINT.unlink(missing_ok=True)
        if not checkpoint_url:
            raise RuntimeError(
                f"Existing checkpoint is invalid: {reason} "
                "Set SAM_CHECKPOINT_URL to redownload a valid file."
            )

    if not checkpoint_url:
        raise FileNotFoundError(
            f"SAM checkpoint not found at: {SAM_CHECKPOINT}. "
            "Set SAM_CHECKPOINT_URL in deployment environment to auto-download it."
        )

    download_checkpoint(checkpoint_url)


@st.cache_resource
def load_predictor():
    ensure_checkpoint()
    device = torch.device("cpu")
    try:
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    except Exception as e:
        # Most common cause in deployment is a corrupted/incomplete checkpoint download.
        if "UnpicklingError" in type(e).__name__ or "pickle" in str(e):
            ensure_checkpoint(force_download=True)
            sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
        else:
            raise
    sam.to(device)
    return SamPredictor(sam)


@st.cache_resource
def load_btcv_loader():
    dataset_json = BTCV_PATH / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(
            f"BTCV dataset not found at: {BTCV_PATH}. "
            "Expected to find dataset.json in that directory."
        )
    return BTCVDataLoader(str(BTCV_PATH))


def init_state():
    defaults = {
        "score": 0,
        "current_image_index": 0,
        "current_question_data": None,
        "current_image": None,
        "user_mask": None,
        "overlay_image": None,
        "last_click": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_overlay(base_image, user_mask=None, ground_truth=None):
    overlay = base_image.copy()
    if ground_truth is not None:
        overlay[ground_truth > 0] = overlay[ground_truth > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
    if user_mask is not None:
        if user_mask.shape != overlay.shape[:2]:
            user_mask = cv2.resize(
                user_mask.astype(np.uint8),
                (overlay.shape[1], overlay.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        overlay[user_mask > 0] = overlay[user_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    return overlay.astype(np.uint8)


def load_question(loader, predictor):
    question_data = loader.get_random_organ_question(st.session_state["current_image_index"])
    if question_data is None:
        st.error("No organs found in this image.")
        return

    st.session_state["current_question_data"] = question_data
    st.session_state["current_image"] = question_data["image"]
    st.session_state["user_mask"] = None
    st.session_state["overlay_image"] = question_data["image"]
    st.session_state["last_click"] = None
    predictor.set_image(st.session_state["current_image"])


def segment_at_click(predictor, click_x, click_y):
    image = st.session_state["current_image"]
    h, w = image.shape[:2]
    scale_x = w / DISPLAY_SIZE
    scale_y = h / DISPLAY_SIZE

    orig_x = int(click_x * scale_x)
    orig_y = int(click_y * scale_y)

    input_point = np.array([[orig_x, orig_y]])
    input_label = np.array([1])

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    st.session_state["user_mask"] = masks[0]
    st.session_state["overlay_image"] = render_overlay(
        st.session_state["current_image"],
        user_mask=st.session_state["user_mask"],
    )
    st.session_state["last_click"] = (click_x, click_y)


def check_answer(loader):
    if st.session_state["user_mask"] is None:
        st.warning("Click on the image first.")
        return

    question = st.session_state["current_question_data"]
    ground_truth = question["ground_truth_mask"]
    user_mask = st.session_state["user_mask"]

    if user_mask.shape != ground_truth.shape:
        user_mask = cv2.resize(
            user_mask.astype(np.uint8),
            (ground_truth.shape[1], ground_truth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    dice_score = loader.calculate_dice_score(user_mask, ground_truth)
    organ_name = question["organ_name"]

    if dice_score > 0.7:
        st.session_state["score"] += 10
        st.success(
            f"Excellent! You correctly identified the {organ_name}. "
            f"Accuracy: {dice_score * 100:.1f}%. Points: +10."
        )
    elif dice_score > 0.5:
        st.session_state["score"] += 5
        st.info(
            f"Good! You partially identified the {organ_name}. "
            f"Accuracy: {dice_score * 100:.1f}%. Points: +5."
        )
    elif dice_score > 0.2:
        st.warning(
            f"Partially correct for {organ_name}. Accuracy: {dice_score * 100:.1f}%. "
            "Try covering more of the organ area."
        )
    else:
        st.error(
            f"Incorrect for {organ_name}. Accuracy: {dice_score * 100:.1f}%. "
            "Try again in a different region."
        )

    st.session_state["overlay_image"] = render_overlay(
        st.session_state["current_image"],
        user_mask=st.session_state["user_mask"],
        ground_truth=ground_truth,
    )


def main():
    st.set_page_config(page_title="Anatomy Teacher Web", layout="centered")
    st.title("Anatomy Teaching App (Web)")
    st.caption("Green = your selection, Blue = correct answer")

    if not SAM_CHECKPOINT.exists() and not os.getenv("SAM_CHECKPOINT_URL", "").strip():
        st.error(
            "Missing SAM model checkpoint. Either add "
            "`anatomy-teacher-app/ckpt/sam_vit_b_01ec64.pth` locally "
            "or set `SAM_CHECKPOINT_URL` in deployment to auto-download."
        )
        st.stop()

    if not (BTCV_PATH / "dataset.json").exists():
        st.error(
            "Missing BTCV dataset. Add the dataset at "
            "`IMIS-Bench-main/dataset/BTCV` (including `dataset.json`)."
        )
        st.stop()

    init_state()
    predictor = load_predictor()
    loader = load_btcv_loader()

    if st.session_state["current_question_data"] is None:
        load_question(loader, predictor)

    question = st.session_state["current_question_data"]
    st.subheader(f"Score: {st.session_state['score']}")
    st.write(question["question"])

    display_image = cv2.resize(st.session_state["overlay_image"], (DISPLAY_SIZE, DISPLAY_SIZE))

    click = streamlit_image_coordinates(
        display_image,
        key=f"image-{st.session_state['current_image_index']}",
        width=DISPLAY_SIZE,
        height=DISPLAY_SIZE,
    )

    if click is not None:
        click_x = int(click["x"])
        click_y = int(click["y"])
        if st.session_state["last_click"] != (click_x, click_y):
            segment_at_click(predictor, click_x, click_y)
            st.rerun()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Submit Answer", use_container_width=True):
            check_answer(loader)

    with col2:
        if st.button("Clear Selection", use_container_width=True):
            st.session_state["user_mask"] = None
            st.session_state["overlay_image"] = st.session_state["current_image"]
            st.session_state["last_click"] = None
            st.rerun()

    with col3:
        if st.button("Next Question", use_container_width=True):
            st.session_state["current_image_index"] += 1
            if st.session_state["current_image_index"] >= len(loader.training_data):
                st.session_state["current_image_index"] = 0
                st.info(f"Wrapped to start of dataset. Current score: {st.session_state['score']}")
            load_question(loader, predictor)
            st.rerun()


if __name__ == "__main__":
    main()
