import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import mesh_utils as mesh_utils
from streamlit_stl import stl_from_file
import numpy as np


# File uploader
image_array = []
points_var_names = [
    "nose",
    "head",
    "neck",
    "left shoulder",
    "left elbow",
    "left wrist",
    "left front paw",
    "right shoulder",
    "right elbow",
    "right wrist",
    "right front paw",
    "left hip",
    "left knee",
    "left ankle",
    "left back paw",
    "right hip",
    "right knee",
    "right ankle",
    "right back paw",
]


def upload_imgs():
    uploaded_files = st.file_uploader(
        "Add a dog file", type=["jpg", "jpeg"], accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            image_array.append(image)


def scroll_imgs_buttons():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️"):
            st.session_state.img_index -= 1
            st.session_state["points"] = []
            if st.session_state.img_index < 0:
                st.session_state.img_index = len(image_array) - 1  # wrap around
    with col2:
        if st.button("➡️"):
            st.session_state.img_index += 1
            st.session_state["points"] = []
            if st.session_state.img_index >= len(image_array):
                st.session_state.img_index = 0


def get_ellipse_coords(point, radius=5):
    x, y = point
    return [x - radius, y - radius, x + radius, y + radius]


def keypoints_generated() -> bool:
    return st.session_state["points"] and len(st.session_state["points"]) == len(
        points_var_names
    )


def text_body_part():
    print(len(st.session_state["points"]))
    if keypoints_generated():
        st.text("Keypoints selection done")
        return
    st.text(f"Click on {points_var_names[len(st.session_state["points"])]}")


def display_img():
    img = image_array[st.session_state.img_index]
    draw = ImageDraw.Draw(img)
    if keypoints_generated():
        for point in st.session_state["points"]:
            coords = get_ellipse_coords(point)
            draw.ellipse(coords, fill="green")
        st.image(img, use_container_width=True)
        return

    if st.session_state["points"]:
        coords = get_ellipse_coords(st.session_state["points"][-1])
        draw.ellipse(coords, fill="red")

    value = streamlit_image_coordinates(img, key="pil")
    if not value:
        return
    point = (value["x"], value["y"])
    if point == st.session_state["last_clicked"]:
        return
    st.session_state["last_clicked"] = point
    st.session_state["points"].append(point)
    st.rerun()


def previous_keypoint_button():
    if not st.session_state["points"]:
        return
    if st.button("previous keypoint"):
        st.session_state["points"].pop()
        st.rerun()


def render_skeleton():
    progress_text = "Might take a minute"
    progress_bar = st.progress(0, text=progress_text)
    for progress, mesh_res in mesh_utils.export_skeleton_mesh(
        [np.array(joint) for joint in st.session_state["points"]]
    ):
        progress_bar.progress(progress, text=progress_text)
        if mesh_res:
            break
    stl_from_file(
        mesh_utils.DOG_SKELETON_FILE,
        cam_v_angle=0,
        cam_h_angle=180,
        auto_rotate=False,
        color="#FF9900",
    )


def rerun_skeleton_optimizer():
    if st.button("rerun optimizer"):
        st.rerun()


def main():
    st.title("📸 Dog skeleton fiting from keypoints")
    upload_imgs()
    if not image_array:
        return
    scroll_imgs_buttons()
    text_body_part()
    display_img()
    print(st.session_state["points"])
    previous_keypoint_button()
    if not keypoints_generated():
        return
    render_skeleton()
    rerun_skeleton_optimizer()


if "img_index" not in st.session_state:
    st.session_state.img_index = 0
    st.session_state["points"] = []
    st.session_state["last_clicked"] = None
main()
