import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import mesh_utils as mesh_utils
from streamlit_stl import stl_from_text
import numpy as np


# File uploader
image_array = [
    "./images/1.jpg",
    "./images/2.jpg",
    "./images/3.jpg",
    "./images/11.jpg",  # the running dog deserves an early spot
    "./images/4.jpg",
    "./images/5.jpg",
    "./images/6.jpg",
    "./images/7.jpg",
    "./images/8.jpg",
    "./images/9.jpg",
    "./images/10.jpg",
]

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


def reset_img():
    st.session_state["img"] = None
    st.session_state["points"] = []


def dropdown_preloaded_loadnew():
    preloaded = "Preloaded image"
    upload_new = "upload new image"
    opt = st.selectbox(
        "Upload new image or use preloaded ones",
        [preloaded, upload_new],
        on_change=reset_img,
    )
    if opt == preloaded:
        st.session_state["upload"] = False
        if not st.session_state["img"]:
            st.session_state["img"] = Image.open(
                image_array[st.session_state["img_index"]]
            )
    else:
        st.session_state["upload"] = True


def upload_imgs():
    if not st.session_state["upload"]:
        return
    uploaded_file = st.file_uploader(
        "Upload a picture of a dog", type=["jpg", "jpeg", "png"], on_change=reset_img
    )
    if not st.session_state["img"] and uploaded_file:
        st.session_state["img"] = Image.open(uploaded_file)


def scroll_imgs_buttons():
    if st.session_state["upload"]:
        return
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️"):
            st.session_state["img_index"] -= 1
            reset_img()
            if st.session_state["img_index"] < 0:
                st.session_state["img_index"] = len(image_array) - 1  # wrap around
            st.rerun()
    with col2:
        if st.button("➡️"):
            st.session_state["img_index"] += 1
            reset_img()
            if st.session_state["img_index"] >= len(image_array):
                st.session_state["img_index"] = 0
            st.rerun()


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
    if not st.session_state["img"]:
        return
    img = st.session_state["img"]
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
    content = None
    for progress, mesh_res in mesh_utils.export_skeleton_mesh(
        [np.array(joint) for joint in st.session_state["points"]]
    ):
        progress_bar.progress(progress, text=progress_text)
        if mesh_res:
            content = mesh_res
            break
    assert content is not None
    stl_from_text(content, cam_v_angle=0, cam_h_angle=180, color="#FF9900")


def rerun_skeleton_optimizer():
    if st.button("rerun optimizer"):
        st.rerun()


def main():
    st.title("📸 Dog skeleton fiting from keypoints")
    dropdown_preloaded_loadnew()
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
    st.session_state["img"] = None
    st.session_state["img_index"] = 0
    st.session_state["points"] = []
    st.session_state["last_clicked"] = None
    st.session_state["upload"] = False
main()
