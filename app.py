import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="MACE-X² Research Demo",
    page_icon="🩻",
    layout="wide"
)

st.title("MACE-X²")
st.subheader("Medical Image Anomaly Detection — Research Prototype v0.1")

st.warning(
    "Research prototype only — not intended for clinical diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded X-ray",
        width=500
    )

    st.success("Image uploaded successfully.")

st.divider()

st.caption(
    "MACE-X² | Global + Local Evidence Research Framework"
)

print("✅ STEP 41 COMPLETE — STREAMLIT TEST APP READY")