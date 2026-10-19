"""Run with: uv run streamlit run app/streamlit_app.py"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from introduction_to_deeplearning.config import MODELS_DIR
from introduction_to_deeplearning.predict import load_predictor, predict_bytes
from introduction_to_deeplearning.explain import make_gradcam

st.title("NeuroVision AI")
st.write("Upload a brain MRI image to explore your trained model.")
st.warning("Educational prototype. This is not a medical diagnosis. A 'notumor' prediction does not rule out disease.")

# Each experiment keeps its own model and matching class order in one folder.
runs = sorted([p.parent for p in MODELS_DIR.glob("*/best.keras")], reverse=True)
if not runs:
    st.info("Train a model in notebooks/training.ipynb first. Its best model will appear here.")
    st.stop()
run_dir = st.selectbox("Training run", runs, format_func=lambda p: p.name)


@st.cache_resource
def get_model(path, modified_time):
    # The timestamp refreshes this cache if training replaces a checkpoint.
    return load_predictor(path)


upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
if upload is not None:
    try:
        model, names = get_model(str(run_dir), (run_dir / "best.keras").stat().st_mtime_ns)
        result = predict_bytes(upload.getvalue(), model, names)
        st.image(result["image"][..., 0], caption="Model input: grayscale, 224 × 224", width=280)
        st.subheader(f"Predicted class: {result['class']}")
        st.bar_chart(pd.Series(result["scores"], name="Model score"))
        st.caption("Softmax scores are model outputs, not calibrated medical confidence.")
        if st.checkbox("Show Grad-CAM"):
            heatmap = make_gradcam(result["image"], model)
            fig, ax = plt.subplots()
            ax.imshow(result["image"][..., 0], cmap="gray")
            ax.imshow(heatmap, cmap="jet", alpha=0.4, vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Highlighted areas influence the class score. This is not a tumor segmentation.")
    except (ValueError, OSError) as error:
        st.error(str(error))
