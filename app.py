import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from src.inference import MACEX2Inference


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="MACE-X² Research Demo",
    page_icon="🩻",
    layout="wide"
)


# ============================================================
# LOAD MODEL ONCE
# ============================================================

@st.cache_resource
def load_engine():
    return MACEX2Inference()


engine = load_engine()


# ============================================================
# HEADER
# ============================================================

st.title("MACE-X²")
st.subheader(
    "Multi-Witness Medical Image Anomaly Detection"
)

st.caption(
    "Research Prototype v0.1"
)

st.warning(
    "Research use only — this prototype is not intended "
    "for clinical diagnosis or medical decision-making."
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    left, right = st.columns(
        [1, 1.4]
    )


    # ========================================================
    # IMAGE
    # ========================================================

    with left:

        st.markdown(
            "### Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )


    # ========================================================
    # ANALYSIS
    # ========================================================

    with right:

        st.markdown(
            "### MACE-X² Evidence Analysis"
        )

        analyze = st.button(
            "Run Research Analysis",
            type="primary",
            use_container_width=True
        )


        if analyze:

            with st.spinner(
                "Extracting global and local evidence..."
            ):

                result = engine.predict(
                    image
                )


            st.success(
                "Analysis complete."
            )


            # =================================================
            # EVIDENCE LEVEL
            # =================================================

            st.markdown(
                "### Research Evidence Level"
            )

            evidence = result[
                "evidence_level"
            ]

            if evidence == "Higher anomaly evidence":

                st.error(
                    "Higher anomaly evidence"
                )

            elif evidence == "Intermediate anomaly evidence":

                st.warning(
                    "Intermediate anomaly evidence"
                )

            else:

                st.success(
                    "Lower anomaly evidence"
                )


            # =================================================
            # SCORES
            # =================================================

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Global Witness",
                f"{result['global_score']:.4f}"
            )

            c2.metric(
                "Local Witness",
                f"{result['local_score']:.4f}"
            )

            c3.metric(
                "MACE C1 Score",
                f"{result['mace_score']:.4f}"
            )


            # =================================================
            # THRESHOLDS
            # =================================================

            st.markdown(
                "### Locked Research Thresholds"
            )

            st.write(
                f"Primary threshold: "
                f"`{result['primary_threshold']:.4f}`"
            )

            st.write(
                f"High-sensitivity threshold: "
                f"`{result['high_sensitivity_threshold']:.4f}`"
            )


            # =================================================
            # LOCAL EVIDENCE MAP
            # =================================================

            st.markdown(
                "### Local Spatial Evidence Map"
            )

            fig, ax = plt.subplots(
                figsize=(5, 5)
            )

            heatmap = ax.imshow(
                result["local_map"],
                interpolation="nearest"
            )

            ax.set_title(
                "7 × 7 Local Anomaly Evidence"
            )

            ax.set_xlabel(
                "Spatial feature position"
            )

            ax.set_ylabel(
                "Spatial feature position"
            )

            fig.colorbar(
                heatmap,
                ax=ax,
                label="Local anomaly score"
            )

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)


            # =================================================
            # INTERPRETATION
            # =================================================

            st.info(
                "The local map represents CNN spatial feature "
                "locations. It should not be interpreted as "
                "clinically segmented anatomical regions."
            )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.markdown(
    "### About the Research Prototype"
)

st.markdown(
    """
**Global Witness**
- Frozen ImageNet-pretrained ResNet18
- 512-dimensional whole-image representation
- Cosine 3-nearest-normal anomaly score

**Local Witness**
- ResNet18 layer4 spatial features
- 7 × 7 feature grid
- 49 position-aware normal reference prototypes

**MACE Configuration-1**
- Global weight: **0.75**
- Local weight: **0.25**

The current research experiments found substantial
correlation between the Global and Local witnesses.
The initial MACE Configuration-1 therefore did not
demonstrate a clear AUROC improvement over the strongest
global-only baseline.
"""
)


st.divider()

st.caption(
    "MACE-X² Research Prototype v0.1 | "
    "Independent Collaborative Research"
)