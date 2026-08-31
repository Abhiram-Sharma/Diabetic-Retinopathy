import os

# ============================================================
# BACKEND CONFIGURATION
# ============================================================

# Keras 3 with PyTorch backend.
# This must be configured before importing Keras.
os.environ["KERAS_BACKEND"] = "torch"

import io
import numpy as np
import pydicom
import streamlit as st
import keras
import torch
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

MODEL_PATH = "final_model.keras"
SAMPLES_DIR = "samples"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy",
]

CLASS_SHORT_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative",
]

HEATMAP_ALPHA = 0.40


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="DR",
    layout="wide",
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the locally stored Keras model.

    The model is loaded once and cached by Streamlit.
    """

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = keras.saving.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# DICOM READING
# ============================================================

def read_dicom_bytes(file_bytes):
    """
    Read DICOM bytes and convert the pixel data into
    an RGB PIL image.

    Returns:
        image
        dataset
    """

    if not file_bytes:
        raise ValueError(
            "The DICOM file is empty."
        )

    dataset = pydicom.dcmread(
        io.BytesIO(file_bytes)
    )

    if not hasattr(dataset, "PixelData"):
        raise ValueError(
            "The DICOM file does not contain pixel data."
        )

    pixel_array = dataset.pixel_array.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Handle MONOCHROME1 images.
    # --------------------------------------------------------

    photometric = getattr(
        dataset,
        "PhotometricInterpretation",
        ""
    )

    if photometric == "MONOCHROME1":

        pixel_array = (
            np.max(pixel_array)
            - pixel_array
        )

    # --------------------------------------------------------
    # Handle multi-frame images.
    # --------------------------------------------------------

    if pixel_array.ndim == 3:

        # RGB / RGBA image.
        if pixel_array.shape[-1] in (3, 4):
            pass

        # Multiple grayscale frames.
        else:
            pixel_array = pixel_array[0]

    elif pixel_array.ndim != 2:

        raise ValueError(
            "Unsupported DICOM pixel dimensions: "
            f"{pixel_array.shape}"
        )

    # --------------------------------------------------------
    # Normalize to 0-255.
    # --------------------------------------------------------

    minimum = np.min(pixel_array)
    maximum = np.max(pixel_array)

    if maximum > minimum:

        pixel_array = (
            (pixel_array - minimum)
            / (maximum - minimum)
            * 255.0
        )

    else:

        pixel_array = np.zeros_like(
            pixel_array
        )

    pixel_array = np.clip(
        pixel_array,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Convert to RGB.
    # --------------------------------------------------------

    if pixel_array.ndim == 2:

        image = Image.fromarray(
            pixel_array,
            mode="L"
        ).convert("RGB")

    else:

        image = Image.fromarray(
            pixel_array
        ).convert("RGB")

    return image, dataset


def read_dicom_file(file_object):
    """
    Read a DICOM file-like object.
    """

    file_bytes = file_object.getvalue()

    return read_dicom_bytes(
        file_bytes
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Prepare image for model inference.

    The image is:
        1. Converted to RGB.
        2. Resized to 224 x 224.
        3. Converted to float32.
        4. Expanded to batch dimension.

    Pixel values remain in the 0-255 range because the
    saved Keras model is assumed to contain its own
    preprocessing layers.

    IMPORTANT:
    The preprocessing must match the preprocessing used
    during model training.
    """

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(model, image):
    """
    Run model inference.

    Returns:
        predicted_index
        confidence
        probabilities
    """

    input_array = preprocess_image(
        image
    )

    predictions = model(
        input_array,
        training=False
    )

    # --------------------------------------------------------
    # Convert tensor output to NumPy.
    # --------------------------------------------------------

    if isinstance(
        predictions,
        torch.Tensor
    ):

        predictions = (
            predictions
            .detach()
            .cpu()
            .numpy()
        )

    else:

        predictions = np.asarray(
            predictions
        )

    predictions = np.asarray(
        predictions
    ).squeeze()

    # --------------------------------------------------------
    # Verify five-class output.
    # --------------------------------------------------------

    if predictions.ndim != 1:

        raise ValueError(
            "Unexpected model output shape: "
            f"{predictions.shape}"
        )

    if len(predictions) != 5:

        raise ValueError(
            "The loaded model does not produce exactly "
            f"5 outputs. Detected {len(predictions)}."
        )

    # --------------------------------------------------------
    # Convert logits to probabilities if necessary.
    # --------------------------------------------------------

    probability_sum = float(
        np.sum(predictions)
    )

    if (
        np.min(predictions) < 0
        or
        not np.isclose(
            probability_sum,
            1.0,
            atol=1e-3
        )
    ):

        predictions = torch.softmax(
            torch.tensor(
                predictions,
                dtype=torch.float32
            ),
            dim=0
        ).numpy()

    probabilities = np.asarray(
        predictions,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Final prediction.
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_index,
        confidence,
        probabilities
    )


# ============================================================
# XAI - INPUT GRADIENT SALIENCY
# ============================================================

def generate_saliency_map(
    model,
    image,
    class_index
):
    """
    Generate an input-gradient saliency map.

    The gradient of the selected class score with respect
    to the input image is used to identify pixels that have
    the greatest influence on the model output.

    This is NOT Grad-CAM.
    """

    input_array = preprocess_image(
        image
    )

    input_tensor = torch.tensor(
        input_array,
        dtype=torch.float32,
        requires_grad=True
    )

    # --------------------------------------------------------
    # Clear model gradients.
    # --------------------------------------------------------

    model.zero_grad()

    # --------------------------------------------------------
    # Forward pass.
    # --------------------------------------------------------

    predictions = model(
        input_tensor,
        training=False
    )

    if not isinstance(
        predictions,
        torch.Tensor
    ):

        predictions = torch.as_tensor(
            predictions
        )

    # --------------------------------------------------------
    # Select target class.
    # --------------------------------------------------------

    target_score = predictions[
        0,
        class_index
    ]

    # --------------------------------------------------------
    # Calculate input gradients.
    # --------------------------------------------------------

    gradients = torch.autograd.grad(
        outputs=target_score,
        inputs=input_tensor,
        retain_graph=False,
        create_graph=False
    )[0]

    # --------------------------------------------------------
    # Convert RGB gradients into a single pixel importance
    # value.
    # --------------------------------------------------------

    saliency = torch.abs(
        gradients
    )

    saliency = torch.max(
        saliency,
        dim=-1
    ).values

    saliency = (
        saliency[0]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Normalize to 0-1.
    # --------------------------------------------------------

    minimum = np.min(
        saliency
    )

    maximum = np.max(
        saliency
    )

    if maximum > minimum:

        saliency = (
            saliency - minimum
        ) / (
            maximum - minimum
        )

    else:

        saliency = np.zeros_like(
            saliency
        )

    # --------------------------------------------------------
    # Suppress low-level background noise.
    # --------------------------------------------------------

    threshold = np.percentile(
        saliency,
        70
    )

    saliency[
        saliency < threshold
    ] = 0

    maximum = np.max(
        saliency
    )

    if maximum > 0:

        saliency = (
            saliency / maximum
        )

    return saliency


def generate_xai_heatmap(
    model,
    image,
    class_index
):
    """
    Generate XAI heatmap and return the method name.
    """

    heatmap = generate_saliency_map(
        model=model,
        image=image,
        class_index=class_index
    )

    return (
        heatmap,
        "Input Gradient Saliency"
    )


# ============================================================
# HEATMAP OVERLAY
# ============================================================

def create_heatmap_overlay(
    image,
    heatmap
):
    """
    Overlay the XAI heatmap on the original retinal image.
    """

    original = image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Resize heatmap.
    # --------------------------------------------------------

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        original.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = (
        np.asarray(
            heatmap_image
        )
        / 255.0
    )

    # --------------------------------------------------------
    # Convert grayscale heatmap to color.
    # --------------------------------------------------------

    colormap = plt.get_cmap(
        "jet"
    )

    colored_heatmap = colormap(
        heatmap_array
    )[:, :, :3]

    colored_heatmap = (
        colored_heatmap * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Blend.
    # --------------------------------------------------------

    original_array = np.asarray(
        original
    ).astype(np.float32)

    overlay = (
        (1 - HEATMAP_ALPHA)
        * original_array
        +
        HEATMAP_ALPHA
        * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        overlay
    )


# ============================================================
# XAI REPORT
# ============================================================

def create_xai_report(
    original,
    overlay,
    heatmap,
    prediction,
    confidence
):
    """
    Create a downloadable three-panel XAI report.
    """

    figure = plt.figure(
        figsize=(15, 5)
    )

    # --------------------------------------------------------
    # Original.
    # --------------------------------------------------------

    ax1 = figure.add_subplot(
        1,
        3,
        1
    )

    ax1.imshow(
        original
    )

    ax1.set_title(
        "Original Retinal Image"
    )

    ax1.axis("off")

    # --------------------------------------------------------
    # Overlay.
    # --------------------------------------------------------

    ax2 = figure.add_subplot(
        1,
        3,
        2
    )

    ax2.imshow(
        overlay
    )

    ax2.set_title(
        "XAI Saliency Overlay"
    )

    ax2.axis("off")

    # --------------------------------------------------------
    # Heatmap.
    # --------------------------------------------------------

    ax3 = figure.add_subplot(
        1,
        3,
        3
    )

    ax3.imshow(
        heatmap,
        cmap="jet"
    )

    ax3.set_title(
        "Saliency Heatmap"
    )

    ax3.axis("off")

    figure.suptitle(
        "Prediction: "
        f"{prediction} | "
        f"Confidence: {confidence:.2%}",
        fontsize=14
    )

    figure.tight_layout()

    buffer = io.BytesIO()

    figure.savefig(
        buffer,
        format="png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    buffer.seek(0)

    return buffer


# ============================================================
# SAMPLE DICOM FILES
# ============================================================

def get_sample_files():
    """
    Return available DICOM files from the samples folder.
    """

    if not os.path.isdir(
        SAMPLES_DIR
    ):
        return []

    return sorted(
        [
            filename
            for filename in os.listdir(
                SAMPLES_DIR
            )
            if filename.lower().endswith(
                ".dcm"
            )
        ]
    )


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.title(
        "Diabetic Retinopathy Detection"
    )

    st.write(
        "Upload a retinal DICOM image or select a sample "
        "image to estimate diabetic retinopathy severity "
        "and visualize the image regions contributing to "
        "the model prediction."
    )

    st.warning(
        "Research and decision-support demonstration only. "
        "This model is not clinically validated and should "
        "not be used as a standalone medical diagnosis."
    )

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.header(
            "Model Information"
        )

        st.write(
            "Model file:"
        )

        st.code(
            MODEL_PATH
        )

        st.write(
            "Input:"
        )

        st.write(
            "224 x 224 RGB"
        )

        st.write(
            "Output:"
        )

        st.write(
            "5 DR classes"
        )

        st.write(
            "XAI:"
        )

        st.write(
            "Input Gradient Saliency"
        )

        st.divider()

        st.subheader(
            "Severity Classes"
        )

        for index, class_name in enumerate(
            CLASS_NAMES
        ):

            st.write(
                f"{index}: {class_name}"
            )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:

        model = load_model()

    except Exception as error:

        st.error(
            "Unable to load final_model.keras."
        )

        st.code(
            str(error)
        )

        st.info(
            "Make sure final_model.keras is located "
            "in the same directory as drd.py."
        )

        st.stop()

    # ========================================================
    # VALIDATE MODEL
    # ========================================================

    try:

        output_shape = tuple(
            model.output.shape
        )

        if output_shape[-1] != 5:

            st.error(
                "The loaded model does not appear to be "
                "a five-class classifier."
            )

            st.write(
                f"Detected output shape: {output_shape}"
            )

            st.stop()

    except Exception:
        pass

    # ========================================================
    # INPUT SOURCE
    # ========================================================

    st.subheader(
        "Select Retinal Image"
    )

    input_mode = st.radio(
        "Choose image source:",
        [
            "Upload DICOM",
            "Use Sample DICOM"
        ],
        horizontal=True
    )

    uploaded_file = None
    selected_sample = None
    sample_path = None

    # ========================================================
    # UPLOAD MODE
    # ========================================================

    if input_mode == "Upload DICOM":

        uploaded_file = st.file_uploader(
            "Upload retinal DICOM file",
            type=["dcm"],
            accept_multiple_files=False
        )

        if uploaded_file is None:

            st.info(
                "Upload a .dcm retinal image to begin."
            )

            st.stop()

    # ========================================================
    # SAMPLE MODE
    # ========================================================

    else:

        sample_files = get_sample_files()

        if not sample_files:

            st.warning(
                "No DICOM files were found in the "
                "'samples' folder."
            )

            st.info(
                "Place one or more .dcm files inside "
                "the samples folder."
            )

            st.stop()

        selected_sample = st.selectbox(
            "Select a sample DICOM:",
            sample_files
        )

        sample_path = os.path.join(
            SAMPLES_DIR,
            selected_sample
        )

        st.caption(
            f"Selected sample: {selected_sample}"
        )

    # ========================================================
    # READ DICOM
    # ========================================================

    try:

        if input_mode == "Upload DICOM":

            image, dicom_dataset = (
                read_dicom_file(
                    uploaded_file
                )
            )

        else:

            with open(
                sample_path,
                "rb"
            ) as sample_file:

                sample_bytes = (
                    sample_file.read()
                )

            image, dicom_dataset = (
                read_dicom_bytes(
                    sample_bytes
                )
            )

    except Exception as error:

        st.error(
            "Unable to read the DICOM file."
        )

        st.code(
            str(error)
        )

        st.stop()

    # ========================================================
    # DICOM INFORMATION
    # ========================================================

    modality = getattr(
        dicom_dataset,
        "Modality",
        "N/A"
    )

    photometric = getattr(
        dicom_dataset,
        "PhotometricInterpretation",
        "N/A"
    )

    # ========================================================
    # DISPLAY INPUT
    # ========================================================

    st.subheader(
        "Input Retinal Image"
    )

    image_column, metadata_column = st.columns(
        [2, 1]
    )

    with image_column:

        st.image(
            image,
            caption="Retinal DICOM image",
            use_container_width=True
        )

    with metadata_column:

        st.subheader(
            "Image Information"
        )

        st.write(
            f"Image size: "
            f"{image.width} x {image.height}"
        )

        st.write(
            f"Modality: {modality}"
        )

        st.write(
            f"Photometric: {photometric}"
        )

        st.caption(
            "Patient-identifying DICOM metadata is "
            "not displayed."
        )

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "Analyze Retinal Image",
        type="primary",
        use_container_width=True
    )

    if not analyze:
        st.stop()

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    with st.spinner(
        "Running diabetic retinopathy analysis..."
    ):

        try:

            (
                predicted_index,
                confidence,
                probabilities
            ) = predict(
                model,
                image
            )

        except Exception as error:

            st.error(
                "Model inference failed."
            )

            st.code(
                str(error)
            )

            st.stop()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "Prediction Result"
    )

    result_column, confidence_column = st.columns(
        2
    )

    with result_column:

        st.metric(
            "Predicted Severity",
            predicted_class
        )

    with confidence_column:

        st.metric(
            "Model Confidence",
            f"{confidence:.2%}"
        )

    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    st.subheader(
        "Class Probability Distribution"
    )

    for index, probability in enumerate(
        probabilities
    ):

        probability = float(
            probability
        )

        st.write(
            f"{CLASS_NAMES[index]}: "
            f"{probability:.2%}"
        )

        st.progress(
            min(
                max(
                    probability,
                    0.0
                ),
                1.0
            )
        )

    # ========================================================
    # XAI
    # ========================================================

    st.divider()

    st.subheader(
        "Explainable AI - Input Gradient Saliency"
    )

    st.write(
        "The saliency map highlights retinal image "
        "regions where changes in the input pixels have "
        "the greatest effect on the selected prediction."
    )

    try:

        with st.spinner(
            "Generating XAI heatmap..."
        ):

            (
                heatmap,
                xai_method
            ) = generate_xai_heatmap(
                model=model,
                image=image,
                class_index=predicted_index
            )

            overlay = create_heatmap_overlay(
                image=image,
                heatmap=heatmap
            )

    except Exception as error:

        st.error(
            "Prediction succeeded, but the XAI "
            "visualization could not be generated."
        )

        st.code(
            str(error)
        )

        st.info(
            "The model prediction remains available, "
            "but an input-gradient explanation could "
            "not be generated for this model."
        )

        st.stop()

    # ========================================================
    # XAI VISUALIZATION
    # ========================================================

    original_column, overlay_column = st.columns(
        2
    )

    with original_column:

        st.image(
            image,
            caption="Original Retinal Image",
            use_container_width=True
        )

    with overlay_column:

        st.image(
            overlay,
            caption="Input Gradient Saliency Overlay",
            use_container_width=True
        )

    st.caption(
        f"XAI method: {xai_method}"
    )

    # ========================================================
    # DOWNLOAD XAI REPORT
    # ========================================================

    report = create_xai_report(
        original=image,
        overlay=overlay,
        heatmap=heatmap,
        prediction=predicted_class,
        confidence=confidence
    )

    st.download_button(
        label="Download XAI Visualization",
        data=report,
        file_name="drd_xai_analysis.png",
        mime="image/png",
        use_container_width=True
    )

    # ========================================================
    # FINAL DISCLAIMER
    # ========================================================

    st.divider()

    st.info(
        "The XAI visualization explains model behavior "
        "and should not be interpreted as a clinically "
        "confirmed lesion map. Model predictions should "
        "not be used as a standalone medical diagnosis."
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()