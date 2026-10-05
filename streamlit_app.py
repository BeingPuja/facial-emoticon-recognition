"""
streamlit_app.py

Streamlit version of the Facial Emotion Recognition demo — built for
deployment to Streamlit Community Cloud (streamlit.io/cloud), which is
free for public apps and deploys directly from a GitHub repo.

Two modes, as two tabs:
1. "Live Webcam (Real-Time)" — uses streamlit-webrtc to continuously stream
   your browser's camera to the model and overlay the predicted emotion on
   each frame, live. This works both locally and once deployed.
2. "Upload a Photo" — simple single-image upload and prediction, as a
   reliable fallback if someone's browser/network struggles with the live
   WebRTC stream (e.g. during a live presentation on conference wifi).

Run locally with:
    streamlit run streamlit_app.py

For deployment instructions, see DEPLOY.md.
"""

# IMPORTANT: TensorFlow must be imported before OpenCV (cv2). On Linux,
# importing cv2 first can cause a native segmentation fault when TensorFlow
# is imported afterward, due to the two libraries bundling conflicting
# versions of low-level math/threading libraries. This doesn't show up on
# Windows, which is why it can pass locally but crash on Streamlit Cloud
# (which runs Linux) if the order is wrong.
import tensorflow as tf  # noqa: F401  (imported first deliberately, see above)
from tensorflow.keras.models import load_model

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from data_preprocessing import EMOTION_LABELS, apply_clahe

IMG_SIZE = 48
MODEL_PATH = "outputs/custom_cnn.keras"  # change to mobilenet_transfer.keras if preferred

st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


model = get_model()
face_cascade = get_face_cascade()

# STUN server config so WebRTC can establish a peer connection when deployed
# (needed for the browser's camera stream to reach the Streamlit server).
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def predict_on_bgr_frame(frame_bgr):
    """
    Runs face detection + emotion prediction on a single BGR frame (OpenCV's
    native format). Returns the annotated BGR frame and the probability dict
    (or None if no face found).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return frame_bgr, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = gray[y:y + h, x:x + w]
    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    face_clahe = apply_clahe(face_resized)
    face_input = face_clahe.astype("float32") / 255.0
    face_input = np.expand_dims(face_input, axis=(0, -1))

    preds = model.predict(face_input, verbose=0)[0]
    result = {EMOTION_LABELS[i]: float(preds[i]) for i in range(len(EMOTION_LABELS))}
    top_label = EMOTION_LABELS[int(np.argmax(preds))]

    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 200, 0), 3)
    cv2.putText(
        frame_bgr, top_label, (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2
    )
    return frame_bgr, result


class EmotionVideoProcessor(VideoProcessorBase):
    """
    Called automatically by streamlit-webrtc on every incoming video frame
    from the browser's webcam. This is what makes the live tab real-time.
    """
    def __init__(self):
        self.last_result = None

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        annotated, result = predict_on_bgr_frame(img_bgr)
        if result is not None:
            self.last_result = result
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


st.title("Facial Emotion Recognition using CNN")
st.caption(
    "Detects a face and classifies its expression into one of 7 emotions: "
    "Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. "
    "Built with a custom CNN trained on FER-2013, CLAHE preprocessing, "
    "and Haar Cascade face detection."
)

tab_live, tab_upload = st.tabs(["Live Webcam (Real-Time)", "Upload a Photo"])

with tab_live:
    st.write("Allow camera access, then look at the camera — the box and "
             "label update continuously as your expression changes.")
    webrtc_streamer(
        key="emotion-live",
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with tab_upload:
    st.write("Use this if the live tab is slow on your connection, or you "
             "just want to test a single photo.")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated, result = predict_on_bgr_frame(img_bgr.copy())

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected face")
        with col2:
            if result is None:
                st.warning("No face detected — try a clearer, front-facing photo.")
            else:
                st.subheader(f"Predicted: {max(result, key=result.get)}")
                st.bar_chart(result)
