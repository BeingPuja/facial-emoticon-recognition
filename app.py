"""
app.py

A Gradio web demo for the Facial Emotion Recognition CNN. This is what gets
deployed to Hugging Face Spaces to give you a public, shareable link —
e.g. to put on LinkedIn or in your project report.

Two modes, as two tabs:
1. "Snapshot" — upload a photo or capture one still frame from your webcam,
   click Predict, see the result. Works everywhere, zero surprises.
2. "Live Webcam (Real-Time)" — continuously streams your webcam to the
   model and updates the predicted emotion live as your expression changes,
   using Gradio's streaming image input. This is the "real-time" demo, and
   it works both locally AND once deployed to a public Hugging Face Space
   link, so the same file covers your viva demo and your LinkedIn demo.

Run locally with:
    python app.py
Then open the local URL it prints (usually http://127.0.0.1:7860).

For deployment instructions, see DEPLOY.md.
"""

import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

from data_preprocessing import EMOTION_LABELS, apply_clahe

IMG_SIZE = 48
MODEL_PATH = "outputs/custom_cnn.keras"  # change to mobilenet_transfer.keras if preferred

print(f"Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def predict_emotion(image):
    """
    Core inference function, shared by both the snapshot and live-streaming
    tabs below.

    image: numpy array (H, W, 3) in RGB, as provided by Gradio's image input.
    Returns: (label_dict_for_gr_Label, annotated_image_with_box_and_text)
    """
    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    annotated = image.copy()

    if len(faces) == 0:
        return {"No face detected": 1.0}, annotated

    # Use the largest detected face (most likely the main subject)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face_crop = gray[y:y + h, x:x + w]
    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    face_clahe = apply_clahe(face_resized)
    face_input = face_clahe.astype("float32") / 255.0
    face_input = np.expand_dims(face_input, axis=(0, -1))

    preds = model.predict(face_input, verbose=0)[0]
    result = {EMOTION_LABELS[i]: float(preds[i]) for i in range(len(EMOTION_LABELS))}

    top_label = EMOTION_LABELS[int(np.argmax(preds))]
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 3)
    cv2.putText(
        annotated, top_label, (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2
    )

    return result, annotated


with gr.Blocks(title="Facial Emotion Recognition — CNN Demo") as demo:
    gr.Markdown(
        """
        # Facial Emotion Recognition using CNN
        Detects a face and classifies its expression into one of 7 emotions:
        **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.
        Built with a custom CNN trained on FER-2013, with CLAHE-based
        preprocessing and Haar Cascade face detection.
        """
    )

    with gr.Tabs():
        with gr.Tab("Live Webcam (Real-Time)"):
            gr.Markdown(
                "Allow camera access, then just look at the camera — the "
                "prediction updates continuously as your expression changes."
            )
            with gr.Row():
                with gr.Column():
                    live_input = gr.Image(
                        label="Live camera", sources=["webcam"], streaming=True
                    )
                with gr.Column():
                    live_annotated = gr.Image(label="Detected face")
                    live_label = gr.Label(label="Predicted emotion", num_top_classes=7)

            # Note: newer Gradio versions (>=4.x) support a `stream_every`
            # argument here to throttle how often inference runs (e.g.
            # stream_every=0.5 for twice a second). It's left out for
            # compatibility with older installs; if your installed Gradio
            # supports it and predictions feel too rapid/laggy, add
            # `stream_every=0.5` as an argument to .stream() below.
            live_input.stream(
                fn=predict_emotion, inputs=live_input,
                outputs=[live_label, live_annotated],
            )

        with gr.Tab("Snapshot (Upload or Single Capture)"):
            gr.Markdown(
                "Use this if the live tab is slow on your connection, or "
                "you just want to test a single photo."
            )
            with gr.Row():
                with gr.Column():
                    snap_input = gr.Image(
                        label="Upload or capture a photo", sources=["upload", "webcam"]
                    )
                    submit_btn = gr.Button("Predict Emotion", variant="primary")
                with gr.Column():
                    snap_annotated = gr.Image(label="Detected face")
                    snap_label = gr.Label(label="Predicted emotion", num_top_classes=7)

            submit_btn.click(
                fn=predict_emotion, inputs=snap_input,
                outputs=[snap_label, snap_annotated]
            )

if __name__ == "__main__":
    demo.launch()
