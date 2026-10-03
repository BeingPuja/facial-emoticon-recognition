"""
webcam_demo.py

Real-time facial emotion detection using your webcam. Uses OpenCV's Haar
Cascade for face detection, then feeds each detected face crop through your
trained CNN. Good for a live demo during presentation.

Usage:
    python webcam_demo.py --model outputs/custom_cnn.keras

Press 'q' to quit.
"""

import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from data_preprocessing import EMOTION_LABELS, apply_clahe

IMG_SIZE = 48


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    model = load_model(args.model)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/index.")

    print("Starting webcam demo. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
            face_clahe = apply_clahe(face_resized)
            face_input = face_clahe.astype("float32") / 255.0
            face_input = np.expand_dims(face_input, axis=(0, -1))

            preds = model.predict(face_input, verbose=0)[0]
            emotion_idx = int(np.argmax(preds))
            emotion_label = EMOTION_LABELS[emotion_idx]
            confidence = preds[emotion_idx]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{emotion_label} ({confidence:.2f})",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        cv2.imshow("Facial Emotion Recognition Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
