Facial Emotion Recognition

A real-time Facial Emotion Recognition system built using deep learning and computer vision.

The project uses the FER-2013 dataset to classify facial expressions into seven emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The trained model is also connected to a Streamlit web application that supports both webcam-based prediction and image upload.

---

Demo

Live Demo

Streamlit App:
https://facial-emoticon-recognition-5fyercdfcffrjmelgvjcjq.streamlit.app/

Source Code

GitHub:
https://github.com/BeingPuja/facial-emoticon-recognition

---

Project Overview

The main goal of this project was to build and test a facial emotion recognition system, from training the model to running it as an interactive web application.

The project includes:

- A custom CNN trained from scratch
- Transfer learning using MobileNetV2
- Image preprocessing with CLAHE
- Model evaluation using accuracy, classification reports, and confusion matrices
- Grad-CAM for visualizing model attention
- Real-time webcam prediction
- Image upload prediction
- Streamlit deployment

---

Models

1. Custom CNN

A convolutional neural network was designed and trained from scratch using the FER-2013 dataset.

Test Accuracy: "57.8%"

2. MobileNetV2

A pretrained MobileNetV2 model with ImageNet weights was fine-tuned for the seven emotion classes.

Test Accuracy: "52.4%"

In my experiments, the custom CNN performed better than the MobileNetV2 model.

---

Dataset

The project uses the FER-2013 (Facial Expression Recognition 2013) dataset.

The dataset contains grayscale facial images divided into seven emotion classes:

Label| Emotion
0| Angry
1| Disgust
2| Fear
3| Happy
4| Sad
5| Surprise
6| Neutral

---

Preprocessing

CLAHE

Contrast Limited Adaptive Histogram Equalization (CLAHE) was used to improve local contrast in facial images.

This can help make facial features more visible, particularly when the original image has low or uneven contrast.

---

Model Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visualize which regions of an input image contributed to the model's prediction.

This provides a way to inspect whether the model is focusing on meaningful facial regions when making predictions.

---

Results

For the custom CNN:

- Test accuracy: 57.8%
- Best-performing emotion: Happy
- Happy F1-score: 82%
- Most difficult emotion: Fear
- Common confusion: Fear and Surprise

The results show that some expressions are easier for the model to distinguish than others.

---

Web Application

The trained model is connected to a Streamlit application with two main options.

Live Webcam

The application can access the webcam and perform emotion prediction on the live video stream.

The predicted emotion is updated as the facial expression changes.

Upload a Photo

A user can upload a photo containing a face and run the emotion prediction on the image.

This is also useful as a fallback when webcam access is unavailable.

---

Running Locally

1. Clone the repository

git clone https://github.com/yourusername/your-repository.git
cd your-repository

2. Install the dependencies

For the Streamlit application:

pip install -r requirements.txt

If you are working with the training environment as well, see:

requirements_training.txt

3. Make sure the trained model exists

The Streamlit application expects:

outputs/custom_cnn.keras

4. Start the application

streamlit run streamlit_app.py

The application will normally be available at:

http://localhost:8501

Open the URL in your browser and choose either the webcam or image-upload option.

---

Deployment

The application can be deployed using Streamlit Community Cloud.

The basic deployment flow is:

GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Public Streamlit URL

For the complete deployment instructions, see:

"deploy.md" (deploy.md)

---

Project Structure

A simplified project structure looks like this:

facial-emotion-recognition/
│
├── outputs/
│   └── custom_cnn.keras
│
├── streamlit_app.py
├── requirements.txt
├── requirements_training.txt
├── deploy.md
├── README.md
└── ...

The exact structure may vary depending on the training notebooks and other files included in the repository.

---

Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit
- MobileNetV2
- Grad-CAM
- FER-2013

---

What I Learned

This project gave me the opportunity to work through different parts of a computer vision project, from preprocessing and model training to evaluation, explainability, and deployment.

One of the interesting parts was comparing a custom CNN with a pretrained MobileNetV2 model. The custom CNN performed better in my experiments, which was a useful reminder that a more complex or pretrained model does not automatically produce better results.

I also learned that getting the model to work is only part of the process. Making the application reliable involved dealing with package versions, dependencies, webcam access, and deployment issues as well.

There is still a lot more to explore, but this project was a good hands-on learning experience with deep learning and computer vision.

---

Future Improvements

Some possible improvements include:

- Improve the model's overall accuracy
- Experiment with data augmentation
- Handle class imbalance more effectively
- Try different CNN architectures
- Tune MobileNetV2 and other transfer-learning models
- Improve face detection
- Add prediction confidence scores
- Improve real-time prediction stability
- Optimize the application for faster inference

---

Note

This project is intended as a learning and experimentation project. The predictions are model outputs and should not be treated as a reliable way of determining a person's actual emotional state.

---
