Complete Guide: Real-Time Demo + Public Deployment
This guide covers how to turn a trained Facial Emotion Recognition model into a working web application that can run locally and be shared through a public URL.
The workflow is:
Trained Model → Streamlit App → Local Testing → GitHub → Public Deployment
What you'll end up with
streamlit_app.py
A Streamlit web application with two tabs:
Live Webcam (Real-Time) — captures webcam frames and displays the predicted emotion.
Upload a Photo — lets users upload an image and run emotion prediction.
The photo-upload option is a useful fallback if webcam access or network conditions cause problems.
Two ways to run the application
Locally: useful for development, testing, and demonstrations.
Streamlit Community Cloud: useful for creating a public URL to share.
Part 1 — Prerequisite: a trained model
Before starting, make sure the trained model is available at:
outputs/custom_cnn.keras
A typical project structure might look like this:
fer_project/
├── outputs/
│   └── custom_cnn.keras
├── streamlit_app.py
├── requirements.txt
├── requirements_training.txt
├── README.md
└── ...
Your actual project structure may differ.
Part 2 — Run the application locally
It is best to get the application working locally before deploying it.
1. Open the project folder
Open a terminal in your fer_project directory:
cd path/to/fer_project
2. Install the dependencies
If you have a separate requirements file for the Streamlit app:
pip install -r requirements_streamlit.txt
If your project uses a single requirements file, use:
pip install -r requirements.txt
Avoid installing multiple requirements files that specify conflicting versions of the same packages.
3. Check the model file
Confirm that this file exists:
outputs/custom_cnn.keras
4. Start Streamlit
streamlit run streamlit_app.py
Streamlit will provide a local URL, usually:
http://localhost:8501
Open the URL in your browser.
5. Test the webcam
Open Live Webcam (Real-Time), allow camera access, and start the webcam.
The application should capture frames, process the face image, run the trained CNN, and display the predicted emotion. Try changing your expression and see how the prediction responds.
6. Test image upload
Open Upload a Photo, upload a face image, and confirm that the model produces a prediction.
Testing both tabs gives you a fallback if webcam access is unavailable.
Part 3 — Prepare the GitHub repository
Streamlit Community Cloud can deploy an application directly from a GitHub repository. GitHub also provides a backup of your source code and deployment files.
1. Initialize Git
From the project directory:
git init
2. Add a .gitignore
Before committing, make sure unnecessary or sensitive files are excluded. For example:
.venv/
venv/
__pycache__/
*.pyc
.env
.ipynb_checkpoints/
Do not commit passwords, API keys, .env files containing secrets, personal files, or unnecessary temporary files.
3. Add and commit the project
git add .
git commit -m "Add facial emotion recognition project"
4. Create a GitHub repository
Create a repository on GitHub, for example:
facial-emotion-recognition
Make it public if you want others to view the source code and use it with a public Streamlit deployment.
5. Connect and push the repository
Use the repository URL shown by GitHub. The commands will look similar to:
git remote add origin https://github.com/yourusername/your-repository.git
git branch -M main
git push -u origin main
Replace the example URL with your own repository URL.
Important — model file size
The trained model must be available to the deployed application. Check the size of:
outputs/custom_cnn.keras
GitHub restricts the size of files that can be pushed normally. If the model is within the normal file-size limit, you can commit it directly:
git add outputs/custom_cnn.keras
git commit -m "Add trained CNN model"
git push
If GitHub rejects the file because it is too large, consider Git LFS or a suitable external model-storage service.
Basic Git LFS setup:
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add outputs/custom_cnn.keras
git commit -m "Track model with Git LFS"
git push
If you encounter a size-related error, check the exact error message before changing the repository.
Part 4 — Prepare deployment requirements
Streamlit Community Cloud looks for a dependency file named exactly:
requirements.txt
It should list the packages needed to run the web application. For example:
streamlit
tensorflow
opencv-python-headless
numpy
pillow
Use package versions that are compatible with the environment in which the application was tested.
Keep training and deployment dependencies separate
If your current requirements.txt was created for model training, consider keeping the two dependency lists separate:
requirements.txt — packages needed to run the Streamlit app.
requirements_training.txt — packages used for training and experimentation.
If requirements_streamlit.txt already contains the correct deployment dependencies, rename it to requirements.txt and rename the old training file to requirements_training.txt.
After making the changes:
git add .
git commit -m "Prepare deployment requirements"
git push
Part 5 — Deploy to Streamlit Community Cloud
Once the app works locally and the repository is ready:
Open Streamlit Community Cloud.
Sign in with your GitHub account.
Create a new app.
Select your GitHub repository.
Select the main branch.
Set the main file path to:
streamlit_app.py
Start the deployment.
Streamlit will create an environment and install the packages listed in requirements.txt. The first deployment may take a while, especially when packages such as TensorFlow need to be installed.
After deployment, Streamlit will provide a public URL similar to:
https://your-app-name.streamlit.app
Open the URL and test the application before sharing it.
Part 6 — Test the public application
Check that:
The page loads correctly.
The model loads successfully.
Upload a Photo produces predictions.
Live Webcam (Real-Time) connects and produces predictions.
No visible error messages appear.
The application works from another browser or device, if possible.
If these checks pass, you can share the public URL.
Part 7 — Troubleshoot webcam and WebRTC issues
Webcam applications can be affected by browser permissions and network conditions.
Possible causes include:
Camera permissions are blocked.
Browser restrictions.
Firewall or proxy settings.
Network restrictions.
WebRTC connectivity problems.
University or office network policies.
If the webcam does not connect:
Check the browser's camera permission.
Try another browser.
Try another network, such as a mobile hotspot.
Use the Upload a Photo tab as a fallback.
Keep the local version working even after the public app has been deployed.
Part 8 — Keep a local backup
Do not rely only on the hosted application. Keep the source code, model, and dependency files on your computer and in your repository.
Before a demonstration:
Start the app locally.
Confirm that the model loads.
Test the webcam.
Test image upload.
Keep the public URL available as another option.
A short screen recording of the working application can also be useful if someone cannot access the live demo.
Part 9 — Share the project
You can share:
Live demo: your Streamlit application URL.
Source code: your GitHub repository URL.
Demo video: a short 10–15 second recording showing the webcam prediction updating.
Results: your confusion matrix or another evaluation result, if useful.
A working demo alongside the source code helps people see both the result and how the project was built.
Common issues
ModuleNotFoundError
Check that the missing package is included in the requirements.txt used by Streamlit Community Cloud. Commit and push any changes, then restart or redeploy the app as needed.
Model file cannot be found
Check that the path in your code matches the repository structure. For example:
model = tf.keras.models.load_model(
    "outputs/custom_cnn.keras"
)
The directory and filename must match exactly.
Works locally but fails after deployment
This can happen when the local and hosted environments use different package versions. Check the Python, TensorFlow, NumPy, and OpenCV versions, and pin compatible versions in requirements.txt if needed.
Deployment takes a long time
The first deployment can take time while dependencies are installed. Startup time and availability can vary with the hosting environment.
GitHub rejects the model file
Check the file size of outputs/custom_cnn.keras. If it exceeds GitHub's normal file-size limits, use Git LFS or an external model-storage solution.
Recommended project structure
A clean repository could look like this:
facial-emotion-recognition/
├── outputs/
│   └── custom_cnn.keras
├── streamlit_app.py
├── requirements.txt
├── requirements_training.txt
├── README.md
├── .gitignore
└── ...
Adjust this structure to match the rest of your project.
Final checklist
[ ]
Model trained successfully.
[ ]
custom_cnn.keras is available.
[ ]
Streamlit app works locally.
[ ]
Webcam prediction works.
[ ]
Photo-upload prediction works.
[ ]
GitHub repository is created.
[ ]
.gitignore is configured.
[ ]
Deployment requirements are prepared.
[ ]
The model is available to the deployed app.
[ ]
Streamlit deployment succeeds.
[ ]
Public URL is tested.
[ ]
GitHub repository is accessible.
[ ]
A short demo recording is ready, if desired.
Project workflow
FER-2013 Dataset
↓
Image Preprocessing
↓
CNN Training
↓
Model Evaluation
↓
Grad-CAM / Model Interpretation
↓
Streamlit Application
↓
Real-Time Webcam Prediction
↓
GitHub
↓
Public Deployment
