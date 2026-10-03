# Complete Guide: Real-Time Demo + Public Deployment for LinkedIn

**Update:** Hugging Face Spaces now requires a paid plan for anything with a
Python backend (Gradio/Docker) — free accounts only get Static (HTML-only)
Spaces, which can't run our model. So this guide uses **Streamlit Community
Cloud** instead, which is still free for public apps and deploys straight
from a GitHub repo.

This covers everything from "I have a trained model" to "I have a public
link on LinkedIn that anyone can open and see my face's emotion detected
live," plus showing it live, in person, for your viva.

---

## What you'll end up with

1. **`streamlit_app.py`** — a Streamlit web app with two tabs:
   - **"Live Webcam (Real-Time)"** — continuously streams your camera via
     WebRTC and overlays the predicted emotion live, updating as your
     expression changes.
   - **"Upload a Photo"** — single-image fallback, useful on unreliable
     connections (including during a live presentation).
2. **The same app running in two places:**
   - **Locally** — for your viva demo, full control, no internet needed.
   - **Deployed on Streamlit Community Cloud** — a free public URL for
     LinkedIn.

---

## Part 1 — Prerequisite: a trained model

You need `outputs/custom_cnn.keras` on your computer (from the Colab
training steps covered earlier) before any of this works.

---

## Part 2 — Run the real-time demo locally (for your viva)

1. Open a terminal in your `fer_project` folder.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install -r requirements_streamlit.txt
   ```
3. Confirm `outputs/custom_cnn.keras` exists in that folder.
4. Run:
   ```
   streamlit run streamlit_app.py
   ```
5. It opens automatically in your browser (usually `http://localhost:8501`).
6. Click **"Live Webcam (Real-Time)"**, click **Start**, allow camera
   access — the box and predicted emotion should update continuously as
   you change expression. This is what you demo live in your viva.

---

## Part 3 — Push your project to GitHub

Streamlit Community Cloud deploys directly from a GitHub repo, so this
step is required (and doubles as the code backup you wanted anyway).

1. In your `fer_project` folder:
   ```
   git init
   git add .
   git commit -m "Facial emotion recognition CNN project"
   ```
2. Create a new repository on github.com (public, so Streamlit Cloud's free
   tier can deploy it).
3. Follow GitHub's shown instructions to connect and push, typically:
   ```
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

**Important — model file size:** your `outputs/custom_cnn.keras` file needs
to be pushed to GitHub too, since Streamlit Cloud reads the repo as-is.
Plain `git add`/`push` handles files up to 100MB fine, and a compact custom
CNN like this one is normally well under that. If `git push` complains
about a large file, tell me the exact size and error and I'll walk you
through Git LFS instead.

---

## Part 4 — Deploy to Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **New app** (or **"Create app"**).
3. Choose:
   - Repository: your GitHub repo from Part 3
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Before deploying, Streamlit Cloud needs to know your dependencies. It
   looks for a file named exactly `requirements.txt` in your repo — but
   your repo already has one (for the training pipeline, with full
   `tensorflow`). To avoid conflicts:
   - Rename the training-time `requirements.txt` in your **deployed repo**
     to something else (e.g. `requirements_training.txt`), and
   - Rename `requirements_streamlit.txt` to `requirements.txt`
   - Commit and push this change before deploying (or right after, then
     click "Reboot app")
5. Click **Deploy**. The first build takes a few minutes (installing
   TensorFlow, OpenCV, etc.).
6. Once running, you'll get a public URL like
   `https://your-app-name.streamlit.app` — open it yourself first and test
   both tabs.

---

## Part 5 — Share on LinkedIn

- Post the Streamlit Cloud link (try it live) and your GitHub repo link
  (see the code)
- A short screen recording (10-15 seconds) of the Live Webcam tab updating
  as your expression changes is much more engaging than a screenshot
- Briefly mention what you built and for what course

---

## Common issues

- **"ModuleNotFoundError" on deploy**: double check you renamed
  `requirements_streamlit.txt` to `requirements.txt` in the repo Streamlit
  Cloud is reading, and that the old training `requirements.txt` was
  renamed out of the way.
- **Live tab doesn't connect / spins forever**: WebRTC sometimes struggles
  behind restrictive networks/firewalls (common on university or office
  wifi). Try a different network (mobile hotspot often works), or fall
  back to the "Upload a Photo" tab for your live presentation.
- **App goes to sleep**: Streamlit Community Cloud puts free apps to sleep
  after about 12 hours of no visits. Just open the link yourself a few
  minutes before your viva/demo to wake it up, or note in your LinkedIn
  post "may take a few seconds to wake up."
- **Model file too large for GitHub**: use Git LFS (`git lfs install`,
  `git lfs track "*.keras"`) — ask me if you hit this and I'll walk you
  through it step by step.
- **Prefer not to deal with any of this free-tier juggling?** Hugging
  Face's PRO plan (a few dollars/month) removes the Gradio-Spaces
  restriction entirely, if that's an acceptable option for you — but it's
  not required for this project.
