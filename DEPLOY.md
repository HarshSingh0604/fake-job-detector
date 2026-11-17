# Deploy for free on Render

[`render.yaml`](render.yaml) is a Render **Blueprint**. It creates both services in one go from this repo:

| Service | What it runs | Built from | Expected URL |
|---|---|---|---|
| `fake-job-detector-api` | FastAPI model API | `Dockerfile` | `https://fake-job-detector-api.onrender.com` |
| `fake-job-detector` | Streamlit page, which calls the API over HTTPS | `requirements-ui.txt` | `https://fake-job-detector.onrender.com` |

**Where the model comes from:** the model isn't in git. It's published as the [v1.0 release](https://github.com/HarshSingh0604/fake-job-detector/releases/tag/v1.0) (`fraud_detector.joblib`, 18.1 MB). With no local model, the Docker build downloads it automatically: that URL is the `MODEL_URL` default in the Dockerfile. You don't need to set anything for it.

**About the expected URLs:** Render bases a service's `onrender.com` address on its name, but adds a suffix if the name is already taken. So check the real URLs after the first deploy (step 6).

## Free tier facts (from Render's docs, November 2026)
- Each free web service gets **512 MB RAM and 0.1 CPU**. The API used 352–377 MB in local tests.
- A free service **spins down after 15 minutes** with no traffic, and the next request takes **about a minute** to wake it. If both services are asleep, the first visit can take about 2 minutes.
- Each workspace gets **750 free instance hours a month**, shared by both services. Services only use hours while awake, so a demo that's idle most of the time stays within it.
- Free services can't receive private-network traffic, so the page calls the API's **public** URL. Render can't fill that URL in automatically, which is why step 4 asks you for it.

## Deploy with the Blueprint (exact clicks)
1. Go to https://dashboard.render.com and sign in. If you don't have an account, choose **GitHub** to sign up.
2. Click **New** (top right), then **Blueprint**.
3. Find **`HarshSingh0604/fake-job-detector`** in the repo list and click **Connect** next to it.
   If it isn't listed, click **Configure account** / **Connect GitHub**, choose **Only select repositories**, pick `fake-job-detector`, approve, and come back.
4. Fill in the form:
   - **Blueprint Name:** `fake-job-detector`
   - **Branch:** `main`
   - **Blueprint Path:** leave as the default (`render.yaml`)
   - Render lists the 2 services it will create, `fake-job-detector-api` and `fake-job-detector`, both on the **Free** plan.
   - It asks for **`API_URL`**. Enter exactly: `https://fake-job-detector-api.onrender.com`
5. Click **Deploy Blueprint** and wait for both services to finish their first deploy. The API's Docker build takes the longest.
   - In the API's **Logs**, a good build shows:
     ```
     Downloading model from https://github.com/HarshSingh0604/fake-job-detector/releases/download/v1.0/fraud_detector.joblib
     Model OK: RandomForest smote threshold 0.274
     ```
   - After that, the API logs `Application startup complete`.
6. **Check the API's real URL.** Open the `fake-job-detector-api` service; its URL is shown under the name at the top.
   - If it's exactly `https://fake-job-detector-api.onrender.com`, go to step 7.
   - If it's different (for example with a suffix like `-abcd`), fix the page's setting:
     1. Open the **`fake-job-detector`** service and go to **Environment**.
     2. Edit **`API_URL`** to the real API URL, with no trailing slash.
     3. Save, and choose the option that **rebuilds and deploys**.
7. Test both services:
   - **API:** open `<API URL>/health`. It should show `{"status":"ok","model":"RandomForest",...}`. Then open `<API URL>/docs`.
   - **Page:** open the `fake-job-detector` service's URL, click **Try a fake-looking example**, then **Check posting**. You should see **"Likely fake"** with risk score **0.812**.
   - The first page load can take a minute or two while the services wake up.
8. Send me both URLs and I'll put them in the README.

## Updating
- **Code changes:** push to `main`. Both services redeploy automatically.
- **A new model:**
  1. Retrain.
  2. Publish the file as a new release:
     ```bash
     gh release create v2.0 models/fraud_detector.joblib --title "v2.0: trained model" --notes "..."
     ```
  3. On the API service, go to **Environment** and add `MODEL_URL` = `https://github.com/HarshSingh0604/fake-job-detector/releases/download/v2.0/fraud_detector.joblib`. Render passes environment variables to `docker build` as build arguments, so this overrides the Dockerfile default.
  4. Save and redeploy.
- **Changing `API_URL` later:** Blueprint syncs ignore `sync: false` variables, so change it only in the dashboard (step 6).

## Troubleshooting
| Symptom | Cause and fix |
|---|---|
| The page says "Can't reach the model API" | The API is asleep: wait a minute and refresh. Otherwise `API_URL` is wrong: redo step 6. |
| API build fails at `Downloading model` with `HTTP Error 404` | The release file is missing, renamed, or the repo isn't public. Open the URL from the log in a browser. |
| API restarts with an out-of-memory error | The limit is 512 MB and the API uses about 375 MB. Don't add `--workers 2`, which loads a second copy of the model. |
| The page's build installs Python 3.14 | `.python-version` (3.11) must be in the repo root. Or set `PYTHON_VERSION` on that service to a full 3.11 version, including the patch number. |

## Without the Blueprint
You can also create the two services by hand: **New → Web Service**, using the same settings as in `render.yaml`.
- **API:** runtime **Docker**, plan **Free**, health check path `/health`.
- **Page:** runtime **Python 3**, plan **Free**, the build and start commands from `render.yaml`, and the `API_URL` variable.

## Why not Hugging Face Spaces?
Hugging Face's docs (checked November 2026) say: *"Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts."* Streamlit is no longer a native Space type: the `sdk` setting only accepts `gradio`, `docker` or `static`. So this stack can't be hosted there for free.

## Run the container locally
```bash
docker build -t fake-job-detector .                 # uses models/fraud_detector.joblib if present, else downloads v1.0
docker run --rm -p 8000:8000 fake-job-detector      # API on http://localhost:8000/docs
streamlit run streamlit_app.py                      # page on http://localhost:8501, calls the container
```
