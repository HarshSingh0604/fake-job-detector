# Deploy for free on Render

The project runs as two free Render web services, both built from the same GitHub repo:

| Service | What it runs | Built from |
|---|---|---|
| `fake-job-api` | FastAPI model API | `Dockerfile` |
| `fake-job-ui` | Streamlit page, which calls the API over HTTPS | `requirements-ui.txt` |

The model file is not in git, because `models/` is gitignored. You publish it as a **GitHub release asset**, and the Docker build downloads it from the URL in `MODEL_URL`.

## Free tier facts (from Render's docs, November 2026)
- Each free web service gets **512 MB RAM and 0.1 CPU**. The API used 352–377 MB in local tests.
- A free service **spins down after 15 minutes** with no traffic, and the next request takes **about a minute** to wake it. If both services are asleep, the first visit can take about 2 minutes.
- Each workspace gets **750 free instance hours a month**, shared by both services. That's enough for a demo that sleeps when idle.
- There is no persistent disk. That doesn't matter here, because the model is built into the image.
- Free services can't receive private-network traffic, so the page calls the API's **public** URL.

## 0. Before you start
- Push the code to a **public** GitHub repo. The release download has to work without logging in.
- Create a Render account at https://render.com (you can sign up with GitHub).
- Have the trained model at `models/fraud_detector.joblib`. If it's missing, run `python -m src.train`.

## 1. Publish the model as a GitHub release
With the GitHub CLI, from the project root:
```bash
gh release create model-v1 models/fraud_detector.joblib --title "Model v1" \
  --notes "RandomForest + SMOTE, decision threshold 0.274"
```
Or on github.com: open the repo, go to **Releases → Draft a new release**, set the tag to `model-v1`, attach `models/fraud_detector.joblib`, and click **Publish**.

Your model URL will be:
```
https://github.com/<your-user>/<your-repo>/releases/download/model-v1/fraud_detector.joblib
```
Check that it downloads. This should print about 18,083,016 bytes:
```bash
curl -sL -o /dev/null -w "%{size_download} bytes\n" <MODEL_URL>
```

## 2. Deploy the API (Docker)
1. In the Render dashboard, click **New → Web Service**, connect GitHub and choose the repo.
2. Fill in the form:
   - **Name:** `fake-job-api`
   - **Language:** **Docker**
   - **Branch:** `main`
   - **Instance type:** **Free**
3. Under **Environment Variables**, add `MODEL_URL` with the URL from step 1. Render passes environment variables to `docker build` as build arguments, which is how the Dockerfile's `ARG MODEL_URL` gets it.
4. Click **Create Web Service**. In the build log, look for: `Model OK: RandomForest smote threshold 0.274`
5. Once it's live, go to **Settings → Health Checks** and set the path to `/health`.
6. Test it:
   ```bash
   curl https://<api-name>.onrender.com/health
   curl -X POST https://<api-name>.onrender.com/predict \
     -H "Content-Type: application/json" -d @app/examples/fake_posting.json
   ```
   The interactive docs are at `https://<api-name>.onrender.com/docs`.

## 3. Deploy the Streamlit page
1. Click **New → Web Service** and choose the same repo.
2. Fill in the form:
   - **Name:** `fake-job-ui`
   - **Language:** **Python 3**. Render may pre-select Docker because the repo has a Dockerfile, so change it.
   - **Instance type:** **Free**
   - **Build command:** `pip install -r requirements-ui.txt`
   - **Start command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
3. Under **Environment Variables**, add `API_URL` set to `https://<api-name>.onrender.com`. Use the exact URL Render showed for the API.
4. You don't need to set a Python version. Render reads it from the `.python-version` file (3.11) in the repo root.
5. Click **Create Web Service**. Then open the page, click **Try a fake-looking example**, and click **Check posting**.

Put both live links in the README.

## Updating
- **Code changes:** push to GitHub. By default, Render redeploys both services.
- **New model:**
  1. Retrain.
  2. Upload the file as a new release, for example `model-v2`.
  3. Update `MODEL_URL` on the API service. The service rebuilds.

## Troubleshooting
| Symptom | Cause and fix |
|---|---|
| Build log says `No model: run 'python -m src.train' first or pass --build-arg MODEL_URL=<url>` | `MODEL_URL` isn't set on the API service. |
| Build fails with `HTTP Error 404` | The URL is wrong or the repo is private. Re-run the curl check from step 1. |
| The page says "Can't reach the model API" | The API is asleep: wait a minute and refresh. Otherwise `API_URL` is wrong: open `<API_URL>/health` in a browser to check. |
| The API restarts with an out-of-memory error | The limit is 512 MB and the API uses about 375 MB. Don't add `--workers 2`, which loads a second copy of the model. |

## Why not Hugging Face Spaces?
Hugging Face's docs (checked November 2026) say: *"Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts."* Streamlit is no longer a native Space type: the `sdk` setting only accepts `gradio`, `docker` or `static`. So this stack can't be hosted there for free.

With a PRO account it does work:
1. Create a Docker Space.
2. Add `sdk: docker` and `app_port: 8000` to the YAML header of the Space's README.
3. Set `MODEL_URL` as a Space variable. Spaces pass variables to the build as build arguments.
4. Upload the code with `hf upload`.

## Run the container locally
This uses your local `models/fraud_detector.joblib`:
```bash
docker build -t fake-job-detector .
docker run --rm -p 8000:8000 fake-job-detector      # API on http://localhost:8000/docs
streamlit run streamlit_app.py                      # page on http://localhost:8501, calls the container
```
