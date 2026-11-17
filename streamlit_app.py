"""Streamlit page: paste a job posting, get the model's verdict from the FastAPI service.

Run (with the API running):  streamlit run streamlit_app.py
The page only talks to the API over HTTP, so it needs no ML libraries. Set API_URL to use a deployed API.
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 90  # a sleeping free-tier API takes about a minute to wake up
TEXT_AREAS = {"description": "Job description *", "company_profile": "Company profile",
              "requirements": "Requirements", "benefits": "Benefits"}
CHOICES = {"employment_type": "Employment type", "required_experience": "Experience", "required_education": "Education"}
SHORT = {"industry": "Industry", "function": "Job function", "location": "Location (e.g. US, NY, New York)",
         "department": "Department", "salary_range": "Salary range (e.g. 40000-50000)"}
FLAGS = {"has_company_logo": "Company logo shown", "has_questions": "Has screening questions",
         "telecommuting": "Remote / work from home"}
TEXT_FIELDS = ["title", *TEXT_AREAS, *SHORT]

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🔎")


@st.cache_data(ttl=3600, show_spinner="Connecting to the model API (can take a minute if it was asleep)...")
def api_info() -> dict:
    """Dropdown options and examples come from the API's own OpenAPI schema, so the page can't drift from it."""
    spec = requests.get(f"{API_URL}/openapi.json", timeout=TIMEOUT).json()
    props = spec["components"]["schemas"]["JobPosting"]["properties"]
    examples = spec["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["examples"]
    return {"choices": {f: props[f]["anyOf"][0]["enum"] for f in CHOICES},
            "examples": {k: v["value"] for k, v in examples.items()},
            "health": requests.get(f"{API_URL}/health", timeout=TIMEOUT).json()}


def fill(posting: dict):
    """Put a posting into the form (used by the example / clear buttons)."""
    for f in TEXT_FIELDS:
        st.session_state[f] = posting.get(f, "")
    for f in CHOICES:
        st.session_state[f] = posting.get(f)
    for f in FLAGS:
        st.session_state[f] = posting.get(f, False)


st.title("Fake Job Posting Detector")
st.write("Paste a job posting to check whether it looks like the scams in the training data. "
         "Fill in as much as you can: the model expects complete postings.")

try:
    info = api_info()
except requests.RequestException as e:
    st.error(f"Can't reach the model API at `{API_URL}`. Start it with `uvicorn app.main:app` "
             f"or set the `API_URL` environment variable.\n\n`{type(e).__name__}`")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.button("Try a fake-looking example", on_click=fill, args=(info["examples"]["fake"],), width="stretch")
c2.button("Try a real-looking example", on_click=fill, args=(info["examples"]["real"],), width="stretch")
c3.button("Clear", on_click=fill, args=({},), width="stretch")

with st.form("posting"):
    st.text_input("Job title *", key="title")
    for f, label in TEXT_AREAS.items():
        st.text_area(label, key=f, height=180 if f == "description" else 100)
    for col, (f, label) in zip(st.columns(3), CHOICES.items()):
        col.selectbox(label, [None, *info["choices"][f]], key=f, placeholder="Not specified",
                      format_func=lambda v: "Not specified" if v is None else v)
    cols = st.columns(2)
    for i, (f, label) in enumerate(SHORT.items()):
        cols[i % 2].text_input(label, key=f)
    for col, (f, label) in zip(st.columns(3), FLAGS.items()):
        col.checkbox(label, key=f)
    submitted = st.form_submit_button("Check posting", type="primary", width="stretch")

if submitted:
    payload = {f: st.session_state[f].strip() or None for f in TEXT_FIELDS}   # blank optional field = missing
    payload |= {f: payload[f] or "" for f in ["title", "description"]}       # required: let the API explain
    payload |= {f: st.session_state[f] for f in CHOICES} | {f: st.session_state[f] for f in FLAGS}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        st.error(f"The API didn't respond: `{type(e).__name__}`")
        st.stop()
    if r.status_code == 422:  # validation error: show the API's own messages
        labels = {"title": "Job title", **TEXT_AREAS, **CHOICES, **SHORT, **FLAGS}
        st.error("Please fix:\n\n" + "\n".join(f"- **{labels.get(e['loc'][-1], e['loc'][-1]).rstrip(' *')}**: {e['msg']}"
                                             for e in r.json()["detail"]))
        st.stop()
    r.raise_for_status()
    res = r.json()
    p, t = res["fraud_probability"], res["threshold"]

    if res["is_fraud"]:
        st.error(f"### Likely fake\nRisk score **{p:.3f}** is at or above the decision threshold ({t:.3f}).", icon="⚠️")
    else:
        st.success(f"### Looks real\nRisk score **{p:.3f}** is below the decision threshold ({t:.3f}).", icon="✅")
    st.progress(p, text=f"Risk score {p:.3f}  ·  threshold {t:.3f}")

    empty = [f for f in [*TEXT_FIELDS, *CHOICES] if not payload[f]]
    if len(empty) >= 4:
        st.info(f"{len(empty)} of {len(TEXT_FIELDS) + len(CHOICES)} fields were left empty. Fake postings in the "
                "training data are often short and incomplete, so missing fields raise the score.")
    st.caption("The risk score is not a calibrated probability: compare it with the threshold. The model flags "
               "patterns seen in 2012–2014 scam postings; it can't verify that a company or job exists.")

h = info["health"]
st.divider()
st.caption(f"Model: {h['model']} ({h['imbalance']}), trained {h['trained_on']} · API docs: {API_URL}/docs")
