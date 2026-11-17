"""Cleaning + feature engineering for the fake job posting detector.

Everything is wrapped in one sklearn Pipeline (`build_preprocessor`), so training,
cross-validation and the API all apply exactly the same steps to raw posting fields.
Run `python -m src.preprocess` for a sanity check.
"""
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fake_job_postings.csv"
TARGET = "fraudulent"

# Raw fields the pipeline takes in (the API accepts the same ones)
TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
CAT_FIELDS = ["employment_type", "required_experience", "required_education", "industry", "function"]
OTHER_FIELDS = ["location", "department", "salary_range"]  # only used for has_salary / n_missing
FLAG_FIELDS = ["telecommuting", "has_company_logo", "has_questions"]
INPUT_FIELDS = TEXT_FIELDS + CAT_FIELDS + OTHER_FIELDS + FLAG_FIELDS

# Engineered features (each one comes from an EDA finding)
LENGTH_FIELDS = ["company_profile", "description", "requirements"]
NUM_FEATURES = [f"{c}_words" for c in LENGTH_FIELDS] + ["n_missing"]
BIN_FEATURES = FLAG_FIELDS + ["has_salary", "has_email", "has_phone", "has_money_words"]

# The dataset anonymised contacts as #URL_<hash>#, #EMAIL_<hash>#, #PHONE_<hash>#; real postings sent to the
# API contain real ones. Both forms map to the same token, so features mean the same thing at train and serve time.
URL_RE = re.compile(r"https?://\S+|www\.\S+|#URL_\w+#", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}|#EMAIL_\w+#")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b|#PHONE_\w+#")  # strict: skips ID/salary digits
MONEY_RE = re.compile(r"\$\s?\d|\b(?:earn|income|cash|money)\b|\bper (?:hour|day|week)\b", re.I)


def clean_text(text: str) -> str:
    """Raw posting text -> lowercase words. Stopwords are removed later by TfidfVectorizer."""
    text = html.unescape(text)                          # &amp; -> &
    text = re.sub(r"<[^>]+>", " ", text)                # HTML tags (API users may paste HTML)
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)    # glued list items: neededHelping -> needed Helping
    text = re.sub(r"[^a-z]+", " ", text.lower())        # letters only
    return text.strip()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw posting fields -> one column per model input. Learns nothing from data, so it can't leak."""
    df = df.reindex(columns=INPUT_FIELDS)                 # absent fields -> all-NaN columns
    str_cols = TEXT_FIELDS + CAT_FIELDS + OTHER_FIELDS
    df[str_cols] = df[str_cols].astype(object).replace(r"^\s*$", np.nan, regex=True)  # blank string = missing
    text = df[TEXT_FIELDS].fillna("")
    raw = text.agg(" ".join, axis=1)

    out = pd.DataFrame({"text": raw.map(clean_text)}, index=df.index)
    out[CAT_FIELDS] = df[CAT_FIELDS].fillna("Missing")             # missingness is a signal, keep it
    for c in LENGTH_FIELDS:
        out[f"{c}_words"] = text[c].str.split().str.len()
    out["n_missing"] = df[str_cols].isna().sum(axis=1)
    out[FLAG_FIELDS] = df[FLAG_FIELDS].fillna(0).astype(int)
    out["has_salary"] = df["salary_range"].notna().astype(int)
    out["has_email"] = out["text"].str.contains(r"\bemailtoken\b").astype(int)
    out["has_phone"] = out["text"].str.contains(r"\bphonetoken\b").astype(int)
    out["has_money_words"] = raw.str.contains(MONEY_RE).astype(int)
    return out


def build_preprocessor() -> Pipeline:
    """Raw DataFrame -> sparse feature matrix. Vocabulary, scaling and categories are learned in fit()."""
    columns = ColumnTransformer([
        ("text", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3,
                                 max_features=20_000, sublinear_tf=True), "text"),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CAT_FIELDS),
        ("num", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                              StandardScaler()), NUM_FEATURES),                 # skewed counts -> log -> scale
        ("bin", "passthrough", BIN_FEATURES),
    ])
    return Pipeline([("features", FunctionTransformer(build_features)), ("columns", columns)])


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and drop exact duplicate postings (same content, different job_id)."""
    df = pd.read_csv(path)
    n = len(df)
    df = df.drop_duplicates(subset=df.columns.drop("job_id")).reset_index(drop=True)
    print(f"Loaded {n:,} rows; dropped {n - len(df)} exact duplicates -> {len(df):,}")
    return df


def split(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified split: train and test keep the same fake/real ratio."""
    return train_test_split(df[INPUT_FIELDS], df[TARGET], test_size=test_size,
                            stratify=df[TARGET], random_state=SEED)


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    print(f"Train {len(X_train):,} rows ({y_train.mean():.2%} fake) | Test {len(X_test):,} rows ({y_test.mean():.2%} fake)")

    prep = build_preprocessor().fit(X_train)                     # fit on TRAIN only
    Xt_train, Xt_test = prep.transform(X_train), prep.transform(X_test)
    names = prep.named_steps["columns"].get_feature_names_out()
    print(f"\nFeature matrix: train {Xt_train.shape}, test {Xt_test.shape}, type {type(Xt_train).__name__}")
    print("Features per block:", pd.Series([n.split("__")[0] for n in names]).value_counts().to_dict())
    print("Category columns:", [n for n in names if n.startswith("cat__")][:12], "...")

    feats = build_features(X_train)
    print("\nEngineered features, mean by class (train set):")
    print(feats[NUM_FEATURES + BIN_FEATURES].groupby(y_train).mean().T.round(3).rename(columns={0: "real", 1: "fake"}))

    i = X_train.index[X_train["description"].fillna("").str.contains(r"#EMAIL_\w+#[A-Z]|[a-z][A-Z][a-z]")][0]
    print("\nCleaning example (description):\n  RAW:  ", X_train.loc[i, "description"][:350],
          "\n  CLEAN:", clean_text(X_train.loc[i, "description"])[:350])

    # Same pipeline on an API-style input: real email, $ amount, most fields absent
    one = pd.DataFrame([{"title": "Data Entry Clerk", "has_company_logo": 0,
                         "description": "Earn $500 per week from home! Send your CV to hr@quickjobs.com or call (555) 123-4567."}])
    print("\nAPI-style input ->", build_features(one).drop(columns="text").iloc[0].to_dict())
    print("Cleaned text:", build_features(one).loc[0, "text"], "| transformed shape:", prep.transform(one).shape)
