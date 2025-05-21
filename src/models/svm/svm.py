from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing     import StandardScaler
from sklearn.svm               import LinearSVC
from sklearn.metrics           import f1_score, classification_report

# ---------------------------- paths ---------------------------------
DATA_DIR   = Path("data/processed/SVM")
TRAIN_CSV  = DATA_DIR / "train_text.csv"
TEST_CSV   = DATA_DIR / "test_text.csv"

MODEL_DIR  = Path("models/trained")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------- schema --------------------------------
TEXT_COL   = "text_blob"
NUM_COLS   = [
    "store_sales(in millions)","store_cost(in millions)",
    "unit_sales(in millions)","SRP","gross_weight","net_weight",
    "units_per_case","store_sqft","grocery_sqft","frozen_sqft","meat_sqft"
]

def load_xy(split: str = "train"):
    csv = TRAIN_CSV if split == "train" else TEST_CSV
    df  = pd.read_csv(csv)
    X   = df.drop(columns=["cost_class"])
    y   = df["cost_class"]
    return X, y
class SVMClassifier:

    def __init__(self, C: float = 5.0, max_iter: int = 15_000):
        self.C        = C
        self.max_iter = max_iter

        self.model = Pipeline([
            ("prep", ColumnTransformer([
                ("txt", TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1,2),
                    min_df=2,
                    sublinear_tf=True
                ), TEXT_COL),
                ("num", StandardScaler(), NUM_COLS)
            ])),
            ("svm", LinearSVC(
                C=self.C,
                class_weight="balanced",
                max_iter=self.max_iter
            ))
        ])

    def fit(self, split: str = "train"):
        X, y = load_xy(split)
        print(f"▶ Fitting on {len(y):,} rows …")
        self.model.fit(X, y)

    def score(self, split: str = "test") -> float:
        X, y = load_xy(split)
        preds = self.model.predict(X)
        macro = f1_score(y, preds, average="macro")
        print("Classification Report:")
        print(classification_report(y, preds, digits=2))
        return macro

    def save(self, name: str = "svm_text.pkl") -> Path:
        path = MODEL_DIR / name
        joblib.dump(self.model, path)
        return path

    def load(self, path: str | Path):
        self.model = joblib.load(path)
        return self