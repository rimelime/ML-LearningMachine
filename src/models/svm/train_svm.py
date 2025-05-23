from pathlib import Path
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.models.svm.svm import SVMClassifier, load_xy   # your helper

def main() -> None:
    t0 = time.time()

    # ── 1. Train ───────────────────────────────────────────────────────
    clf = SVMClassifier(C=5.0)
    clf.fit("train")
    print("Training finished")

    # ── 2. Evaluate ────────────────────────────────────────────────────
    print("Evaluating …")
    macro = clf.score("test")
    print(f"Macro-F1 Score: {macro:.4f}")

    # ── 3. First-five sanity check ─────────────────────────────────────
    X_test, y_test = load_xy("test")
    preds = clf.model.predict(X_test)

    print("\nFirst 5 predictions (test set):")
    for i in range(5):
        print(f"  Row {i+1:>2}: pred = {preds[i]}   |   true = {y_test.iloc[i]}")

    # ── 4. Confusion-matrix heat map  ──────────────────────────────────
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, preds, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax)                       # default colour map (policy-safe)
    ax.set_title("SVM Confusion Matrix (test set)")
    plt.tight_layout()
    plt.show()

    # ── 5. Save model ──────────────────────────────────────────────────
    model_path: Path = clf.save()
    print("\nModel saved to:", model_path.as_posix())
    print(f"Total time: {time.time()-t0:.1f} s")

if __name__ == "__main__":
    main()