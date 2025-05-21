from pathlib import Path
from src.models.svm.svm import SVMClassifier, load_xy   # ← import helper

def main() -> None:

    clf = SVMClassifier(C=5.0)
    clf.fit("train")

    print("Evaluating …")
    macro = clf.score("test")
    print(f"Macro-F1 Score: {macro:.4f}")

    X_test, y_test = load_xy("test")
    preds = clf.model.predict(X_test)

    print("\nFirst 5 predictions (test set):")
    for i in range(5):
        print(f"  Row {i+1:>2}: pred = {preds[i]}   |   true = {y_test.iloc[i]}")

    model_path: Path = clf.save()
    print("\nModel saved to:", model_path.as_posix())

if __name__ == "__main__":
    main()