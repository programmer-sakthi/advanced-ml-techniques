from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def evaluate_classifier(y_test, y_pred):
    # Confusion Matrix
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("===================")

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=2))
    print("===================")

    # Individual Metrics (weighted average)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    precision = report["weighted avg"]["precision"]

    print(f"accuracy: {accuracy:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"f1-score: {f1:.3f}")
    print(f"precision: {precision:.3f}")


def auc_roc(classifier, X_test, y_test):
    y_prob = classifier.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)
