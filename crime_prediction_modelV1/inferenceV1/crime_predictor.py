import json
import joblib
import numpy as np

class CrimePredictor:
    def __init__(self, model_dir="models/crime_type_v1"):
        self.model = joblib.load(f"{model_dir}/model.joblib")
        self.vectorizer = joblib.load(f"{model_dir}/vectorizer.joblib")
        self.mlb = joblib.load(f"{model_dir}/label_binarizer.joblib")
        with open(f"{model_dir}/label_thresholds.json") as f:
            self.thresholds = json.load(f)

        self.label_names = list(self.mlb.classes_)

    def _apply_thresholds(self, probs):
        y_pred = np.zeros_like(probs, dtype=int)
        for j, label in enumerate(self.label_names):
            th = self.thresholds[label]
            y_pred[:, j] = (probs[:, j] >= th).astype(int)
        return y_pred

    def predict(self, text, mode="citizen"):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # safety: very short text
        safe_texts = []
        MIN_WORDS = 8
        for t in texts:
            if len(t.split()) < MIN_WORDS:
                safe_texts.append(None)
            else:
                safe_texts.append(t)

        X_vec = self.vectorizer.transform([t or "" for t in safe_texts])
        proba_list = self.model.predict_proba(X_vec)
        probs = np.array(proba_list)          # (n, n_labels)
        y_pred = self._apply_thresholds(probs)

        results = []
        for i, t in enumerate(texts):
            if safe_texts[i] is None:
                results.append({
                    "error": "Insufficient information to determine crime type"
                })
                continue

            row_probs = probs[i]
            row_pred = y_pred[i]
            labels = self.mlb.inverse_transform(row_pred.reshape(1, -1))[0]

            # citizen vs lawyer
            raw_labels = list(labels)  
            if mode == "citizen":
                citizen_keep = {
                    "homicide", "sexual_offence", "kidnapping",
                    "domestic_violence", "cyber_crime", "fraud",
                    "robbery", "cheating"
                }
                labels = [l for l in labels if l in citizen_keep]

            # simple confidence = max probability of any positive label
            conf = float(row_probs[row_pred == 1].max()) if row_pred.any() else 0.0

            out = {
                "labels": labels,
                "confidence": conf,
                "warning": "This is only an AI indication, not legal advice."
            }
            results.append(out)
            out = {
                "labels": labels,
                "raw_labels": raw_labels,   # full set, for debugging / lawyer UI
                "confidence": conf,
                "warning": "This is only an AI indication, not legal advice."
            }

        return results if isinstance(text, list) else results[0]
