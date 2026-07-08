import os
import torch
import torch.nn.functional as F
import numpy as np
import joblib
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class ProductionInferenceEngine:
    def __init__(
            self,
            roberta_dir: str = '../artifacts/roberta_model/checkpoint-8232',
            classical_model_path: str = '../artifacts/lr_model.joblib',
            vertorizer_path: str = '../artifacts/tfidf_vectorizer.joblib'
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Production Engine on hardware: {str(self.device).upper()}")

        self.has_transformer = False
        self.has_classical = False
        self.label_mapping = {0: "FAKE NEWS", 1: "REAL NEWS"}
        # Model is Available at HuggingFace Hub
        model_path = "amansshrrma/RoBERTa_pre_trained"
        # Load RoBERTa Transformer Stack
        if os.path.exists(roberta_dir):
            print("Loading Fine-Tuned RoBERTa Transformer weights...")

            # ======== When local Roberta is available ========
            # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            # self.transformer = RobertaForSequenceClassification.from_pretrained(roberta_dir, torch_dtype=torch.float16)
            # =================================================

            # ======== When Cloud Roberta is available ========
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.transformer = RobertaForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.float16)
            # =================================================

            self.transformer.to(self.device)
            self.transformer.eval()
            self.has_transformer = True
        else:
            print(f"WARNING: RoBERTa weights not found at {roberta_dir}. Transformer inference disabled.")
            self.has_transformer = False

        # Load Classical ML Baseline Stack
        if os.path.exists(classical_model_path) and os.path.exists(vertorizer_path):
            print('Loading Classical ML Baseline & Vectorizer artifacts...')
            self.vectorizer = joblib.load(vertorizer_path)
            self.classical_model = joblib.load(classical_model_path)
            self.has_classical = True
        else:
            print("WARNING: Classical baseline components missing. Classical inference disabled.")


    def predict_test(self, raw_text: str):
        # Runs parallel extraction passes through both active model architectures.
        results = {}

        if not raw_text.strip():
            return {"error": "Input text is completely empty."}

        # --- Pipeline A: Classical ML Inference ---
        if self.has_classical:
            # Note: Classical model expects the preprocessed/cleaned tokens
            # (If you have a clean_text function in text_preprocessing.py, you can apply it here)
            vertorized_input = self.vectorizer.transform([raw_text])
            clean_pred = self.classical_model.predict(vertorized_input)[0]

            # Hotfix: Inject missing attribute if unpickling from a different scikit-learn version
            if not hasattr(self.classical_model, 'multi_class'):
                self.classical_model.multi_class = 'auto'

            if hasattr(self.classical_model, 'predict_proba'):
                classical_conf = np.max(self.classical_model.predict_proba(vertorized_input))
            else:
                classical_conf = 1

            results['classical_baseline'] = {
                'prediction': self.label_mapping[clean_pred],
                'confidence': float(classical_conf)
            }

        # --- Pipeline B: RoBERTa Transformer Inference ---
        if self.has_transformer:
            inputs = self.tokenizer(
                raw_text,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                probs = F.softmax(outputs.logits, dim=1).flatten().cpu().numpy()
                pred_ids = np.argmax(probs)

            results['roberta_transformer'] = {
                'prediction': self.label_mapping[pred_ids],
                'confidence': float(probs[pred_ids]),
                'raw_probability': {
                    'Fake': float(probs[0]),
                    'Real': float(probs[1])
                }
            }

        return results

if __name__ == "__main__":
    # Point paths to your local saved artifacts
    # Adjust classical paths if your train.py saved them under different names/directories
    engine = ProductionInferenceEngine()

    print("\n" + "="*20 + " FND INFERENCE CLI INITIALIZED " + "="*20)
    print("Type 'exit' or 'quit' to terminate the stream loop session.\n")

    while True:
        user_input = input('Enter News Article Content Text to Analyze:\n> ')
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Terminating deployment engine session...")
            break

        if not user_input.strip():
            continue

        output_metrics = engine.predict_test(user_input)

        print("\n" + "-"*15 + " MODEL COMPARISON OUTPUT " + "-"*15)
        for model_name, data in output_metrics.items():
            print(f"[{model_name.upper()}]")
            print(f"  └── Classification : {data['prediction']}")
            if "confidence" in data:
                print(f"  └── System Confidence: {data['confidence']:.2%}")
            elif "margin_distance_score" in data:
                print(f"  └── Boundary Margin Distance: {data['margin_distance_score']:.4f}")
        print("-" * 55 + "\n")