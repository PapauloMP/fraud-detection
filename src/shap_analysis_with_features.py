import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import joblib
import gc

from feature_extractor import extract_features

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "models", "best_extended_model_no_metrics_20260719_015113.pt")
CONFIG_PATH = WEIGHTS_PATH.replace(".pt", "_config.json")
DATASET_PATH = os.path.join(PROJECT_ROOT, "inputs", "datasets", "test_dataset.csv")
SHAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "analysis", "shap")

MODEL_DIR = os.path.dirname(WEIGHTS_PATH)
timestamp = WEIGHTS_PATH.replace(".pt", "")[-15:]
SCALER_PATH = os.path.join(MODEL_DIR, f"metrics_scaler_{timestamp}.pkl")

# MAX_SAMPLES_PER_CLASS = 50 

os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        token_embeddings = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return self.classifier(sum_embeddings / sum_mask)

class HybridModel(nn.Module):
    def __init__(self, model_name, num_metrics, num_classes):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.metrics_proj = nn.Sequential(
            nn.Linear(num_metrics, 32),
            nn.ReLU(), 
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size + 32, 256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, metrics):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        token_embeddings = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        metrics_feat = self.metrics_proj(metrics)
        
        return self.classifier(torch.cat([sum_embeddings / sum_mask, metrics_feat], dim=1))

def main():
    print("-> Carregando configurações...")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    temperature = config.get("temperature", 1.0)
    feature_cols = config.get("features_used", [])
    use_features = len(feature_cols) > 0
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    scaler = None

    if use_features:
        print(f"-> Arquitetura Híbrida detectada. Carregando scaler...")
        model = HybridModel(MODEL_NAME, num_metrics=len(feature_cols), num_classes=3)
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            print(f"Erro ao carregar scaler: {e}")
            return
    else:
        print("-> Arquitetura Base detectada.")
        model = BERTClassifier(MODEL_NAME, num_classes=3)

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    def custom_predict_fn(texts):
        if isinstance(texts, str): texts = [texts]
        elif isinstance(texts, np.ndarray): texts = texts.tolist()

        all_probs = []
        batch_size = 16 

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            with torch.no_grad():
                if use_features:
                    batch_metrics = []
                    for txt in batch:
                        feats = extract_features(txt)
                        batch_metrics.append([feats.get(c, 0.0) for c in feature_cols])
                    
                    metrics_arr = scaler.transform(batch_metrics)
                    metrics_tensor = torch.tensor(metrics_arr, dtype=torch.float).to(DEVICE)
                    
                    logits = model(input_ids, attention_mask, metrics_tensor)
                else:
                    logits = model(input_ids, attention_mask)

                calibrated_logits = logits / temperature
                probs = F.softmax(calibrated_logits, dim=1)
                
            all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)

    print("-> Carregando base de dados...")
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    
    if df["classe"].dtype == object:
        df["classe"] = df["classe"].map({"humano": 0, "ia": 1, "copy": 2})

    classes_dict = {0: "Humano", 1: "IA Pura", 2: "Copy-Typing"}
    
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(custom_predict_fn, masker, output_names=["Humano", "IA Pura", "Copy-Typing"])
    
    for class_id, class_name in classes_dict.items():
        print("\n" + "="*50)
        print(f"INICIANDO ETAPA: Classe '{class_name}'")
        
        df_class = df[df["classe"] == class_id]
        class_texts = df_class["texto"].fillna("").astype(str).tolist()

        # if len(class_texts) > MAX_SAMPLES_PER_CLASS:
        #     class_texts = class_texts[:MAX_SAMPLES_PER_CLASS]

        print(f"Calculando Valores SHAP...")
        shap_values = explainer(class_texts)
        
        class_file_name = class_name.lower().replace(' ', '_').replace('-', '_')
        plot_file = os.path.join(SHAP_OUTPUT_DIR, f"shap_bar_plot_{class_file_name}.png")
        html_file = os.path.join(SHAP_OUTPUT_DIR, f"shap_report_{class_file_name}.html")
        
        # Gráfico Global de Barras
        print(f"Salvando gráfico global...")
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values[:, :, class_id], max_display=20, show=False)
        plt.title(f"Top 20 Palavras que definem: {class_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # HTML Dinâmico
        print(f"Salvando relatório HTML...")
        html_content = shap.plots.text(shap_values, display=False)
        with open(html_file, "w", encoding="utf-8") as f:
            f.write('<meta charset="UTF-8">\n')
            f.write(html_content)
            
        print(f"Etapa '{class_name}' salva com sucesso!")
        
        del shap_values
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("ANÁLISE SHAP CONCLUÍDA COM SUCESSO")

if __name__ == "__main__":
    main()