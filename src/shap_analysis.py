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
import gc

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

os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        return logits

print("-> Carregando pesos e calibração...")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

temperature = config.get("temperature", 1.0)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BERTClassifier(MODEL_NAME, num_classes=3)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

def custom_predict_fn(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()

    all_probs = []
    batch_size = 16 

    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            calibrated_logits = logits / temperature
            probs = F.softmax(calibrated_logits, dim=1)
            
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)

def main():
    print("-> Carregando base de dados inteira...")
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")

    classes_dict = {
        0: "Humano",
        1: "IA Pura",
        2: "Copy-Typing"
    }

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(
        custom_predict_fn, 
        masker, 
        output_names=["Humano", "IA Pura", "Copy-Typing"]
    )
    
    for class_id, class_name in classes_dict.items():
        print("="*50)
        print(f"INICIANDO ETAPA: Classe '{class_name}'")
        
        df_class = df[df["classe"] == class_id]
        class_texts = df_class["texto"].fillna("").tolist()

        print(f"Analisando {len(class_texts)} redações...")
        
        shap_values = explainer(class_texts)
        
        class_file_name = class_name.lower().replace(' ', '_').replace('-', '_')
        plot_file = os.path.join(SHAP_OUTPUT_DIR, f"shap_bar_plot_{class_file_name}.png")
        html_file = os.path.join(SHAP_OUTPUT_DIR, f"shap_report_{class_file_name}.html")
        
        # SHAP bar plot
        print(f"Salvando gráfico global para '{class_name}'...")
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values[:, :, class_name], max_display=20, show=False)
        plt.title(f"Top 20 Palavras: {class_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Salvando relatório HTML para '{class_name}'...")
        html_content = shap.plots.text(shap_values, display=False)
        with open(html_file, "w", encoding="utf-8") as f:
            f.write('<meta charset="UTF-8">\n')
            f.write(html_content)
            
        print(f"Etapa '{class_name}' salva com sucesso!")
        
        del shap_values
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("="*50)
    print("ANÁLISE CONCLUÍDA COM SUCESSO")

if __name__ == "__main__":
    main()