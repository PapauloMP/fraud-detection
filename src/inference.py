import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import webbrowser
from transformers import AutoTokenizer, AutoModel
from lime.lime_text import LimeTextExplainer
from datetime import datetime

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "generation/outputs/models/best_model_no_metrics_20260521_031048.pt"
CONFIG_PATH = WEIGHTS_PATH.replace(".pt", "_config.json")
LIME_OUTPUT_DIR = "generation/outputs/analysis/"

os.makedirs(os.path.dirname(LIME_OUTPUT_DIR), exist_ok=True)

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
        
        # MASKED MEAN POOLING
        token_embeddings = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        logits = self.classifier(pooled_output)
        return logits

def get_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-12)) # Entropia de Shannon

def apply_single_threshold(probs, thresholds):
    passed = [i for i in range(len(probs)) if probs[i] >= thresholds[i]]
    if len(passed) == 1:
        return passed[0]
    else:
        return np.argmax(probs)

def predict_text(text, model, tokenizer, temperature, thresholds):
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        
        calibrated_logits = logits / temperature
        
        probs = F.softmax(calibrated_logits, dim=1).cpu().numpy()[0]

    predict_class = apply_single_threshold(probs, thresholds)
    
    return probs, predict_class

def make_lime_predict_fn(model, tokenizer, temperature):
    def predict_fn(texts):
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
    return predict_fn

def main():
    print("="*50)
    print("Carregando...")
    print("="*50)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"\nERRO: Arquivo de pesos não encontrado em: {WEIGHTS_PATH}")
        return
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    
    temperature = config.get("temperature", 1.0)
    thresholds = config.get("thresholds", [0.5, 0.5, 0.5])
    epoch_saved = config.get("epoch", -1)
    val_loss = config.get("loss", "N/A")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BERTClassifier(MODEL_NAME, num_classes=3)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    # Inicializa o Explicador LIME
    labels_map = {0: "Humano", 1: "IA", 2: "Copy-Typing"}
    class_names = [labels_map[0], labels_map[1], labels_map[2]]
    explainer = LimeTextExplainer(class_names=class_names)
    lime_predict_fn = make_lime_predict_fn(model, tokenizer, temperature)
    
    print("\n Modelo Carregado com Sucesso!")
    print(f"Época de melhor performance: {epoch_saved + 1}")
    print(f"Loss de Validação: {val_loss:.4f}" if isinstance(val_loss, float) else f" └─ Loss: {val_loss}")
    print(f"Temperatura de Calibração: {temperature:.4f}")
    print(f"Thresholds Ajustados: {thresholds}")
    
    print("\nBEM-VINDO AO SISTEMA DE DETECÇÃO DE FRAUDE")
    
    while True:
        print("-" * 50)
        user_input = input("\nCole um texto para análise (ou digite 'sair' para encerrar):\n> ")
        
        if user_input.strip().lower() == 'sair':
            print("Encerrando o sistema...")
            break
            
        if len(user_input.strip()) < 10:
            print("Falha: texto muito curto.")
            continue

        print("\nAnalisando...")
        
        probs, predict_class = predict_text(user_input, model, tokenizer, temperature, thresholds)
        
        confidence = probs[predict_class] * 100
        entropy = get_entropy(probs)
        veredict = labels_map[predict_class]

        print("\nRESULTADO DA ANÁLISE:")
        print(f"Veredito: {veredict.upper()}")
        print(f"Certeza : {confidence:.2f}%")
        print(f"Entropia: {entropy:.4f}")
        
        print("\nProbabilidades:")
        print(f"[{probs[0]*100:05.2f}%] Humano")
        print(f"[{probs[1]*100:05.2f}%] IA Pura")
        print(f"[{probs[2]*100:05.2f}%] Copy-Typing")

        print("\nDesconstruindo a decisão com LIME (aguarde alguns segundos)...")
        try:
            exp = explainer.explain_instance(
                user_input, 
                lime_predict_fn, 
                labels=[predict_class],
                num_features=20, 
                num_samples=5000
            )
            
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"lime_{veredict.replace(' ', '_')}_{now}.html"
            file_path = os.path.join(LIME_OUTPUT_DIR, file_name)

            exp.save_to_file(file_path)
            print(f"Explicação LIME gerada com sucesso!")
            
            file_url = 'file://' + os.path.abspath(file_path)
            webbrowser.open(file_url)
            
        except Exception as e:
            print(f"Erro ao gerar LIME: {e}")

if __name__ == "__main__":
    main()