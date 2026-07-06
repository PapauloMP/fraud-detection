import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import webbrowser
import joblib
import shap
from transformers import AutoTokenizer, AutoModel
from lime.lime_text import LimeTextExplainer
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

from feature_extractor import extract_features

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_PATH = "generation/outputs/models/best_extended_hybrid_model_20260620_205437.pt" # "generation/outputs/models/best_hybrid_model_20260607_214345.pt"   
CONFIG_PATH = WEIGHTS_PATH.replace(".pt", "_config.json")
TEST_DATASET_PATH = "generation/inputs/test_dataset.csv"
model_dir = os.path.dirname(WEIGHTS_PATH)
timestamp = WEIGHTS_PATH.replace(".pt", "")[-15:]
SCALER_PATH = os.path.join(model_dir, f"metrics_scaler_{timestamp}.pkl")

OUTPUT_DIR = "generation/outputs/analysis"
LIME_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "lime")
SHAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "shap")
CAPTUM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "captum")
INFERENCE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "inference")

for d in [LIME_OUTPUT_DIR, SHAP_OUTPUT_DIR, CAPTUM_OUTPUT_DIR, INFERENCE_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        return self.classifier(pooled_output)

class HybridModel(nn.Module):
    def __init__(self, model_name, num_metrics, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size 
        self.metrics_proj = nn.Sequential(
            nn.Linear(num_metrics, 32), nn.ReLU(), nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, metrics):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        metrics_feat = self.metrics_proj(metrics)
        combined = torch.cat([pooled_output, metrics_feat], dim=1)
        return self.classifier(combined)

def get_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-12))

def apply_single_threshold(probs, thresholds):
    passed = [i for i in range(len(probs)) if probs[i] >= thresholds[i]]
    return passed[0] if len(passed) == 1 else np.argmax(probs)

def prepare_metrics_batch(texts, feature_cols, scaler):
    """Extrai e normaliza as métricas em lote para o modelo híbrido"""
    batch_metrics = []
    for text in texts:
        feats = extract_features(text)
        batch_metrics.append([feats.get(c, 0.0) for c in feature_cols])
    return scaler.transform(batch_metrics)

def predict_batch(texts, model, tokenizer, temperature, use_features, feature_cols=None, scaler=None, batch_size=16):
    all_probs = []
    model.eval()
    
    disable_tqdm = len(texts) < 50 

    for i in tqdm(range(0, len(texts), batch_size), desc="Inferência", leave=False, disable=disable_tqdm):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            if use_features:
                metrics_arr = prepare_metrics_batch(batch, feature_cols, scaler)
                metrics_tensor = torch.tensor(metrics_arr, dtype=torch.float).to(DEVICE)
                logits = model(input_ids, attention_mask, metrics_tensor)
            else:
                logits = model(input_ids, attention_mask)

            calibrated_logits = logits / temperature
            probs = F.softmax(calibrated_logits, dim=1)
            
        all_probs.extend(probs.cpu().numpy())
        
    return np.array(all_probs)

def make_captum_forward(model, use_features):
    if use_features:
        def forward(inputs_embeds, attention_mask, metrics):
            outputs = model.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state 
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            metrics_feat = model.metrics_proj(metrics)
            return model.classifier(torch.cat([sum_embeddings / sum_mask, metrics_feat], dim=1))
        return forward
    else:
        def forward(inputs_embeds, attention_mask):
            outputs = model.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state 
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return model.classifier(sum_embeddings / sum_mask)
        return forward

def evaluate_test_dataset(model, tokenizer, temperature, thresholds, labels_map, use_features, feature_cols, scaler):
    print("\n" + "="*50)
    print("INICIANDO AVALIAÇÃO DO DATASET DE TESTE")
    print("="*50)

    if not os.path.exists(TEST_DATASET_PATH):
        print(f"ERRO: Arquivo não encontrado em {TEST_DATASET_PATH}")
        return

    df = pd.read_csv(TEST_DATASET_PATH, encoding="utf-8")
    texts = df["texto"].fillna("").astype(str).tolist()
    
    if df["classe"].dtype == object:
        y_true = df["classe"].map({"humano": 0, "ia": 1, "copy": 2}).tolist()
    else:
        y_true = df["classe"].tolist()

    print(f"Extraindo probabilidades de {len(texts)} redações...")
    probs_array = predict_batch(texts, model, tokenizer, temperature, use_features, feature_cols, scaler)
    y_pred = [apply_single_threshold(p, thresholds) for p in probs_array]

    # CÁLCULO DA ANÁLISE DE INCERTEZA GLOBAL (ENTROPIA E CONFIANÇA)
    confidences = [probs_array[i][y_pred[i]] for i in range(len(y_pred))]
    mean_confidence = np.mean(confidences)

    entropies = [get_entropy(p) for p in probs_array]
    mean_entropy = np.mean(entropies)

    errors_confidences = [confidences[i] for i in range(len(y_pred)) if y_true[i] != y_pred[i]]
    if len(errors_confidences) > 0:
        mean_confidence_errors = np.mean(errors_confidences)
    else:
        mean_confidence_errors = 0.0

    class_names = [labels_map[0], labels_map[1], labels_map[2]]
    cm = confusion_matrix(y_true, y_pred)
    cr_text = classification_report(y_true, y_pred, target_names=class_names)
    cr_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cr_dict["uncertainty_analysis"] = {
        "mean_confidence": float(mean_confidence),
        "mean_confidence_errors": float(mean_confidence_errors),
        "mean_entropy": float(mean_entropy)
    }
    
    print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
    print(cr_text)

    print("\n" + "-"*60)
    print("ANÁLISE DE INCERTEZA (MÉTRICAS TERMODINÂMICAS):")
    print(f"Confiança Média Predita : {mean_confidence * 100:.2f}%")
    print(f"Confiança Média nos Erros: {mean_confidence_errors * 100:.2f}%")
    print(f"Entropia Média (Shannon) : {mean_entropy:.4f}")
    print("-" * 60)

    now = datetime.now().strftime('%Y%m%d_%H%M%S')

    # SALVAR O RELATORIO
    csv_filename = os.path.join(INFERENCE_OUTPUT_DIR, f"classification_report_hybrid_{now}.csv")
    df_report = pd.DataFrame(cr_dict).transpose()
    df_report.to_csv(csv_filename)
    
    # HEATMAP
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Previsão do Modelo')
    plt.ylabel('Classe Real (Gabarito)')
    plt.title(f'Matriz de Confusão - Base de teste (Híbrido)')
    cm_filename = os.path.join(INFERENCE_OUTPUT_DIR, f"confusion_matrix_hybrid_{now}.png")
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nMatriz salva em: {cm_filename}")
    print(f"\nRelatório de classificação e incerteza salvo em: {csv_filename}")

def main():
    print("="*60)
    print("INICIALIZANDO MOTOR UNIVERSAL DE INFERÊNCIA XAI")
    print("="*60)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERRO FATAL: Arquivo de pesos não encontrado: {WEIGHTS_PATH}")
        return
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    
    temperature = config.get("temperature", 1.0)
    thresholds = config.get("thresholds", [0.5, 0.5, 0.5])
    feature_cols = config.get("features_used", [])
    use_features = len(feature_cols) > 0
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    scaler = None

    if use_features:
        print("[SISTEMA] Arquitetura detectada: HÍBRIDA (Late Fusion)")
        model = HybridModel(MODEL_NAME, num_metrics=len(feature_cols), num_classes=3)
        try:
            scaler = joblib.load(SCALER_PATH)
            print(f"[OK] Scaler carregado: {len(feature_cols)} features ativas.")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar o Scaler em {SCALER_PATH}. O modelo Híbrido exige o Scaler para normalizar os dados.\n{e}")
            return
    else:
        print("[SISTEMA] Arquitetura detectada: BASE (Somente Texto)")
        model = BERTClassifier(MODEL_NAME, num_classes=3)

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    labels_map = {0: "Humano", 1: "IA", 2: "Copy-Typing"}
    class_names = [labels_map[0], labels_map[1], labels_map[2]]
    
    def xai_predict_wrapper(texts):
        return predict_batch(texts, model, tokenizer, temperature, use_features, feature_cols, scaler)
        
    explainer_lime = LimeTextExplainer(class_names=class_names)
    masker_shap = shap.maskers.Text(tokenizer)
    explainer_shap = shap.Explainer(xai_predict_wrapper, masker_shap, output_names=class_names)
    captum_forward_fn = make_captum_forward(model, use_features)
    
    print("\n[OK] Motor Neural Online. Aguardando comandos.")
    
    while True:
        print("\n" + "="*50)
        print("MENU PRINCIPAL - AUDITORIA FORENSE")
        print("="*50)
        print(" [1] Analisar uma Redação (Completo com LIME/SHAP/IG)")
        print(" [2] Avaliar o Dataset de Teste em Lote")
        print(" [3] Sair")
        print("="*50)
        
        option = input("Escolha uma opção: ").strip()
        
        if option == '3':
            break
            
        elif option == '2':
            evaluate_test_dataset(model, tokenizer, temperature, thresholds, labels_map, use_features, feature_cols, scaler)
            
        elif option == '1':
            user_input = input("\nCole o texto da redação:\n> ").strip()
            if len(user_input) < 10:
                print("Texto muito curto.")
                continue

            probs = xai_predict_wrapper([user_input])[0]
            predict_class = apply_single_threshold(probs, thresholds)
            veredict = labels_map[predict_class]
            now = datetime.now().strftime("%Y%m%d_%H%M%S")

            print("\n" + "-"*40)
            print("LAUDO NEURAL")
            print("-" * 40)
            print(f"Veredito Final: {veredict.upper()}")
            print(f"Entropia (Caos): {get_entropy(probs):.4f}")
            print(f"[{probs[0]*100:05.2f}%] Humano")
            print(f"[{probs[1]*100:05.2f}%] IA")
            print(f"[{probs[2]*100:05.2f}%] Copy-Typing")
            
            if use_features:
                print("\nAssinatura Biomecânica Extraída:")
                feats = extract_features(user_input)
                for col in feature_cols:
                    print(f" -> {col.ljust(20)}: {feats.get(col, 0)}")

            print("\nGerando provas matemáticas de explicabilidade (XAI)...")
            
            # SHAP
            try:
                shap_values = explainer_shap([user_input])
                html_shap = shap.plots.text(shap_values[0, :, predict_class], display=False)
                shap_file = os.path.join(SHAP_OUTPUT_DIR, f"shap_{veredict}_{now}.html")
                with open(shap_file, "w", encoding="utf-8") as f:
                    f.write(f"<html><head><meta charset='utf-8'><style>body{{font-family:sans-serif; padding:20px;}}</style></head><body><h2>SHAP - Predição: {veredict}</h2>")
                    f.write(html_shap)
                    f.write("</body></html>")
                webbrowser.open('file://' + os.path.abspath(shap_file))
                print(" [OK] Relatório SHAP gerado.")
            except Exception as e:
                print(f" [ERRO] SHAP falhou: {e}")

            # LIME
            try:
                exp = explainer_lime.explain_instance(user_input, xai_predict_wrapper, labels=[predict_class], num_features=15, num_samples=300)
                lime_file = os.path.join(LIME_OUTPUT_DIR, f"lime_{veredict}_{now}.html")
                exp.save_to_file(lime_file)
                webbrowser.open('file://' + os.path.abspath(lime_file))
                print(" [OK] Relatório LIME gerado.")
            except Exception as e:
                print(f" [ERRO] LIME falhou: {e}")
                
            # CAPTUM
            try:
                captum_inputs = tokenizer(user_input, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
                input_ids = captum_inputs["input_ids"].to(DEVICE)
                attention_mask = captum_inputs["attention_mask"].to(DEVICE)
                
                model.zero_grad()
                with torch.no_grad():
                    input_embeddings = model.encoder.embeddings(input_ids)
                
                input_embeddings.requires_grad_()
                baseline_embeddings = torch.zeros_like(input_embeddings).to(DEVICE)

                ig = IntegratedGradients(captum_forward_fn)
                
                additional_args = (attention_mask,)
                if use_features:
                    metrics_arr = prepare_metrics_batch([user_input], feature_cols, scaler)
                    metrics_tensor = torch.tensor(metrics_arr, dtype=torch.float).to(DEVICE)
                    additional_args = (attention_mask, metrics_tensor)

                attributions, delta = ig.attribute(
                    inputs=input_embeddings,
                    baselines=baseline_embeddings,
                    additional_forward_args=additional_args,
                    target=int(predict_class),
                    return_convergence_delta=True
                )

                attributions_sum = attributions.sum(dim=-1).squeeze(0)
                attributions_norm = (attributions_sum / torch.norm(attributions_sum)).detach().cpu().numpy()
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                score = float(attributions_sum.sum().detach().cpu().item())

                vis_data = viz.VisualizationDataRecord(
                    word_attributions=attributions_norm, pred_prob=float(probs[predict_class]),
                    pred_class=veredict, true_class="?", attr_class=veredict,
                    attr_score=score, raw_input_ids=tokens, convergence_score=float(delta.item())
                )

                html_captum = viz.visualize_text([vis_data]).data
                captum_file = os.path.join(CAPTUM_OUTPUT_DIR, f"captum_{veredict}_{now}.html")
                with open(captum_file, "w", encoding="utf-8") as f:
                    f.write(html_captum)
                webbrowser.open('file://' + os.path.abspath(captum_file))
                print(" [OK] Relatório Captum IG gerado.")
            except Exception as e:
                print(f" [ERRO] Captum IG falhou: {e}")

if __name__ == "__main__":
    main()