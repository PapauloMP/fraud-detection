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
from transformers import AutoTokenizer, AutoModel
from lime.lime_text import LimeTextExplainer
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "models", "best_extended_model_no_metrics_20260719_015113.pt")
CONFIG_PATH = WEIGHTS_PATH.replace(".pt", "_config.json")
TEST_DATASET_PATH = os.path.join(PROJECT_ROOT, "inputs", "datasets", "test_dataset.csv")

ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
LIME_OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "lime")
CAPTUM_OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "captum")
INFERENCE_OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "inference")

os.makedirs(LIME_OUTPUT_DIR, exist_ok=True)
os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)
os.makedirs(CAPTUM_OUTPUT_DIR, exist_ok=True)

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
    return -np.sum(probs * np.log(probs + 1e-12))

def apply_single_threshold(probs, thresholds):
    passed = [i for i in range(len(probs)) if probs[i] >= thresholds[i]]
    if len(passed) == 1:
        return passed[0]
    else:
        return np.argmax(probs)

def predict_batch(texts, model, tokenizer, temperature, batch_size=16):
    all_probs = []
    model.eval()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processando Lotes", leave=False):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            calibrated_logits = logits / temperature
            probs = F.softmax(calibrated_logits, dim=1)
            
        all_probs.extend(probs.cpu().numpy())
        
    return np.array(all_probs)

def predict_text(text, model, tokenizer, temperature, thresholds):
    probs = predict_batch([text], model, tokenizer, temperature, batch_size=1)[0]
    predict_class = apply_single_threshold(probs, thresholds)
    return probs, predict_class

def make_lime_predict_fn(model, tokenizer, temperature):
    return lambda texts: predict_batch(texts, model, tokenizer, temperature)

def make_captum_forward(model):
    def forward(inputs_embeds, attention_mask):
        outputs = model.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = model.classifier(pooled_output)
        return logits
    return forward

def evaluate_test_dataset(model, tokenizer, temperature, thresholds, labels_map):
    print("\n" + "="*50)
    print("INICIANDO AVALIAÇÃO DO DATASET DE TESTE")
    print("="*50)

    if not os.path.exists(TEST_DATASET_PATH):
        print(f"ERRO: Arquivo não encontrado em {TEST_DATASET_PATH}")
        return

    df = pd.read_csv(TEST_DATASET_PATH, encoding="utf-8")
    
    if "texto" not in df.columns or "classe" not in df.columns:
        print("ERRO: O CSV deve conter as colunas 'texto' e 'classe'.")
        return

    texts = df["texto"].fillna("").tolist()
    y_true = df["classe"].tolist()

    print(f"Foram encontradas {len(texts)} redações na base de dados.")
    print("Extraindo probabilidades do modelo...")
    
    probs_array = predict_batch(texts, model, tokenizer, temperature)
    
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

    # RELATÓRIO
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
    print("ANÁLISE DE INCERTEZA:")
    print(f"Confiança Média Predita : {mean_confidence * 100:.2f}%")
    print(f"Confiança Média nos Erros: {mean_confidence_errors * 100:.2f}%")
    print(f"Entropia Média (Shannon) : {mean_entropy:.4f}")
    print("-" * 60)

    now = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Salva o relatório de classificação em CSV
    csv_filename = os.path.join(INFERENCE_OUTPUT_DIR, f"classification_report_{now}.csv")
    df_report = pd.DataFrame(cr_dict).transpose()
    df_report.to_csv(csv_filename)

    # Salva a matriz de confusão em PNG
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Previsão do Modelo')
    plt.ylabel('Classe Real (Gabarito)')
    plt.title('Matriz de Confusão - Base de Teste')
    
    cm_filename = os.path.join(INFERENCE_OUTPUT_DIR, f"confusion_matrix_{now}.png")
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Matriz de confusão salva em: {cm_filename}")
    print(f"Relatório de classificação salvo em: {csv_filename}")

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
    model.eval()

    # Inicializa o explicador LIME
    labels_map = {0: "Humano", 1: "IA", 2: "Copy-Typing"}
    class_names = [labels_map[0], labels_map[1], labels_map[2]]
    explainer = LimeTextExplainer(class_names=class_names)
    lime_predict_fn = make_lime_predict_fn(model, tokenizer, temperature)
    captum_forward_fn = make_captum_forward(model)
    
    print("\n Modelo Carregado com Sucesso!")
    print(f"Época de melhor performance: {epoch_saved + 1}")
    print(f"Loss de Validação: {val_loss:.4f}" if isinstance(val_loss, float) else f" └─ Loss: {val_loss}")
    print(f"Temperatura de Calibração: {temperature:.4f}")
    print(f"Thresholds Ajustados: {thresholds}")
    
    while True:
        print("\n" + "="*50)
        print("MENU PRINCIPAL - DETECÇÃO DE FRAUDE")
        print("="*50)
        print(" [1] Analisar redações (com LIME e IG)")
        print(" [2] Avaliar o dataset de teste")
        print(" [3] Sair do Sistema")
        print("="*50)
        
        option = input("Escolha uma opção: ").strip()
        
        if option == '3':
            print("Encerrando o sistemsaa...")
            break
            
        elif option == '2':
            evaluate_test_dataset(model, tokenizer, temperature, thresholds, labels_map)
            
        elif option == '1':
            while True:
                print("\n" + "-"*50)
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
                print(f"[{probs[1]*100:05.2f}%] IA")
                print(f"[{probs[2]*100:05.2f}%] Copy-Typing")

                now = datetime.now().strftime("%Y%m%d_%H%M%S")

                print("\nDesconstruindo a decisão com LIME...")
                try:
                    exp = explainer.explain_instance(
                        user_input, 
                        lime_predict_fn, 
                        labels=[predict_class],
                        num_features=15, 
                        num_samples=500
                    )
                    
                    lime_file = os.path.join(LIME_OUTPUT_DIR, f"lime_{veredict}_{now}.html")
                    exp.save_to_file(lime_file)
                    print(f"Explicação LIME gerada com sucesso!")
                    
                    file_url = 'file://' + os.path.abspath(lime_file)
                    webbrowser.open(file_url)
                    
                except Exception as e:
                    print(f"Erro ao gerar LIME: {e}")

                print("\nDesconstruindo a decisão com gradientes integrados...")
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
                    attributions, delta = ig.attribute(
                        inputs=input_embeddings,
                        baselines=baseline_embeddings,
                        additional_forward_args=(attention_mask,),
                        target=int(predict_class),
                        return_convergence_delta=True
                    )

                    attributions_sum = attributions.sum(dim=-1).squeeze(0)
                    attributions_norm = attributions_sum / torch.norm(attributions_sum)
                    attributions_norm = attributions_norm.detach().cpu().numpy()

                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

                    score = float(attributions_sum.sum().detach().cpu().item())

                    vis_data_record = viz.VisualizationDataRecord(
                        word_attributions=attributions_norm,
                        pred_prob=float(probs[predict_class]),
                        pred_class=veredict,
                        true_class="Desconhecido",
                        attr_class=veredict,
                        attr_score=score,
                        raw_input_ids=tokens,
                        convergence_score=float(delta.item())
                    )

                    html_obj = viz.visualize_text([vis_data_record])
                    html_captum = html_obj.data
                    
                    captum_file = os.path.join(CAPTUM_OUTPUT_DIR, f"captum_{veredict}_{now}.html")
                    
                    with open(captum_file, "w", encoding="utf-8") as f:
                        f.write(html_captum)
                        
                    print(f"Gradientes integrados salvos em: {captum_file}")
                    webbrowser.open('file://' + os.path.abspath(captum_file))

                except Exception as e:
                    print(f"Erro ao gerar os gradientes integrados: {e}")

if __name__ == "__main__":
    main()