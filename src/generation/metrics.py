import os
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "inputs", "cp_essays.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "cp_essays_metrics.csv")

def get_burstiness(text):
    phrases = re.split(r'[.!?]+', text)
    sizes = [len(phrase.split()) for phrase in phrases if len(phrase.strip()) > 2]
    if not sizes:
        return 0.0
    return round(np.std(sizes), 4)

def get_jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return round(intersection / union, 4)

def calculate_metrics():
    print(f"Lendo arquivo: {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    if 'texto_ia' not in df.columns or 'texto_copy_typed' not in df.columns:
        raise ValueError("O CSV deve conter 'texto_ia' e 'texto_copy_typed'.")

    df = df.reset_index(drop=True)

    texts_ia = df['texto_ia'].fillna("").astype(str).tolist()
    texts_copy = df['texto_copy_typed'].fillna("").astype(str).tolist()

    print("Inicializando vetorizador TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    print("Carregando modelo semântico... ")
    semantic_model = SentenceTransformer('BAAI/bge-m3')
    
    print("Calculando tensores de textos de IA...")
    embs_ia = semantic_model.encode(texts_ia, batch_size=16, convert_to_tensor=True, show_progress_bar=True)
    
    print("Calculando tensores de textos com copy-typing...")
    embs_copy = semantic_model.encode(texts_copy, batch_size=16, convert_to_tensor=True, show_progress_bar=True)

    # --- CÁLCULO SEMÂNTICO (EMBEDDINGS) ---
    semantic_sims_tensor = torch.nn.functional.cosine_similarity(embs_ia, embs_copy)
    semantic_sims_list = semantic_sims_tensor.cpu().tolist()

    len_ai_list, len_copy_list, size_ratios_list = [], [], []
    burst_ai_list, burst_copy_list = [], []
    jaccard_sim_list, lexical_sim_list, semantic_sim_list = [], [], []
    heuristic_labels_list = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        text_ia = str(row['texto_ia'])
        text_copy = str(row['texto_copy_typed'])
        
        if not text_ia.strip() or not text_copy.strip() or text_ia == 'nan' or text_copy == 'nan':
            for lst in [len_ai_list, len_copy_list, size_ratios_list, burst_ai_list, burst_copy_list, jaccard_sim_list, lexical_sim_list, semantic_sim_list]:
                lst.append(0.0)
            heuristic_labels_list.append("invalido")
            continue

        try:
            semantic_cos = round(semantic_sims_list[index], 4)
            
            len_ai = len(text_ia.split())
            len_copy = len(text_copy.split())
            size_ratio = round((len_copy / len_ai), 4) if len_ai > 0 else 0.0
            
            burst_ai = get_burstiness(text_ia)
            burst_copy = get_burstiness(text_copy)
            
            jaccard_sim = get_jaccard_similarity(text_ia, text_copy)
            
            # --- CÁLCULO LÉXICO (TF-IDF) ---
            tfidf_matrix = vectorizer.fit_transform([text_ia, text_copy])
            lexical_cos = round(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0], 4)

            # --- HEURÍSTICA DE CLASSIFICAÇÃO ---
            heuristic_label = "reescrito"
            if semantic_cos > 0.97:
                heuristic_label = "quase_identico"
            elif semantic_cos > 0.85 and lexical_cos > 0.75 and 0.8 <= size_ratio <= 1.2:
                heuristic_label = "copy_typing_forte"
            elif semantic_cos > 0.80:
                heuristic_label = "copy_typing_leve"
            elif semantic_cos > 0.70:
                heuristic_label = "parafraseado"

            semantic_sim_list.append(semantic_cos)
            len_ai_list.append(len_ai)
            len_copy_list.append(len_copy)
            size_ratios_list.append(size_ratio)
            burst_ai_list.append(burst_ai)
            burst_copy_list.append(burst_copy)
            jaccard_sim_list.append(jaccard_sim)
            lexical_sim_list.append(lexical_cos)
            heuristic_labels_list.append(heuristic_label)

        except Exception as e:
            for lst in [semantic_sim_list, len_ai_list, len_copy_list, size_ratios_list, burst_ai_list, burst_copy_list, jaccard_sim_list, lexical_sim_list]:
                lst.append(0.0)
            heuristic_labels_list.append("erro_calculo")

    df['similaridade_lexica'] = lexical_sim_list
    df['similaridade_semantica'] = semantic_sim_list
    df['similaridade_jaccard'] = jaccard_sim_list
    df['burstiness_ai'] = burst_ai_list
    df['burstiness_copy'] = burst_copy_list
    df['num_palavras_ai'] = len_ai_list
    df['num_palavras_copy'] = len_copy_list
    df['num_palavras_razao'] = size_ratios_list
    df['classificacao_heuristica'] = heuristic_labels_list
    
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nArquivo salvo com sucesso em: {OUTPUT_FILE}")
    print(f"-> Média léxica do lote: {df['similaridade_lexica'].mean():.4f}")
    print(f"-> Média semântica do lote: {df['similaridade_semantica'].mean():.4f}")

if __name__ == "__main__":
    calculate_metrics() 