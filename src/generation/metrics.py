import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "inputs", "cp_essays.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "cp_essays_metrics.csv")

def calculate_metrics():
    print(f"Lendo arquivo: {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    if 'texto_ia' not in df.columns or 'texto_copy_typed' not in df.columns:
        raise ValueError("O CSV deve conter 'texto_ia' e 'texto_copy_typed'.")

    print("Inicializando vetorizador TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    
    print("Carregando modelo semântico... ")
    semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    lexical_sim = []
    semantic_sim = []
    len_ai = []
    len_copy = []
    size_ratios = []

    print("Calculando dupla métrica linha a linha...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        text_ia = str(row['texto_ia'])
        text_copy = str(row['texto_copy_typed'])
        
        if not text_ia.strip() or not text_copy.strip() or text_ia == 'nan' or text_copy == 'nan':
            lexical_sim.append(0.0)
            semantic_sim.append(0.0)
            continue
            
        try:
            len_ai.append(text_ia.str.len())
            len_copy.append(text_copy.str.len())
            size_ratio = (len_copy[-1] / len_ai[-1]) if len_ai[-1] > 0 else 0.0
            size_ratios.append(size_ratio)

            # --- CÁLCULO LÉXICO (TF-IDF) ---
            tfidf_matrix = vectorizer.fit_transform([text_ia, text_copy])
            lexical_cos = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            lexical_sim.append(round(lexical_cos, 4))
            
            # --- CÁLCULO SEMÂNTICO (EMBEDDINGS) ---
            emb_ia = semantic_model.encode(text_ia, convert_to_tensor=True)
            emb_copy = semantic_model.encode(text_copy, convert_to_tensor=True)
            semantic_cos = util.cos_sim(emb_ia, emb_copy).item()
            semantic_sim.append(round(semantic_cos, 4))

            # --- HEURÍSTICA DE CLASSIFICAÇÃO ---
            heuristic_label = "reescrito"
            if semantic_cos > 0.97:
                return "quase_identico"
            elif semantic_cos > 0.85 and lexical_cos > 0.75 and 0.8 <= size_ratio <= 1.2:
                return "copy_typing_forte"
            elif semantic_cos > 0.80:
                return "copy_typing_leve"
            elif semantic_cos > 0.70:
                return "parafraseado"
                
        except Exception as e:
            lexical_sim.append(0.0)
            semantic_sim.append(0.0)

    df['similaridade_lexica'] = lexical_sim
    df['similaridade_semantica'] = semantic_sim
    df['num_palavras_ai'] = len_ai
    df['num_palavras_copy'] = len_copy
    df['num_palavras_razao'] = size_ratios
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nArquivo salvo como: {OUTPUT_FILE}")
    print(f"-> Média léxica do lote: {df['similaridade_lexica'].mean():.4f}")
    print(f"-> Média semântica do lote: {df['similaridade_semantica'].mean():.4f}")

if __name__ == "__main__":
    calculate_metrics()