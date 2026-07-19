import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from spellchecker import SpellChecker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
INPUT_FILE_NAME = "extended_dataset"
INPUT_FILE = os.path.join(PROJECT_ROOT, "inputs", "datasets", f"{INPUT_FILE_NAME}.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{INPUT_FILE_NAME}_with_features.csv")

TOKEN_REGEX = re.compile(r"\b[a-zà-ÿ]+\b")
PUNCTUATION_REGEX = re.compile(r"[.,;:!?()\"'—-]")
spell = SpellChecker(language="pt")

def tokenize(text):
    return TOKEN_REGEX.findall(text.lower())

def get_burstiness(text):
    phrases = re.split(r'[.!?]+', text)
    sizes = [len(phrase.split()) for phrase in phrases if len(phrase.strip()) > 2]

    if len(sizes) <= 1:
        return 0.0

    mean_length = np.mean(sizes)

    if mean_length == 0:
        return 0.0

    return round(np.std(sizes) / mean_length, 4)


def get_mtld(words, ttr_threshold=0.72):
    if len(words) < 10:
        return 0.0

    def mtld_calc(tokens):
        factors = 0
        token_count = 0
        types = set()

        for token in tokens:
            token_count += 1
            types.add(token)

            current_ttr = len(types) / token_count

            if current_ttr <= ttr_threshold:
                factors += 1
                token_count = 0
                types.clear()

        if token_count > 0:
            current_ttr = len(types) / token_count

            excess = (1 - current_ttr) / (1 - ttr_threshold)

            factors += excess

        return len(tokens) / factors if factors > 0 else 0

    forward_mtld = mtld_calc(words)
    backward_mtld = mtld_calc(list(reversed(words)))

    return round((forward_mtld + backward_mtld) / 2, 4)

def get_spelling_features(words):
    if not words:
        return { "qtd_oov": 0, "taxa_oov": 0.0, "palavras_oov": [] }

    unknown_words = spell.unknown(words)
    oov_count = len(unknown_words)

    return {
        "qtd_oov": oov_count,
        "taxa_oov": round(oov_count / len(words), 4),
        "palavras_oov": list(unknown_words)
    }

def get_punctuation_features(text, words):
    if not words:
        return { "densidade_de_pontuacao": 0.0, "qtd_virgula": 0, "qtd_ponto_e_virgula": 0, "qtd_pontos": 0}

    punctuation_matches = PUNCTUATION_REGEX.findall(text)
    punctuation_count = len(punctuation_matches)

    comma_count = 0
    semicolon_count = 0
    dot_count = 0

    for char in punctuation_matches:
        if char == ",":
            comma_count += 1
        elif char == ";":
            semicolon_count += 1
        elif char == ".":
            dot_count += 1

    return {
        "densidade_de_pontuacao": round(punctuation_count / len(words), 4),
        "qtd_virgula": comma_count,
        "qtd_ponto_e_virgula": semicolon_count,
        "qtd_pontos": dot_count,
    }

def extract_features(text):
    words = tokenize(text)

    features = {
        "qtd_palavras": len(words),
        "burstiness": get_burstiness(text),
        "mtld": get_mtld(words),
    }

    features.update(get_spelling_features(words))

    features.update(get_punctuation_features(text, words))

    return features


if __name__ == "__main__":
    print(f"Lendo arquivo: {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    if "texto" not in df.columns or "classe" not in df.columns:
        raise ValueError("O CSV de entrada deve conter as colunas 'texto' e 'classe'.")

    processed_rows = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extraindo features"):
        text = str(row["texto"]).strip() if pd.notna(row["texto"]) else ""
        label = row["classe"]

        if len(text) < 10:
            continue

        features = extract_features(text)

        row_data = {
            "classe": label,
            "texto": text
        }
        
        row_data.update(features)
        
        if "palavras_oov" in row_data:
            row_data["palavras_oov"] = ", ".join(row_data["palavras_oov"])

        processed_rows.append(row_data)

    df_features = pd.DataFrame(processed_rows)
    df_features.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    
    print(f"Processamento concluído! {len(df_features)} textos processados.")
    print(f"Arquivo salvo em: {OUTPUT_FILE}")