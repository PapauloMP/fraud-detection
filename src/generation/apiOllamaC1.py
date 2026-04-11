import os
import time
import pandas as pd
from tqdm import tqdm
from ollama import Client
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
INPUT_FILE = os.path.join(BASE_DIR, "temas.csv")
PROMPTS_FILE = os.path.join(BASE_DIR, "prompts.csv")

load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("OLLAMA_API_KEY")
if not api_key:
    raise ValueError("OLLAMA_API_KEY não encontrada no .env")

client = Client(
    host = "https://ollama.com",
    headers = {'Authorization': f"Bearer {api_key}"}
)

def normalize_model_name(model_name: str) -> str:
    """
    Converte o nome do modelo em formato seguro para filename.
    Ex: 'deepseek-v3.1:671b-cloud' -> 'deepseek_v3_1_671b_cloud'
    """
    import re

    sanitized = re.sub(r'[^a-zA-Z0-9]+', '_', model_name)
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_').lower()


OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "output")
DATASET_DIR = os.path.join(OUTPUT_BASE_DIR, "datasets - class 1")
os.makedirs(DATASET_DIR, exist_ok=True)

MODELS = ["deepseek-v3.1:671b-cloud", "qwen3.5:397b-cloud", "gpt-oss:120b-cloud", "kimi-k2.5:cloud", "gemini-3-flash-preview:cloud"]
MODEL = MODELS[1]
OUTPUT_FILE = os.path.join(DATASET_DIR, f"dataset_ia_{normalize_model_name(MODEL)}.csv")

def load_prompts(path):
    df = pd.read_csv(path)

    if 'perfil' not in df.columns or 'prompt' not in df.columns:
        raise ValueError("prompts.csv deve conter colunas 'perfil' e 'prompt'")

    return dict(zip(df['perfil'], df['prompt']))


def call_ollama(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat(
                model = MODEL,
                messages = [{ "role": "user", "content": prompt }],
                options = { "temperature": 0.7 }
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"\n[Erro Ollama - Tentativa {attempt + 1}/{retries}] {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                return None


def main():
    print(f"Tentando abrir: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE).iloc[24:42]
    prompts_map = load_prompts(PROMPTS_FILE)

    results = []

    print(f"Iniciando geração com {MODEL} via Ollama...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Temas"):
        subject = row['tema']

        for profile, template in prompts_map.items():
            
            if "{tema}" not in template:
                raise ValueError(f"Prompt do perfil '{profile}' não contém '{{tema}}'")

            prompt = template.format(tema=subject)
            essay = call_ollama(prompt)

            results.append({
                "tema": subject,
                "perfil_prompt": profile,
                "texto_ia": essay,
                "fonte_llm": f"Ollama_{MODEL}",
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nConcluído: {OUTPUT_FILE} ({len(df_results)} textos)")

if __name__ == "__main__":
    main()