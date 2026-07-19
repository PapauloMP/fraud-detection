import os
import time
import pandas as pd
from tqdm import tqdm
from ollama import Client
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
THEMES_FILE = os.path.join(PROJECT_ROOT, "inputs", "temas.csv")
PROMPTS_FILE = os.path.join(PROJECT_ROOT, "inputs", "prompts.csv")

load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("OLLAMA_API_KEY")
if not api_key:
    raise ValueError("OLLAMA_API_KEY não encontrada no .env")

client = Client(
    host = "https://ollama.com",
    headers = {'Authorization': f"Bearer {api_key}"},
    timeout = 180.0
)

def normalize_model_name(model_name: str) -> str:
    import re

    sanitized = re.sub(r'[^a-zA-Z0-9]+', '_', model_name)
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_').lower()


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATASET_DIR = os.path.join(OUTPUT_DIR, "datasets - class 2")
os.makedirs(DATASET_DIR, exist_ok=True)

MODELS = ["deepseek-v3.1:671b-cloud", "qwen3.5:397b-cloud", "gpt-oss:120b-cloud", "kimi-k2.5:cloud", "gemini-3-flash-preview:cloud"]
MODEL = MODELS[4]
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
    print(f"Tentando abrir: {THEMES_FILE}")

    df = pd.read_csv(THEMES_FILE).iloc[0:1]
    prompts_map = load_prompts(PROMPTS_FILE)

    target_profile = "specialist"
    if target_profile not in prompts_map:
        raise KeyError(f"O perfil '{target_profile}' não foi encontrado no arquivo {PROMPTS_FILE}")

    template = prompts_map[target_profile]

    if "{tema}" not in template:
        raise ValueError(f"Prompt do perfil '{target_profile}' não contém a tag '{{tema}}'")

    results = []

    print(f"Iniciando geração com {MODEL} via Ollama (Perfil restrito: {target_profile})...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Temas"):
        subject = row['tema']

        prompt = template.format(tema=subject)
        essay = call_ollama(prompt)

        results.append({
            "tema": subject,
            "perfil_prompt": target_profile,
            "texto_ia": essay,
            "fonte_llm": f"Ollama_{MODEL}",
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nConcluído: {OUTPUT_FILE} ({len(df_results)} textos)")

if __name__ == "__main__":
    main()