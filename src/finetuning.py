import os
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

MODEL_NAME = "neuralmind/bert-base-portuguese-cased" #"PORTULAN/albertina-100m-portuguese-ptbr-encoder" 
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-5 #LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "generation/outputs"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_FILE = os.path.join(LOG_DIR, f"training_{timestamp}.log")
PLOT_FILE = os.path.join(PLOT_DIR, f"loss_curve_{timestamp}.png")
MODEL_FILE = os.path.join(MODEL_DIR, f"best_model_no_metrics_{timestamp}.pt")
ANALYSIS_FILE = os.path.join(ANALYSIS_DIR, f"validation_analysis_no_metrics_{timestamp}.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # cls = outputs.last_hidden_state[:, 0, :]  # CLS token - SIMPLIFICAÇÃO (USA APENAS O CLS QUE É O CONTEXTO GERAL)
        #pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        # Usar todos os vetores da saída para um novo classificador 

        # Masked mean pooling para ignorar os tokens de padding visto que foi fixado o tamanho máximo de 512 tokens
        token_embeddings = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        logits = self.classifier(pooled_output)

        return logits

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels):
        self.to(logits.device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
    
def apply_thresholds(probs, thresholds):
    preds = []

    for p in probs:
        passed = [i for i in range(len(p)) if p[i] >= thresholds[i]]

        if len(passed) == 1:
            preds.append(passed[0])
        else:
            preds.append(np.argmax(p))

    return np.array(preds)

def tune_thresholds(probs, y_true):
    best_thresholds = [0.5] * probs.shape[1]
    best_score = 0

    for t0 in np.linspace(0.3, 0.8, 6):
        for t1 in np.linspace(0.3, 0.8, 6):
            for t2 in np.linspace(0.3, 0.8, 6):
                thresholds = [t0, t1, t2]

                preds = apply_thresholds(probs, thresholds)
                score = (preds == y_true).mean()

                if score > best_score or (score == best_score and sum(thresholds) > sum(best_thresholds)):
                    best_score = score
                    best_thresholds = thresholds

    return best_thresholds, best_score    

def get_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-12), axis=1) # Entropia de Shannon

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training..."):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()

    logits_all, probs_all = [], []
 
    preds, true = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)

            predictions = torch.argmax(probs, dim=1)

            logits_all.append(logits.cpu())
            probs_all.append(probs.cpu())

            preds.extend(predictions.cpu().numpy())
            true.extend(labels.cpu().numpy())

    logits_all = torch.cat(logits_all, dim=0)
    probs_all = torch.cat(probs_all, dim=0).numpy()

    preds = np.array(preds)
    true = np.array(true)

    avg_loss = total_loss / len(loader)
    report = classification_report(true, preds, digits=4, zero_division=0)
    cm = confusion_matrix(true, preds)

    return {
        "loss": avg_loss,
        "logits": logits_all,
        "probs": probs_all,
        "preds": preds,
        "true": true,
        "report": report,
        "confusion_matrix": cm
    }

def save_loss_plot(train_losses, val_losses, output_path):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss vs Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()

def main():
    logger.info("Loading dataset...")
    df = pd.read_csv("generation/inputs/compiled_essays_with_metrics.csv")

    texts = df["texto"].fillna("").tolist()
    labels = df["classe"].values

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.5, stratify=labels, random_state=42
    )

    logger.info(f"Training segmentation - Train: {len(X_train)} | Validation: {len(X_val)}")
    logger.info(f"Total texts: {len(texts)}")
    logger.info(f"Class distribution: {np.bincount(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = EssayDataset(X_train, y_train, tokenizer)
    val_dataset = EssayDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        pin_memory=True,
    )

    model = BERTClassifier(MODEL_NAME, num_classes=3)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience = 2
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        logger.info("=" * 50)
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")


        train_loss = train(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        logger.info(f"Train loss: {train_loss:.4f}")

        eval_results = evaluate(model, val_loader, criterion)

        val_loss = eval_results["loss"]
        val_losses.append(val_loss)

        logger.info(f"Val loss: {val_loss:.4f}")
        logger.info("Classification Report:\n%s", eval_results["report"])
        logger.info("Confusion Matrix:\n%s", eval_results['confusion_matrix'])

        # TEMPERATURE SCALINGI
        logits = eval_results["logits"].to(DEVICE)
        labels_tensor = torch.tensor(eval_results["true"]).to(DEVICE)

        scaler = TemperatureScaler()
        scaler.fit(logits, labels_tensor)

        calibrated_logits = scaler(logits)
        probs_calibrated = torch.softmax(calibrated_logits, dim=1).detach().cpu().numpy()

        # THRESHOLD TUNING
        thresholds, best_score = tune_thresholds(probs_calibrated, eval_results["true"])

        logger.info(f"Best thresholds: {thresholds}")
        logger.info(f"Best accuracy (threshold tuning): {best_score}")
        preds_threshold = apply_thresholds(probs_calibrated, thresholds)

        logger.info("Report com threshold tuning:\n%s", classification_report(eval_results["true"], preds_threshold, digits=4, zero_division=0))

        # ANÁLISE DE INCERTEZA
        confidence = np.max(probs_calibrated, axis=1)
        entropy = get_entropy(probs_calibrated)

        logger.info(f"Mean confidence: {confidence.mean()}")
        logger.info(f"Mean entropy: {entropy.mean()}")

        wrong = preds_threshold != eval_results["true"]
        if np.any(wrong):
            logger.info(f"Mean confidence on errors: {confidence[wrong].mean()}")

        if val_loss < best_loss:
            best_loss = val_loss

            torch.save(model.state_dict(), MODEL_FILE)

            metadata = {
                "epoch": epoch,
                "loss": float(val_loss),
                "thresholds": [float(t) for t in thresholds],
                "temperature": float(scaler.temperature.item())
            }

            config_file = MODEL_FILE.replace(".pt", "_config.json")
            with open(config_file, "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=4)

            # SAVE ANALYSIS
            df_analysis = pd.DataFrame({
                "classe_real": eval_results["true"],
                "classe_predita": eval_results["preds"],
                "classe_predita_com_threshold": preds_threshold,
                "confianca": confidence,
                "entropia": entropy
            })

            df_analysis.to_csv(ANALYSIS_FILE, index=False)

            logger.info("New best model saved.")

            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping.")
            break

    save_loss_plot(train_losses, val_losses, PLOT_FILE)

    logger.info(f"Plot saved to: {PLOT_FILE}")
    logger.info(f"Log saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()