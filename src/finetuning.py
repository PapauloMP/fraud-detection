import os
from pyexpat import model
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

MODEL_NAME = "neuralmind/bert-base-portuguese-cased" #"PORTULAN/albertina-100m-portuguese-ptbr-encoder" 
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5 #LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def forward(self, input_ids, attention_mask, metrics):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # cls = outputs.last_hidden_state[:, 0, :]  # CLS token - SIMPLIFICAÇÃO (USA APENAS O CLS QUE É O CONTEXTO GERAL)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        # Usar todos os vetores da saída para um novo classificador 

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

                if score > best_score:
                    best_score = score
                    best_thresholds = thresholds

    return best_thresholds, best_score    

def entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-12), axis=1) # Entropia de Shannon

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in tqdm(loader):
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

    logits_all = [], probs_all = []

    preds, true = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            metrics = batch["metrics"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask, metrics)

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
    report = classification_report(true, preds, digits=4)
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

def main():
    df = pd.read_csv("dataset.csv")

    texts = df["texto_copy_typed"].fillna("").tolist()

    label_map = {"humano": 0, "ia": 1, "copy": 2}
    labels = df["classe"].map(label_map).values

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = EssayDataset(X_train, y_train, tokenizer)
    val_dataset = EssayDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BERTClassifier(MODEL_NAME, num_classes=3)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience = 2
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train(model, train_loader, optimizer, criterion)
        print(f"Train loss: {train_loss:.4f}")
        
        eval_results = evaluate(model, val_loader, criterion)

        print(f"Val loss: {eval_results['loss']:.4f}")
        print(eval_results["report"])
        
        print("Confusion Matrix:")
        print(eval_results["confusion_matrix"])

        # TEMPERATURA SCALING
        logits = eval_results["logits"].to(DEVICE)
        labels_tensor = torch.tensor(eval_results["true"]).to(DEVICE)

        scaler = TemperatureScaler()
        scaler.fit(logits, labels_tensor)

        calibrated_logits = scaler(logits)
        probs_calibrated = torch.softmax(calibrated_logits, dim=1).cpu().numpy()

        # THRESHOLD TUNING
        thresholds, best_score = tune_thresholds(probs_calibrated, eval_results["true"])

        print("\nBest thresholds:", thresholds)
        print("Best accuracy (threshold tuning):", best_score)

        preds_threshold = apply_thresholds(probs_calibrated, thresholds)

        print("\nReport com threshold tuning:")
        print(classification_report(eval_results["true"], preds_threshold, digits=4))

        # UNCERTAINTY ANALYSIS
        confidence = np.max(probs_calibrated, axis=1)
        ent = entropy(probs_calibrated)

        print("\nConfiança média:", confidence.mean())
        print("Entropia média:", ent.mean())

        wrong = preds_threshold != eval_results["true"]
        if np.any(wrong):
            print("Confiança média nos erros:", confidence[wrong].mean())

        # SAVE ANALYSIS
        df_analysis = pd.DataFrame({
            "true": eval_results["true"],
            "pred_base": eval_results["preds"],
            "pred_threshold": preds_threshold,
            "confidence": confidence,
            "entropy": ent
        })

        df_analysis.to_csv("analise_validacao.csv", index=False)

        eval_results = evaluate(model, val_loader, criterion)

        print(f"Val loss: {eval_results['loss']:.4f}")
        print(eval_results["report"])

        print("Confusion Matrix:")
        print(eval_results["confusion_matrix"])

        val_loss = eval_results["loss"]

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model_no_metrics.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping.")
            break


if __name__ == "__main__":
    main()