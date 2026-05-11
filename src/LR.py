import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

COMPILED_ESSAYS_WITH_METRICS_FILE = "generation/inputs/compiled_essays_with_metrics.csv"
print(f"Lendo base de dados: {COMPILED_ESSAYS_WITH_METRICS_FILE}...")
df = pd.read_csv(COMPILED_ESSAYS_WITH_METRICS_FILE)

class_column = 'classe' 

if class_column not in df.columns:
    print(f"ERRO: Coluna '{class_column}' não encontrada.")
    exit()

features = [
    'similaridade_lexica', 
    'similaridade_semantica', 
    'similaridade_jaccard',
    'burstiness_ai', 
    'burstiness_copy', 
    'num_palavras_razao'
]

available_features = [f for f in features if f in df.columns]

X = df[available_features]
y = df[class_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTreinando Regressão Logística (Multinomial)...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"ACURÁCIA GERAL: {acc * 100:.2f}%")
print(f"{'='*40}")
print("\nRelatório de Classificação (Precision / Recall / F1-Score):")
print(classification_report(y_test, y_pred))

print(f"\n{'='*40}")
print("IMPORTÂNCIA DAS VARIÁVEIS (Coeficientes Absolutos)")
print(f"{'='*40}")

weights = np.mean(np.abs(model.coef_), axis=0)
feat_weights = pd.DataFrame({'Métrica': available_features, 'Peso': weights})
feat_weights = feat_weights.sort_values(by='Peso', ascending=False)
for index, row in feat_weights.iterrows():
    print(f"-> {row['Métrica']:<25}: {row['Peso']:.4f}")

class_names = ['Humano', 'IA', 'Copy-Typing']
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão - Regressão Logística')
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Prevista')
plt.tight_layout()
plt.savefig('matriz_confusao_logistica.png')
print("\nGráfico da matriz de confusão salvo como 'matriz_confusao_logistica.png'.")