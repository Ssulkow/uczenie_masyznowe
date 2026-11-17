
from pathlib import Path
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

# Pobranie niezmienionego archiwum z oficjalnymi podziałami train/test.
DATA_DIR = Path("data/zadanie-1")
DATA_DIR.mkdir(parents=True, exist_ok=True)

archive_path = hf_hub_download(
    repo_id="AI-MED-AGH/Recruitment-Task-1",
    filename="original/pen+based+recognition+of+handwritten+digits.zip",
    repo_type="dataset",
    local_dir=DATA_DIR,
)
with ZipFile(archive_path) as archive:
    archive.extractall(DATA_DIR / "pendigits")
print(f"Dane zapisano w: {DATA_DIR / 'pendigits'}")
plik1 = DATA_DIR / "pendigits" / "pendigits.tra"
plik2 = DATA_DIR / "pendigits"/"pendigits.tes"

print("========================")
print("zbior treningowy")
print("=======================")
df_train = pd.read_csv(plik1,header=None)
print(f"liczba pustych lini: {df_train.isnull().sum().sum()}")
print(f"liczba zduplikowanych lini: {df_train.duplicated().sum()}")
print(f"opis danych:\n {df_train.describe()}")
print("========================")
print("zbior testowy")
print("=======================")
df_test = pd.read_csv(plik2,header=None)
print(f"liczba pustych lini: {df_test.isnull().sum().sum()}")
print(f"liczba zduplikowanych lini: {df_test.duplicated().sum()}")
print(f"opis danych:\n {df_test.describe()}")
models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Random Forest": Pipeline([
        ("scaler",StandardScaler()),
        ("clf",RandomForestClassifier())
    ]),
    "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)) 
    ])
    
}

X_train = df_train.iloc[:,0:16]
Y_train = df_train.iloc[:,16]
X_test = df_test.iloc[:,0:16]
Y_test = df_test.iloc[:,16]
i = 1 
results = []
dane = []
for name,model in models.items():
    model.fit(X_train,Y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(Y_test,pred)
    f1 = f1_score(Y_test,pred,average="macro")
    dane.append ({
        "Name": name,
        "Accuracy": acc,
        "f1": f1
    })

    plt.subplot(1,3,i)
    print("===================")
    print(f"{name}")
    print("===================")
    print(classification_report(Y_test,pred))
    cm = confusion_matrix(Y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("przewidywana cyfra")
    plt.ylabel("odpowiedz modelu")
    plt.grid()
    i+=1
df_dane = pd.DataFrame(dane)
print(df_dane)
plt.show()
print("klasyfikacja")
