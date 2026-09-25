import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# El servidor MLflow se elige con la variable de entorno MLFLOW_TRACKING_URI
# (p.ej. http://127.0.0.1:5000 en local o https://dagshub.com/<user>/<repo>.mlflow)
parser = argparse.ArgumentParser()
parser.add_argument("--n-estimators", type=int, default=100)
args = parser.parse_args()
n_estimators = args.n_estimators

# Cargar el conjunto de datos desde el archivo CSV (versionado con DVC)
try:
    iris = pd.read_csv('data/iris_dataset.csv')
except FileNotFoundError:
    print("Error: El archivo 'data/iris_dataset.csv' no fue encontrado.")
    sys.exit(1)

# Dividir el DataFrame en características (X) y etiquetas (y)
X = iris.drop('target', axis=1)
y = iris['target']

mlflow.set_experiment("iris-random-forest")

# Iniciar un experimento de MLflow
with mlflow.start_run():
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Inicializar y entrenar el modelo
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Realizar predicciones y calcular la precisión
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Guardar el modelo entrenado en un archivo .pkl
    joblib.dump(model, 'model.pkl')

    # Registrar el modelo con MLflow
    mlflow.sklearn.log_model(
        model,
        name="random-forest-model",
        # MLflow >=3.x serializa con skops; el árbol lo generamos nosotros, es seguro
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # Registrar parámetros y métricas
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("n_samples", len(iris))
    mlflow.log_metric("accuracy", accuracy)

    # Matriz de confusión como artefacto
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/confusion_matrix.png")
    plt.close(fig)
    mlflow.log_artifact("outputs/confusion_matrix.png")

    print(f"Modelo entrenado y precisión: {accuracy:.4f}")
    print("Experimento registrado con MLflow.")
