import argparse

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# El servidor MLflow se elige con la variable de entorno MLFLOW_TRACKING_URI
# (p.ej. http://127.0.0.1:5000 en local o https://dagshub.com/<user>/<repo>.mlflow)
parser = argparse.ArgumentParser()
parser.add_argument("--n-estimators", type=int, default=100)
args = parser.parse_args()
n_estimators = args.n_estimators

# Cargar el conjunto de datos
iris = datasets.load_iris()
X = iris.data
y = iris.target

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
    mlflow.log_metric("accuracy", accuracy)

    print(f"Modelo entrenado y precisión: {accuracy:.4f}")
    print("Experimento registrado con MLflow.")
