from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Crear carpeta del modelo si no existe
os.makedirs("Model", exist_ok=True)

# Cargar dataset Iris
data = load_iris()
X = data.data
y = data.target

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Crear pipeline de scikit-learn
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

# Entrenar modelo
pipeline.fit(X_train, y_train)

# Guardar modelo entrenado y datos de prueba
joblib.dump(pipeline, "Model/model.pkl")
joblib.dump((X_test, y_test), "Model/test_data.pkl")

print("Modelo entrenado y guardado correctamente.")