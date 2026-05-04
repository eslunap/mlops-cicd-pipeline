from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

# Crear carpeta de resultados si no existe
os.makedirs("Results", exist_ok=True)

# Cargar modelo y datos de prueba
model = joblib.load("Model/model.pkl")
X_test, y_test = joblib.load("Model/test_data.pkl")

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Guardar métricas en archivo de texto
with open("Results/metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# Guardar matriz de confusión como imagen
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.savefig("Results/model_results.png")
plt.close()

print("Evaluación finalizada. Métricas e imagen guardadas en Results.")