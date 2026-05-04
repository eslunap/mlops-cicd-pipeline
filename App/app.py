import gradio as gr
import joblib
import numpy as np
from pathlib import Path

# Ruta del modelo.
# En Hugging Face, la carpeta Model quedará al mismo nivel que App.
MODEL_PATH = Path(__file__).resolve().parent / "Model" / "model.pkl"

model = joblib.load(MODEL_PATH)

target_names = ["setosa", "versicolor", "virginica"]

def predict(sepal_length, sepal_width, petal_length, petal_width):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)[0]
    return target_names[prediction]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal length"),
        gr.Number(label="Sepal width"),
        gr.Number(label="Petal length"),
        gr.Number(label="Petal width"),
    ],
    outputs=gr.Textbox(label="Predicted class"),
    title="Iris Classification App",
    description="Aplicación de clasificación usando un modelo entrenado con scikit-learn.",
)

if __name__ == "__main__":
    demo.launch()