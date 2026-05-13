import onnxruntime as ort
import streamlit as st
import numpy as np
from PIL import Image

CLASS_NAMES = ["glass_insulator", "lightning_rod_suspension", "vari_grip"]

@st.cache_resource
def load_classfier_model():
    session = ort.InferenceSession("models/classifier.onnx")
    return session

def preprocess(image: Image.Image):
    img = image.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1, 3, 224, 224)


def classify(image: Image.Image):
    session = load_classfier_model()
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: preprocess(image)})[0]
    class_id = int(np.argmax(logits))
    return class_id, CLASS_NAMES[class_id]


# if __name__ == "__main__":
#     from PIL import Image

#     img = Image.open("test02.jpg").convert("RGB")

#     class_id, class_name = classify(img)

#     print(f"예측 클래스 ID : {class_id}")
#     print(f"예측 클래스명  : {class_name}")