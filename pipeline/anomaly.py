import onnxruntime as ort
import streamlit as st
import numpy as np
import cv2
from PIL import Image

@st.cache_resource
def load_anomaly_model():
    sessions = {
        i: ort.InferenceSession(f"models/anomaly_class{i}.onnx")
        for i in range(3)
    }
    return sessions

def run_anomaly(image: Image.Image, class_id: int):
    threshold = {0: 0.55, 1:0.45, 2:0.72}
    sessions = load_anomaly_model()
    session = sessions[class_id]
    input_name = session.get_inputs()[0].name

    img = image.resize((384, 384))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


    outputs = session.run(None, {input_name: arr})

    anomaly_score = float(outputs[0][0]) 
    
    score_map = outputs[2][0, 0]

    # 히트맵 생성
    heatmap_norm = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (384, 384))

    # 오버레이 생성
    original = np.array(image.resize((384, 384)))
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
    overlay_pil = Image.fromarray(overlay)

    is_anomaly = anomaly_score > threshold[class_id]  # 임계값 조정 필요

    return anomaly_score, is_anomaly, heatmap_pil, overlay_pil


if __name__ == "__main__":
    from PIL import Image

    img = Image.open("vari_grip_test_04.jpg").convert("RGB")
    class_id = 1 # 테스트할 클래스 ID


    score, is_anomaly, heatmap, overlay = run_anomaly(img, class_id)

    # 결과 출력
    print(f"Anomaly Score : {score:.4f}")
    print(f"이상 여부     : {'이상' if is_anomaly else '정상'}")

    # 이미지 저장해서 눈으로 확인
    heatmap.save("test_heatmap.png")
    overlay.save("test_overlay_lightning.png")
    print("test_heatmap.png, test_overlay.png 저장 완료 → 눈으로 확인!")