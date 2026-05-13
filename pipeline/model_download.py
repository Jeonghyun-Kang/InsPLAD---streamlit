import os
from optimum.intel import OVSentenceTransformer

save_path = "./models/bge-small-ov2"
model_id = "BAAI/bge-small-en-v1.5"
# model_id = "./models/bge-small-local"
os.makedirs("./models", exist_ok=True)

model = OVSentenceTransformer.from_pretrained(model_id, export=True)
model.save_pretrained(save_path)

print("OpenVINO 모델 저장 완료!")