import chromadb
import streamlit as st
from optimum.intel import OVModelForFeatureExtraction
from optimum.intel import OVSentenceTransformer

import torch
import numpy as np
from transformers import AutoTokenizer

class OVSentenceEncoder:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = OVModelForFeatureExtraction.from_pretrained(model_path)
    
    def encode(self, sentences, batch_size=32, normalize_embeddings=True, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.numpy())
        
        return np.vstack(all_embeddings)

@st.cache_resource
def model_load():
    return OVSentenceEncoder("./models/bge-small-ov")

@st.cache_resource
def get_chroma():
    return chromadb.PersistentClient(path="./chroma_db")

@st.cache_resource
def get_collection():
    client = get_chroma()
    return client.get_collection("insplad_docs")

def search(query: str, class_name: str = None, top_k: int = 3):
    with st.status("전력 설비 데이터 분석 중...", expanded=True) as status:
        st.write("0. 모델 준비 중...")
        collection = get_collection()
        model = model_load()

        st.write("1. 텍스트 임베딩 생성중... (BGE-small)")
        embedding = model.encode(
            [f"Represent this sentence: {query}"],
            normalize_embeddings=True,
        ).tolist()

        st.write("2. chromaDB 내 이상 징후 검색 중")

        where = {"class": class_name} if class_name else None

        results = collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            where=where  # 필터
        )
        st.write("3. 기술 문서 추출 중...")

        status.update(label="분석 완료!", state="complete", expanded=False)
    return results["documents"][0]  


# if __name__ == "__main__":

#     test_queries = [
#         ("glass_insulator anomaly defect",     "glass_insulator"),
#         ("lightning rod suspension inspection", "lightning_rod_suspension"),
#         ("vari_grip normal condition",          "vari_grip"),
#     ]

#     # sample = collection.get(limit=1)
#     # print(f"저장된 메타데이터 예시: {sample['metadatas']}")

#     for query, class_name in test_queries:
#         # print(f"현재 컬렉션의 데이터 개수: {collection.count()}")
#         print(f"\n쿼리: '{query}' (클래스: {class_name})")
#         print("-" * 50)
#         results = search(query, class_name=class_name, top_k=2)
#         for i, doc in enumerate(results):
#             print(f"[{i+1}] {doc[:200]}...")