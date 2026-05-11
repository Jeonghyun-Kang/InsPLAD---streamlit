import chromadb
import streamlit as st
from optimum.intel import OVSentenceTransformer
# from sentence_transformers import SentenceTransformer

@st.cache_resource
def model_load():
    return OVSentenceTransformer.from_pretrained("./models/bge-small-ov2") 

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection("insplad_docs")

def search(query: str, class_name: str = None, top_k: int = 3):
    collection = get_collection()
    model = model_load()
    embedding = model.encode(
        [f"Represent this sentence: {query}"],
        normalize_embeddings=True
    ).tolist()

    where = {"class": class_name} if class_name else None

    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        where=where  # 필터
    )
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
#         print(f"현재 컬렉션의 데이터 개수: {collection.count()}")
#         print(f"\n쿼리: '{query}' (클래스: {class_name})")
#         print("-" * 50)
#         results = search(query, class_name=class_name, top_k=2)
#         for i, doc in enumerate(results):
#             print(f"[{i+1}] {doc[:200]}...")