import os
import re
import chromadb
from optimum.intel import OVSentenceTransformer
# from sentence_transformers import SentenceTransformer

model = OVSentenceTransformer.from_pretrained("./models/bge-small-ov2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("insplad_docs")

DOCS_DIR = "./docs"
CLASS_FOLDERS = ["glass_insulator", "lightning_rod_suspension", "vari_grip"]

def parse_markdown_chunks(md_text: str, source_class: str, filename: str):
    chunks = []
    metadatas = []

    sections = re.split(r'\n(?=#{1,3} )', md_text.strip())

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) < 50:
            continue

        header_match = re.match(r'^#{1,3} (.+)', section)
        header = header_match.group(1) if header_match else filename

        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', section)  # bold
        clean = re.sub(r'__(.+?)__', r'\1', clean)         # bold
        clean = re.sub(r'`(.+?)`', r'\1', clean)           # inline code
        clean = re.sub(r'^#{1,3} ', '', clean, flags=re.MULTILINE)  # 헤더 #
        clean = re.sub(r'^\s*[-*]\s+', '', clean, flags=re.MULTILINE)  # 리스트
        clean = re.sub(r'\n{3,}', '\n\n', clean)           # 빈줄 정리
        clean = clean.strip()

        chunks.append(clean)
        metadatas.append({
            "class": source_class,
            "file": filename,
            "header": header,
        })

    return chunks, metadatas


all_chunks = []
all_metadatas = []
all_ids = []
chunk_idx = 0

for class_name in CLASS_FOLDERS:
    folder_path = os.path.join(DOCS_DIR, class_name)

    if not os.path.exists(folder_path):
        print(f"폴더 없음, 스킵: {folder_path}")
        continue

    md_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]
    print(f"\n[{class_name}] .md 파일 {len(md_files)}개 발견")

    for md_file in md_files:
        filepath = os.path.join(folder_path, md_file)
        text = open(filepath, encoding="utf-8").read()

        chunks, metadatas = parse_markdown_chunks(text, class_name, md_file)
        print(f"  {md_file} → {len(chunks)}개 청크")

        for chunk, meta in zip(chunks, metadatas):
            all_chunks.append(chunk)
            all_metadatas.append(meta)
            all_ids.append(f"chunk_{chunk_idx}")
            chunk_idx += 1

print(f"\n총 {len(all_chunks)}개 청크 임베딩 중...")

embeddings = model.encode(
    [f"Represent this sentence: {c}" for c in all_chunks],
    normalize_embeddings=True,
    show_progress_bar=True
).tolist()

collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadatas,
    ids=all_ids,
)

print(f"\n ChromaDB 저장 완료! 총 {len(all_chunks)}개 청크")