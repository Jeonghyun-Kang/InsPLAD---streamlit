# insPLAD Streamlit Anomaly Detection
This project implements a web-based AI inspection workflow for power equipment anomaly detection using image classification, class-specific anomaly detection, RAG-based report generation, and cloud storage.
The system takes an uploaded equipment image, identifies the equipment class, detects potential anomalies, visualizes the result with heatmaps and overlays, retrieves relevant domain knowledge, and generates an inspection report.
---
## Project Overview
Power equipment inspection traditionally relies on manual visual inspection, which can be costly, time-consuming, and risky in high-voltage or high-altitude environments.
This project aims to support safer and more scalable inspection by combining:
- Equipment classification
- Class-specific anomaly detection
- Heatmap and overlay visualization
- Retrieval-Augmented Generation-based inspection report generation
- S3 and DynamoDB-based result storage
---
## Pipeline
```text
User Image Upload
        ↓
Equipment Classifier
        ↓
Class-specific PatchCore Anomaly Detection
        ↓
Heatmap / Overlay Generation
        ↓
ChromaDB Document Retrieval
        ↓
LLM-based Inspection Report Generation
        ↓
S3 + DynamoDB Storage

⸻

Main Features

* Equipment Classification
    A lightweight classifier identifies the target equipment class from the uploaded image.
* Class-specific Anomaly Detection
    PatchCore ONNX models are used for anomaly detection on each equipment class.
* Visual Explanation
    The system generates anomaly heatmaps and overlay images to highlight suspicious regions.
* RAG-based Report Generation
    Relevant domain documents are retrieved from ChromaDB and passed to an LLM to generate a structured inspection report.
* Cloud Storage
    Original images, heatmaps, overlays, and reports are stored in S3, while metadata is logged in DynamoDB.

⸻

Target Classes

The system supports the following power equipment classes:

glass-insulator
lightning-rod-suspension
vari-grip

⸻

Demo Results

Glass Insulator

Lightning-Rod Suspension

Vari-Grip

⸻

Repository Structure

insPLAD---streamlit/
├── app.py
├── requirement.txt
├── pipelines/
│   ├── classify.py
│   ├── anomaly.py
│   ├── rag.py
│   ├── llm.py
│   ├── storage.py
│   ├── evaluation.py
│   ├── model_download.py
│   └── model_tokenizer.py
├── scripts/
│   └── build_vectordb.py
├── docs/
├── assets/
├── .env.example
├── .gitignore
└── README.md

⸻

Required Model Files

Model files are not included in this repository due to file size.

Before running the app, place the ONNX model files under the models/ directory.

models/
├── classifier.onnx
├── anomaly_class0.onnx
├── anomaly_class1.onnx
└── anomaly_class2.onnx

Expected class mapping:

class 0: glass-insulator
class 1: lightning-rod-suspension
class 2: vari-grip

⸻

Environment Variables

Create a .env file in the project root.

You can refer to .env.example.

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=ap-northeast-2
S3_BUCKET_NAME=
DYNAMODB_TABLE_NAME=insplad_log
LUXIA_API_KEY=
OPENAI_API_KEY=
OPEN_AI_API_KEY=
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

.env must not be uploaded to GitHub.

⸻

AWS Setup

This project uses AWS S3 and DynamoDB for result storage.

S3

Create an S3 bucket.

Example:

insplad-anomaly-detection-<account-id>-ap-northeast-2

The app stores:

images/{inspection_id}/original.png
images/{inspection_id}/heatmap.png
images/{inspection_id}/overlay.png
reports/{inspection_id}.txt

DynamoDB

Create a DynamoDB table:

Table name: insplad_log
Partition key: inspection_id
Type: String

⸻

Installation

Create and activate a virtual environment.

python3 -m venv .venv
source .venv/bin/activate

Install dependencies.

pip install -r requirement.txt

⸻

Build Vector Database

Before running the Streamlit app, build the ChromaDB vector database.

python scripts/build_vectordb.py

This creates a local chroma_db/ directory.

The vector database is not included in GitHub because it can be regenerated from the source documents.

⸻

Run the App

streamlit run app.py

Then open the local URL shown in the terminal.

Example:

http://localhost:8501

⸻

Notes on Embedding Model

The project uses BGE-small for document retrieval.

BAAI/bge-small-en-v1.5

Depending on the local setup, the model may be downloaded during the first run. After the first download, it will be cached locally.

⸻

Notes on Model Threshold

The anomaly detection threshold is managed separately from the ONNX model.

For example, the lightning-rod-suspension PatchCore model uses a threshold calculated from test-set anomaly scores.

if anomaly_score >= threshold:
    abnormal
else:
    normal

The threshold should be used with the ONNX model exported from the same checkpoint.

⸻

Technologies Used

* Python
* Streamlit
* ONNX Runtime
* PatchCore
* MobileNet-style classifier
* ChromaDB
* Sentence Transformers
* Luxia / OpenAI-compatible LLM API
* AWS S3
* AWS DynamoDB
* Boto3
* RAGAS

⸻

Security Notice

The following files and directories are intentionally excluded from GitHub:

.env
.venv/
models/
chroma_db/
data/
dataset/
insplad/
outputs/
*.onnx
*.ckpt
*.pth

Do not upload API keys, AWS credentials, or model files directly to a public repository.

⸻

Authors

* Jeonghyun Kang
* Eui Seung Jeon