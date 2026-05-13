import requests, json, os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPEN_AI_API_KEY")

def generate_report(class_name, anomaly_score, is_anomaly, context_docs=None, model="gpt-4o-mini"):
    url = f"https://bridge.luxiacloud.com/llm/openai/chat/completions/{model}/create"

    headers = {
        "apikey": API_KEY,
        "Content-Type": "application/json"
    }
    status = "ANOMALY DETECTED" if is_anomaly else "NORMAL"

    if context_docs:
            context = "\n\n".join(context_docs)
            prompt = f"""You are an industrial inspection expert. Your task is to provide a HIGHLY RELEVANT and technical report based on the insPLAD dataset results.

[Reference Context]
{context}

[Inspection Data]
- Component: {class_name}
- Anomaly Score: {anomaly_score:.4f}
- Detection Status: {status}

[Objective]
Based on the data above, generate a report that DIRECTLY addresses the inspection findings. Ensure the answer is dense with relevant information and avoids generic disclaimers.

[Task Instructions]
1. Focus on the relationship between the {class_name} and the {status} status.
2. Integrate the Reference Context naturally into the analysis to explain the {anomaly_score:.4f} score.
3. If specific details are missing in the context, use your expert reasoning to maintain a professional report flow that remains consistent with the {class_name} inspection.
4. Keep the response concise, technical, and strictly focused on the user's implicit question: "What does this inspection result mean?"

[Report Structure]
1. Summary: A direct overview of the {class_name}'s current state.
2. Root Cause Analysis: Link the {status} status to potential technical failures mentioned in the context.
3. Recommendations: Specific, actionable steps for a {class_name} with an anomaly score of {anomaly_score:.4f}.
4. Reference Standards: List the technical guidelines utilized from the context.

Tone: Professional, Direct, and Highly Relevant."""
    else:
        # RAG 없음 — 모델 자체 지식 사용
        prompt = f"""You are an industrial inspection expert analyzing insPLAD dataset results.

Inspection result data:
- Component class: {class_name}
- Anomaly score: {anomaly_score:.4f}
- Status: {status}

[Report Structure]
Generate a structured inspection report in English:
1. Summary: Brief overview of the current inspection state.
2. Root cause analysis: Identify potential causes based on your expert knowledge.
3. Recommendations: Technical actions to address the findings.
4. Reference standards: Cite relevant industry standards if applicable.

Be concise and technical."""

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False 
    }  

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            result_json = response.json()
            # Luxia 응답 구조가 OpenAI 표준과 동일하다면 아래와 같이 추출
            return result_json['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"An exception occurred: {str(e)}"


# if __name__ == "__main__":

#     test_contexts = [
#         "Glass insulators are used in high-voltage transmission lines to support conductors.",
#         "Common defects include surface cracks, contamination, and partial discharge.",
#         "Anomaly detection threshold is typically set at 0.5 for PatchCore models.",
#     ]

#     report = generate_report(
#         class_name="glass-insulator",
#         anomaly_score=0.823,
#         is_anomaly=True,
#         context_docs=test_contexts
#     )

#     print("=" * 60)
#     print(report)
#     print("=" * 60)