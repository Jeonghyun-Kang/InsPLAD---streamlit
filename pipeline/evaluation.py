from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

from langchain_openai import ChatOpenAI
# from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import BaseRagasLLM
# from ragas.llms.base import PromptValue
import requests, os
import streamlit as st
from dotenv import load_dotenv
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

load_dotenv()
API_KEY = os.getenv("OPEN_AI_API_KEY")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

class LuxiaRagasLLM(BaseRagasLLM):
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.api_key = api_key
        self.url = f"https://bridge.luxiacloud.com/llm/openai/chat/completions/{model}/create"
        self.model = model

    def is_finished(self, *args):
        return True
    
    def set_usage(self, usage, *args):
        pass

    def generate_text(self, prompt, **kwargs):
        headers = {"apikey": self.api_key, "Content-Type": "application/json"}
        prompt_str = prompt.to_string() if hasattr(prompt, 'to_string') else str(prompt)
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt.to_string()}],
        }
        response = requests.post(self.url, headers=headers, json=data)
        res_json = response.json()
        
        from ragas.llms.base import LLMResult
        from ragas.llms.base import Generation
        
        text = res_json['choices'][0]['message']['content']
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(self, prompt, **kwargs):
        return self.generate_text(prompt)


def _run_ragas(question, answer, contexts, reference, ragas_llm, ragas_embeddings):
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts if contexts else [""],
        reference=reference,
    )
    dataset = EvaluationDataset(samples=[sample])

    if contexts:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    else:
        metrics = [answer_relevancy]

    result = evaluate(dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings)

    scores = {
        "faithfulness": round(result["faithfulness"][0], 3) if contexts else "-",
        "answer_relevancy": round(result["answer_relevancy"][0], 3),
        "context_precision": round(result["context_precision"][0], 3) if contexts else "-",
        "context_recall": round(result["context_recall"][0], 3) if contexts else "-",
    }
    return scores


def evaluate_report(question, contexts, answer_a, answer_b=None, answer_c=None):
    """
    비교 모드 전용 평가
    answer_a: gpt-4o + RAG      → reference (baseline)
    answer_b: gpt-4o-mini, no RAG
    answer_c: gpt-4o-mini + RAG (현재 시스템)
    """
    ragas_llm = LuxiaRagasLLM(api_key=API_KEY, model="gpt-4o-mini")
    ragas_llm = LuxiaRagasLLM(api_key=API_KEY, model="gpt-4o-mini")
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    reference = answer_a  # gpt-4o+RAG 답변을 ground truth로 사용

    print("=== [A] gpt-4o + RAG (baseline) ===")
    scores_a = _run_ragas(question, answer_a, contexts, reference, ragas_llm, ragas_embeddings)

    print("=== [B] gpt-4o-mini, no RAG ===")
    scores_b = _run_ragas(question, answer_b, None, reference, ragas_llm, ragas_embeddings)

    print("=== [C] gpt-4o-mini + RAG (현재 시스템) ===")
    scores_c = _run_ragas(question, answer_c, contexts, reference, ragas_llm, ragas_embeddings)

    return {"A_gpt4o_rag": scores_a, "B_mini_no_rag": scores_b, "C_mini_rag": scores_c}


def evaluate_single(question, contexts, answer, ragas_llm=None, ragas_embeddings=None):
    """단일 모드 전용 평가 (gpt-4o-mini + RAG)"""
    if ragas_llm is None:
        ragas_llm = LuxiaRagasLLM(api_key=API_KEY, model="gpt-4o-mini")
    if ragas_embeddings is None:
        ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=answer,
    )
    dataset = EvaluationDataset(samples=[sample])

    # faithfulness + answer_relevancy만 측정
    metrics = [faithfulness, answer_relevancy]
    result = evaluate(dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings)

    return {
        "faithfulness": round(result["faithfulness"][0], 3),
        "answer_relevancy": round(result["answer_relevancy"][0], 3),
        "context_precision": "-",
        "context_recall": "-",
    }

# if __name__ == "__main__":

#     # 가짜 데이터로 평가
#     question = "What are common defects in glass insulators?"
#     answer   = "Common defects include surface cracks, contamination, and partial discharge which can lead to failure."
#     contexts = [
#         "Glass insulators can develop surface cracks due to mechanical stress.",
#         "Contamination on glass insulators causes partial discharge and eventual failure.",
#     ]
#     # reference = "The common defects in glass insulators are surface cracks, contamination, and partial discharge."

#     scores = evaluate_report(question, answer, contexts)

#     print("RAGAS 평가 결과:")
#     for metric, score in scores.items():
#         print(f"  {metric:25s}: {score}")