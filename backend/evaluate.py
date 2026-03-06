import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from datasets import Dataset
import mlflow

judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    google_api_key=os.environ["GEMINI_API_KEY"],
))

judge_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


faithfulness.llm = judge_llm
answer_relevancy.llm = judge_llm
answer_relevancy.embeddings = judge_embeddings
context_precision.llm = judge_llm



def evaluate_rag(question: str, answer: str, contexts: list[str], ground_truth: str = ""):
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth if ground_truth else answer],
    }

    dataset = Dataset.from_dict(data)

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    return {
        "faithfulness": float(scores["faithfulness"][0]),
        "answer_relevancy": float(scores["answer_relevancy"][0]),
        "context_precision": float(scores["context_precision"][0]),
    }


def log_to_mlflow(question: str, answer: str, scores: dict, latency: float, session_id: str = ""):
    mlflow.set_experiment("RAG_Evaluation")

    with mlflow.start_run():
        mlflow.log_param("question", question[:200])
        mlflow.log_param("session_id", session_id)

        mlflow.log_metric("faithfulness", scores["faithfulness"])
        mlflow.log_metric("answer_relevancy", scores["answer_relevancy"])
        mlflow.log_metric("context_precision", scores["context_precision"])
        mlflow.log_metric("latency_seconds", latency)
