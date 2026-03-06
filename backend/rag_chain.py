import os
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # ← correct import
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import dotenv_values

config = dotenv_values(".env")  # add this near the top of the file


load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def load_vectorstore(chroma_path: str = CHROMA_PATH):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )
    return vectorstore

def build_rag_chain(chroma_path: str = CHROMA_PATH):
    vectorstore = load_vectorstore(chroma_path)
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate.from_template("""You are a helpful research assistant.
Use the following context from research papers to answer the question.
If you don't know the answer based on the context, say "I don't have enough information in the provided documents."

Context:
{context}

Question: {question}

Answer:""")

    llm = ChatGoogleGenerativeAI(  # ← correct class
        model="gemma-3-27b-it",  # ← correct chat model
        google_api_key=config["GEMINI_API_KEY"],
        temperature=0.2,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


if __name__ == "__main__":
    print("[INFO] Building RAG chain...")
    chain, retriever = build_rag_chain()

    query = "Based on the Home EEG data provided in Figure 1, which specific sleep metric demonstrated a statistically significant reduction in Alzheimer’s Disease (AD) patients compared to controls, and why would relying on the Pittsburgh Sleep Quality Index (PSQI) or actigraphy (WASO) be insufficient for predicting this specific physiological change?"
    print(f"\n[INFO] Query: {query}")

    answer = chain.invoke(query)
    print(f"\n[INFO] Answer:\n{answer}")

    # Show sources separately
    print("\n[INFO] Sources:")
    docs = retriever.invoke(query)
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] Page {doc.metadata.get('page', '?')} — {doc.metadata.get('source', '?')}")
