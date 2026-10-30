# Module 09 - LangChain
# 9.8 Building a Full RAG Chain with LangChain
#
# A complete Retrieval-Augmented Generation (RAG) pipeline:
#   1. Ingestion: Raw Documents with domain knowledge
#   2. Chunking: RecursiveCharacterTextSplitter
#   3. Vector Storage: Embedding + VectorStore index (Chroma / InMemoryVectorStore)
#   4. Retrieval: VectorStore retriever with top-k similarity
#   5. Augmented Prompt: System prompt with strictly bounded context
#   6. LCEL Composition: {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | parser

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Apply compatibility patches and model helpers from utils
from utils import get_chat_model, get_embeddings, DeterministicMockEmbeddings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def build_vector_store(chunks: list[Document], embeddings):
    """Initializes vector store, preferring Chroma if supported, or InMemoryVectorStore with fallback."""
    try:
        from langchain_community.vectorstores import Chroma
        return Chroma.from_documents(chunks, embeddings)
    except Exception:
        pass

    try:
        return InMemoryVectorStore.from_documents(chunks, embeddings)
    except Exception:
        # If remote embeddings failed, re-index with offline deterministic embeddings
        fallback_emb = DeterministicMockEmbeddings(dim=32)
        return InMemoryVectorStore.from_documents(chunks, fallback_emb)


def run_full_rag_demo() -> None:
    print("=" * 60, flush=True)
    print("9.8 Building a Full RAG Chain with LangChain", flush=True)
    print("=" * 60, flush=True)

    # ── 1. Documents ─────────────────────────────────────────────────────────
    docs = [
        Document(page_content="RAG stands for Retrieval-Augmented Generation. It connects LLMs to external knowledge bases."),
        Document(page_content="The retrieval step uses embeddings and vector search to find relevant documents for a given query."),
        Document(page_content="The generation step takes the retrieved context and the user question to produce a grounded answer."),
        Document(page_content="Vector databases like ChromaDB, Pinecone, and Weaviate store embeddings for fast similarity search."),
        Document(page_content="Fine-tuning bakes knowledge into model weights. RAG retrieves knowledge at inference time. Both have trade-offs."),
    ]
    print(f"1. Prepared {len(docs)} knowledge base documents.")

    # ── 2. Split ─────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    print(f"2. Split into {len(chunks)} chunks.")

    # ── 3. Embed and store in VectorStore ─────────────────────────────────────
    embeddings = get_embeddings()
    vectorstore = build_vector_store(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"3. Vector store initialized ({vectorstore.__class__.__name__}) with top-k=3.")

    # ── 4. RAG Prompt Template ───────────────────────────────────────────────
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain the answer, say "I don't have that information."

Context:
{context}"""),
        ("human", "{question}"),
    ])

    # ── 5. Compose Chain with LCEL ───────────────────────────────────────────
    parser = StrOutputParser()

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    mock_answers = [
        "RAG stands for Retrieval-Augmented Generation.",
        "Fine-tuning modifies model weights during training, while RAG retrieves relevant knowledge at inference time.",
        "I don't have that information.",
    ]
    llm = get_chat_model(mock_responses=mock_answers)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | parser
    )

    # ── 6. Query the Chain ───────────────────────────────────────────────────
    print("\n4. Running RAG Query Evaluations:\n", flush=True)
    questions = [
        "What does RAG stand for?",
        "How is RAG different from fine-tuning?",
        "What is the capital of France?",  # Not in context - should say "I don't have that information."
    ]

    for i, q in enumerate(questions):
        print(f"Q: {q}")
        try:
            answer = rag_chain.invoke(q)
            print(f"A: {answer}\n")
        except Exception:
            # Offline grounded responder
            offline_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | rag_prompt
                | FakeListChatModel(responses=[mock_answers[i]])
                | parser
            )
            print(f"A (offline): {offline_chain.invoke(q)}\n")

    print("=" * 60, flush=True)
    print("9.8 Full RAG Chain Completed Successfully!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_full_rag_demo()
