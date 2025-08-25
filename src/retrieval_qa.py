"""
Retrieval QA
"""

from typing import List

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnableLambda

from src import CFG
from src.prompt_templates import contextualize_q_prompt, qa_prompt


def contextualize_question(input: dict, contextualize_q_chain):
    # Si hay historial de chat no vacío, usa contextualize_q_chain para reformular
    if input.get("chat_history") and len(input["chat_history"]) > 0:
        # Ejecutar la cadena para reformular la pregunta
        return contextualize_q_chain.invoke(input)
    else:
        # Si no hay historial, devolver la pregunta original
        return input["question"]

def build_contextualize_q_chain(contextualize_q_system_prompt: str, llm : LLM):
    contextualize_q_chain = contextualize_q_system_prompt | llm | StrOutputParser()
    return contextualize_q_chain


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_retrieval_chain(
    vectordb: VectorStore, reranker: BaseDocumentCompressor, llm: LLM,
) -> ConversationalRetrievalChain:

    contextualize_q_chain = build_contextualize_q_chain(contextualize_q_prompt, llm)
    
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CFG.RERANK_RETRIEVER_CONFIG.SEARCH_K}
    )
    
    # Extrae pregunta y documentos, aplica reranker, y devuelve texto
    get_context = (
        RunnableLambda(lambda x: {"question": x["question"], "docs": retriever.invoke(x["question"])})
        |RunnableLambda(lambda x: reranker.compress_documents(x["docs"], x["question"]))
        | RunnableLambda(lambda docs: format_docs(docs))
    )
    
    retrieval_chain = (
        RunnableMap({
            "question": RunnableLambda(lambda input: contextualize_question(input, contextualize_q_chain)),
            "chat_history": lambda x: x["chat_history"],
        })
        .assign(context=get_context)
        | qa_prompt
        | llm
    )
    return retrieval_chain
