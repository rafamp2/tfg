"""
Retrieval QA
"""

from typing import List

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from src import CFG
from src.prompt_templates import contextualize_q_prompt, qa_prompt


class VectorStoreRetrieverWithScores(VectorStoreRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Adapted from https://github.com/langchain-ai/langchain/blob/2f8dd1a1619f25daa4737df4d378b1acd6ff83c4/
        libs/core/langchain_core/vectorstores.py#L692
        """
        if self.search_type == "similarity":
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query, **self.search_kwargs
            )
            for doc, score in docs_and_scores:
                doc.metadata = {**doc.metadata, "similarity_score": f"{score}:.4f"}
            docs = [doc for doc, _ in docs_and_scores]
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            for doc, score in docs_and_similarities:
                doc.metadata = {**doc.metadata, "similarity_score": f"{score:.4f}"}
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

#Contextual Compression retriever that uses reranking as the base compressor for the documents
def build_rerank_retriever(
    vectordb: VectorStore, reranker: BaseDocumentCompressor
) -> ContextualCompressionRetriever:
    base_retriever = VectorStoreRetrieverWithScores(
        vectorstore=vectordb, search_kwargs={"k": CFG.RERANK_RETRIEVER_CONFIG.SEARCH_K}
    )
    return ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )


def build_contextualize_q_chain(contextualize_q_system_prompt: str, llm : LLM):

    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    return contextualize_q_chain

def contextualize_question(input: dict, llm: LLM, contextualize_q_chain: RunnableSequence):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_RAG_chain(
    vectordb: VectorStore, reranker: BaseDocumentCompressor, llm: LLM
) -> ConversationalRetrievalChain:
    """Builds a conversational retrieval chain model.

    Args:
        vectordb (VectorStore): The vector database to use.
        llm (LLM): The language model to use.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain model.
    """
    contextualize_q_chain = build_contextualize_q_chain(contextualize_q_prompt, llm)
    retriever = build_rerank_retriever(vectordb, reranker)
    rag_chain = (
        RunnablePassthrough.assign(
            context = contextualize_question | retriever | format_docs
        )
        | qa_prompt 
        | llm
    )

    return rag_chain


