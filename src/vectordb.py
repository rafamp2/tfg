"""
VectorDB
"""

import shutil
import os
from typing import Optional, Sequence

from chromadb.config import Settings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS

from src import CFG, logger
from src.parser import load_pdf, text_split, propositionize, simple_text_split


def build_vectordb(filename: str, embedding_function: Embeddings) -> None:
    #Construye una base de datos vectorial a partir de documentos
    parts = load_pdf(filename)

    if CFG.TEXT_SPLIT_MODE == "default":
        docs = simple_text_split(parts, CFG.CHUNK_SIZE, CFG.CHUNK_OVERLAP)
        save_vectordb(docs, embedding_function, CFG.VECTORDB_PATH, CFG.VECTORDB_TYPE)
    elif CFG.TEXT_SPLIT_MODE == "propositionize":
        docs = propositionize(parts)
        save_vectordb(docs, embedding_function, CFG.VECTORDB_PATH, CFG.VECTORDB_TYPE)
    else:
        raise NotImplementedError


def save_vectordb(
    docs: Sequence[Document],
    embedding_function: Embeddings,
    persist_directory: str,
    vectordb_type: str,
) -> None:
    #Guarda  nuestra base de datos vectorial
    logger.info(f"Guardando vectordb en directorio '{persist_directory}'")

    if vectordb_type == "faiss":
        vectorstore = FAISS.from_documents(docs, embedding_function)
        vectorstore.save_local(persist_directory)
    elif vectordb_type == "chroma":
        vectorstore = Chroma(
            collection_name="langchain",
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )

        _ = vectorstore.add_documents(docs)
    else:
        raise NotImplementedError


def delete_vectordb(persist_directory: str, vectordb_type: str) -> None:
    #Deletes vector database.
    logger.info(f"Borrando vectordb en directorio '{persist_directory}'")
    if vectordb_type == "faiss":
        shutil.rmtree(persist_directory)
    elif vectordb_type == "chroma":
        vectorstore = Chroma(
            collection_name="langchain",
            persist_directory=persist_directory,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        vectorstore.delete_collection()
    else:
        raise NotImplementedError


def load_faiss(
    embedding_function: Embeddings, persist_directory: Optional[str] = None
) -> VectorStore:
    #Loads a FAISS index from disk.
    if persist_directory is None:
        persist_directory = CFG.VECTORDB_PATH
    logger.info(f"Usando Faiss, con directorio persistente  = {persist_directory}")

    return FAISS.load_local(
        persist_directory, embedding_function, allow_dangerous_deserialization=True
    )


def load_chroma(
    embedding_function: Embeddings, persist_directory: Optional[str] = None
) -> VectorStore:
    #Loads a Chroma index from disk.
    if persist_directory is None:
        persist_directory = CFG.VECTORDB_PATH
    if not os.path.exists(persist_directory):
        raise FileNotFoundError
    logger.info(f"Usando Chroma, con directorio persistente = {persist_directory}")

    return Chroma(
        collection_name="langchain",
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
