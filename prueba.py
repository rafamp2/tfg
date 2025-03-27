from app import load_vectordb
from src.retrieval_qa import build_rerank_retriever, build_retrieval_chain
from streamlit_app.utils import load_base_embeddings, load_llm, load_reranker
import streamlit as st
import speech_recognition as sr
from PIL import Image

from src import CFG
from src.retrieval_qa import build_retrieval_chain
from src.vectordb import build_vectordb, delete_vectordb, load_faiss, load_chroma
from streamlit_app.utils import perform, load_base_embeddings, load_llm, load_reranker

from src.audio_player import AudioManager

st.set_page_config(page_title="Conversación con Don Francisco de Arobe",layout="wide")
user_mode = CFG.DEV_MODE

LLM = load_llm()
RERANKER = load_reranker()
BASE_EMBEDDINGS = load_base_embeddings()
vectordb = load_vectordb()

retrieval_chain = build_retrieval_chain(vectordb, RERANKER, LLM)

response = retrieval_chain.invoke({
                        "question": "Hola", 
                        "chat_history": list()},)