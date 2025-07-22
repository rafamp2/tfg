
#Embeddings

import os

from langchain.chains import HypotheticalDocumentEmbedder, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

from src import CFG


def build_base_embeddings():
    #Builds base embeddings defined in config.
    base_embeddings = HuggingFaceEmbeddings(
        model_name=os.path.join(CFG.MODELS_DIR, CFG.EMBEDDINGS_PATH),
        model_kwargs={"device": CFG.DEVICE},
    )
    return base_embeddings
