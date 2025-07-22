"""
LLM
"""

import os
from src import CFG
from langchain.callbacks import StreamingStdOutCallbackHandler

# Para usar CTransformers
from langchain_community.llms.ctransformers import CTransformers
from auto_gptq import exllama_set_max_input_length

# Para usar LlamaCpp
from langchain_community.llms.llamacpp import LlamaCpp

# Para usar una API de OpenAI
from langchain_openai import ChatOpenAI

# Para usar la biblioteca transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
import torch




def build_llm():
    """Builds LLM defined in config."""
    if CFG.WRAPPER == "TRANSFORMERS":
            return build_transformers(
                os.path.join(CFG.MODELS_DIR, CFG.LLM_PATH),
                config={
                    "max_new_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                    "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                    "repetition_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
                },
                max_input_length = CFG.LLM_CONFIG.CONTEXT_LENGTH
            )
    elif CFG.LLM_PATH.endswith(".gguf"):
        
        if CFG.WRAPPER == "LLAMACPP":
            return build_llamacpp(
                os.path.join(CFG.MODELS_DIR, CFG.LLM_PATH),
                config={
                    "max_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                    "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                    "repeat_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
                    "n_ctx": CFG.LLM_CONFIG.N_CTX,
                    "n_gpu_layers": CFG.LLM_CONFIG.N_GPU_LAYERS,
                },
            )
            
        elif CFG.WRAPPER == "CTRANSFORMERS":
            return build_ctransformers(
                os.path.join(CFG.MODELS_DIR, CFG.LLM_PATH),
                config={
                    "max_new_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                    "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                    "repetition_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
                    "context_length": CFG.LLM_CONFIG.CONTEXT_LENGTH,
                },
            )
        else:
            raise NotImplementedError
    elif CFG.LLM_PATH.startswith("http"):
        return chatopenai(
            CFG.LLM_PATH,
            config={
                "max_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                "temperature": CFG.LLM_CONFIG.TEMPERATURE,
            },
        )
    else:
        raise NotImplementedError


def build_ctransformers(
    model_path: str, config: dict | None = None, debug: bool = False, **kwargs
):
    """Builds LLM using CTransformers."""
    if config is None:
        config = {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "repetition_penalty": 1.1,
            "context_length": 4000,
        }

    llm = CTransformers(
        model=model_path,
        config=config,
        callbacks=[StreamingStdOutCallbackHandler()] if debug else None,
        **kwargs,
    )
    return llm


def build_llamacpp(
    model_path: str, config: dict | None = None, debug: bool = False, **kwargs
):
    """Builds LLM using LlamaCpp."""
    if config is None:
        config = {
            "max_tokens": 512,
            "temperature": 0.2,
            "repeat_penalty": 1.1,
            "n_ctx": 16,
        }

    llm = LlamaCpp(
        model_path=model_path,
        **config,
        callbacks=[StreamingStdOutCallbackHandler()] if debug else None,
        **kwargs,
    )
    return llm

def build_transformers(
    model_name_or_path: str,
    config: dict | None = None,
    debug: bool = False,
    max_input_length: int = 4096,
    **kwargs
):
    """
    Builds a HuggingFacePipeline-compatible LLM for LangChain using transformers.

    Args:
        model_name_or_path (str): HuggingFace model ID or path.
        config (dict, optional): Generation config. Defaults to common values.
        debug (bool): If True, prints model and config info.
        **kwargs: Extra kwargs passed to the pipeline.

    Returns:
        HuggingFacePipeline: LangChain-compatible LLM.
    """
    if config is None:
        config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "top_p": 0.95,
        }

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    # Para el tamaño del contexto
    model = exllama_set_max_input_length(model, max_input_length=max_input_length)

    # Convertimos el modelo en un pipeline para más tarde ajustarlo a la interfaz de modelo de Langchain
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **config,
        **kwargs
    )

    if debug:
        print(f"Loaded model: {model_name_or_path}")
        print(f"Generation config: {config}")

    # HuggingFacePipeline transforma el pipeline en una instancia de modelo compatible con Langchain
    return HuggingFacePipeline(pipeline=pipe)

def chatopenai(openai_api_base: str, config: dict | None = None, **kwargs):
    """For LLM deployed as an API."""
    if config is None:
        config = {
            "max_tokens": 512,
            "temperature": 0.2,
        }

    llm = ChatOpenAI(
        openai_api_base=openai_api_base,
        openai_api_key="sk-xxx",
        **config,
        streaming=True,
        **kwargs,
    )
    return llm
