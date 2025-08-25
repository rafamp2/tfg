from src import CFG
import os

import os

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.runnables import Runnable
from langchain.prompts.chat import ChatPromptValue
from langchain_core.messages import AIMessage, HumanMessage
from typing import Optional, Union


# Para usar LlamaCpp
from llama_cpp import Llama

# Para usar una API de OpenAI
from langchain_openai import ChatOpenAI

# Para usar transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from auto_gptq import exllama_set_max_input_length

# Para restaurar la caché
import torch
import gc

def build_llm():
    return Llm(
        model_path = os.path.join(CFG.MODELS_DIR,CFG.LLM_PATH), 
        wrapper = CFG.WRAPPER
    )

class Llm(Runnable):
    tokenizer: Optional[AutoTokenizer]
    model: Union[AutoModelForCausalLM, Llama, ChatOpenAI]
    pipe: Optional[HuggingFacePipeline]

    def __init__(   
        self,
        model_path: str,
        wrapper: str,
    ):
        torch.cuda.empty_cache()
        gc.collect()

        """Builds LLM defined in config."""
        if wrapper == "TRANSFORMERS":
            self.build_transformers(
                model_path,
                config={
                    "max_new_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                    "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                    "repetition_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
                    "top_p": CFG.LLM_CONFIG.TOP_P
                },
                max_input_length = CFG.LLM_CONFIG.CONTEXT_LENGTH
            )
        elif wrapper == "LLAMACPP":
            self.build_llamacpp(
                model_path=model_path,
                model_name=CFG.LLM_NAME,
                config={
                    "n_ctx": CFG.LLM_CONFIG.CONTEXT_LENGTH,
                    "n_gpu_layers": CFG.LLM_CONFIG.N_GPU_LAYERS,
                    "n_threads":  CFG.LLM_CONFIG.N_THREADS,
                },
            )

        elif wrapper.startswith("http"):
            self.chatopenai(
                CFG.LLM_PATH,
                config={
                    "max_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                    "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                },
            )
        else:
            raise NotImplementedError
        

        

    def invoke(self, input: ChatPromptValue, config=None) -> str:
        # Obtener la lista de mensajes de ChatPromptValue
        chat = input.messages

        formatted_chat = []

        for msg in chat:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                continue  # o lanza error si es necesario

            formatted_chat.append({"role": role, "content": msg.content})

        if CFG.WRAPPER == "TRANSFORMERS":
            return self.invoke_transformers(formatted_chat)
        elif CFG.WRAPPER == "LLAMACPP":
            return self.invoke_llamacpp(formatted_chat)
        elif CFG.LLM_PATH.startswith("http"):
            return self.invoke_chatopenai(formatted_chat)


    # Métodos para usar transformers

    def build_transformers(
        self,
        model_name: str,
        config: dict | None = None,
        max_input_length: int = 4096
    ):
        """
        Builds a HuggingFacePipeline-compatible LLM for LangChain using transformers.

        Args:
            model_name_or_path (str): HuggingFace model ID or path.
            config (dict, optional): Generation config. Defaults to common values.

        Returns:
            HuggingFacePipeline: LangChain-compatible LLM.
        """
        if config is None:
            config = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "repetition_penalty": 1.1,
                "do_sample": False,
                "top_p": 0.95,
            }

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=CFG.LLM_DEVICE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            revision="float16",
        )

        # Para el tamaño del contexto
        if hasattr(self.model.config, "max_position_embeddings"):
            self.model.config.max_position_embeddings = max_input_length
        elif hasattr(self.model, "exllama_config") and quant_config and getattr(quant_config, "desc_act", False):
            self.model = exllama_set_max_input_length(self.model, max_input_length=max_input_length)
        
        
        self.pipe =  pipeline(
            "text-generation",
            model= self.model,
            tokenizer= self.tokenizer,
            config=config,            
        )

    def invoke_transformers(self,formatted_chat: list[dict[str, str]]):
        # Aplica la plantilla del tokenizer
        prompt = self.tokenizer.apply_chat_template(
            formatted_chat,
            tokenize=False,
            add_generation_prompt=CFG.LLM_CONFIG.ADD_PROMPT
        )

         # Genera texto usando pipeline
        outputs = self.pipe(
            prompt,
            return_full_text=False  # Para que solo devuelva la respuesta generada, sin repetir el prompt
        )

        # Extrae y limpia la respuesta generada
        response = outputs[0]["generated_text"].strip()
        return response


    # Métodos para usar llama_cpp

    def build_llamacpp(
        self,
        model_path: str, 
        model_name: str,
        config: dict | None = None, 
        debug: bool = False, 
        **kwargs
    ):
        """Builds LLM using LlamaCpp."""
        if config is None:
            config = {
                "max_tokens": 512,
                "temperature": 0.2,
                "repeat_penalty": 1.1,
                "n_ctx": 16,
            }



        self.model = Llama(
            model_path=os.path.join(model_path,model_name),
            **config,
        )

    def invoke_llamacpp(self,formatted_chat: list[dict[str, str]]):
        return self.model.create_chat_completion(
            messages=formatted_chat,
            max_tokens= CFG.LLM_CONFIG.MAX_NEW_TOKENS,
            temperature= CFG.LLM_CONFIG.TEMPERATURE,
            repeat_penalty= CFG.LLM_CONFIG.REPETITION_PENALTY,
        )['choices'][0]['message']['content']

    # Métodos para usar API de OpenAI

    def chatopenai(
        self,
        openai_api_base: str, 
        config: dict | None = None, 
        **kwargs
    ):
        """For LLM deployed as an API."""
        if config is None:
            config = {
                "max_tokens": 512,
                "temperature": 0.2,
            }

        self.chat_openai_model = ChatOpenAI(
            openai_api_base=openai_api_base,
            openai_api_key="sk-xxx",
            **config,
            streaming=True,
            **kwargs,
        )

    def invoke_chatopenai(self,formatted_chat: list[dict[str, str]]):
        # Definir en caso de que se quiera usar OpenAI
        return None
