from src import CFG
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

def build_template(system: str, user: str, prompt_type: str):
    if prompt_type == "llama":
        template = f"""<s>[INST] <<SYS>>{system}<</SYS>>\n{user}\n[/INST]"""
    elif prompt_type == "mistral":
        template = f"""<s>[INST] {system}\n{user}\n[/INST]"""
    elif prompt_type == "zephyr":
        template = f"""<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>"""
    elif prompt_type == "gemma":
        template = f"""{system}\n{user}\n"""
    else:
        raise NotImplementedError(f"Prompt type {prompt_type} not supported.")
    
    return template

llama_format = """<s>[INST] <<SYS>>{system}<</SYS>>
{user}
[/INST]"""

mistral_format = """<s>[INST] {system}
{user}
[/INST]"""

zephyr_format = """<|system|>
{system}</s>
<|user|>
{user}</s>
<|assistant|>"""

gemma_format = """<start_of_turn>user
{system}
{user}<end_of_turn>
<start_of_turn>model"""


if CFG.PROMPT_TYPE == "llama":
    _chat_format = llama_format
elif CFG.PROMPT_TYPE == "mistral":
    _chat_format = mistral_format
elif CFG.PROMPT_TYPE == "zephyr":
    _chat_format = zephyr_format
elif CFG.PROMPT_TYPE == "gemma":
    _chat_format = gemma_format
else:
    raise NotImplementedError


class QA:
    system = (
        "Vas a actuar como Don Francisco de Arobe. Un personaje histórico del siglo 16. "
        "Asegúrate siempre de que tus respuestas son breves y que solamente das una respuesta a lo que te ha preguntado el usuario. "
        "Usa las piezas de contexto recuperado para ayudarte a responder a las preguntas del usuario. "
        "Siempre responde permaneciendo en personaje y usando la primera persona. "
        "Responde en 5 líneas de texto o menos. "
        "Si algo preguntado no te parece adecuado que lo responda el personaje, no respondas. "
        "Si no sabes la respuesta a una pregunta, contesta que no sabes la respuesta, no intentes inventarte una respuesta. "
        "Responde en español. "
    )
    user = "Pregunta: {question}\nContexto:\n{context}\n"


class CondenseQuestion:
    system = (
        "Tu tarea es reformular la pregunta del usuario para que sea completamente independiente, "
        "sin depender de contexto anterior. Responde solo con la pregunta reformulada. "
        "Responde en español. "
    )
    user = (
        "Historial del chat:\n{chat_history}\n"
        "Siguiente pregunta: {question}\n"
        "Pregunta independiente:"
    )

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Tu tarea es reformular la pregunta del usuario para que sea completamente independiente, \
                                sin depender de contexto anterior. Responde solo con la pregunta reformulada. \
                                Responde en español."""

QA_SYSTEM_PROMPT = """Vas a actuar como Don Francisco de Arobe. Un personaje histórico del siglo 16. \
        Asegúrate siempre de que tus respuestas son breves y que solamente das una respuesta a lo que te ha preguntado el usuario. \
        Usa las piezas de contexto recuperado para ayudarte a responder a las preguntas del usuario. \
        Siempre responde permaneciendo en personaje y usando la primera persona. \
        Responde en 5 líneas de texto o menos. \
        Si algo preguntado no te parece adecuado que lo responda el personaje, no respondas. \
        Si no sabes la respuesta a una pregunta, contesta que no sabes la respuesta, no intentes inventarte una respuesta. \
        Responde en español."""

QA_TEMPLATE = build_template(QA.system, QA.user, CFG.PROMPT_TYPE)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("USER", "{question}"),
    ]
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

"""
QA_TEMPLATE = _chat_format.format(system=QA.system, user=QA.user)
CONDENSE_QUESTION_TEMPLATE = _chat_format.format(
    system=CondenseQuestion.system, user=CondenseQuestion.user
)
HYDE_TEMPLATE = _chat_format.format(system=Hyde.system, user=Hyde.user)
MULTI_QUERIES_TEMPLATE = _chat_format.format(
    system=MultipleQueries.system, user=MultipleQueries.user
)
"""
