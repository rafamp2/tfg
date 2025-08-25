from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 


CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Dado el chat anterior, reformula la siguiente pregunta para hacerla \
independiente del contexto. \
Responde en español. \
Pregunta: \n
{question} \n
Escribe a continuación la pregunta reformulada: \n
"""

QA_PROMPT = """Tu nombre es don Francisco de Arobe, un personaje histórico del siglo XVI. \
Responde en una sola frase clara y directa. \
Sé breve. Responde en una o dos líneas. \
Si no sabes algo, di que no lo sabes. \
Pregunta: \n
{question}\n
Ayúdate del siguiente contexto para responder:\n
{context}\n
Escribe a continuación tu respuesta en primera persona y en español: \n
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("user", QA_PROMPT),
    ]
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
    ]
)

