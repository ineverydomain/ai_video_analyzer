#ActionableItems , decision, questions

# from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os 

def get_llm(mode: str = "api"):
    if mode.lower() == "local":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model="llama2", temperature=0.2)
        except ImportError:
            raise  RuntimeError("Local LLM requires 'langchain-ollama' package and Ollama runtime.")
    # return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.2)
    return ChatOpenAI(model = "gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"),temperature=0.2)



def build_chain(system_prompt : str, mode: str = "api"):
    llm = get_llm(mode=mode)
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript:str, mode: str = "api")->str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'",
        mode=mode
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str, mode: str = "api") -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'",
        mode=mode
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str, mode: str = "api") -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'",
        mode=mode
    )
    return chain.invoke(transcript)