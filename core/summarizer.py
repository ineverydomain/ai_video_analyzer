# from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough , RunnableLambda

import os

def get_llm(mode: str = "api"):
    if mode.lower() == "local":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model="llama2", temperature=0.3)
        except ImportError:
            raise RuntimeError("Local LLM requires 'langchain-ollama' package and Ollama runtime.")
    # return ChatMistralAI(model = "mistral-small-latest" , mistral_api_key = os.getenv("MISTRAL_API_KEY"), temperature=0.3 )
    return ChatOpenAI(model = "gpt-4o-mini" , api_key = os.getenv("OPENAI_API_KEY"), temperature=0.3 )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)

def summarize(transcript : str, mode: str = "api") -> str:
    llm = get_llm(mode=mode)

    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)

def generate_title(transcipt : str, mode: str = "api") -> str:
    llm = get_llm(mode=mode)

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    return title_chain.invoke(transcipt[:2000])
