import re
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from docu_bot.constants import MODEL_TYPES


class RETRIEVAL_TYPES:
    EMPTY = "empty"
    DEFAULT = "default"
    GENERATIVE = "generative"
    RERANK = "rerank"
    QUERY_ALTERATION = "query_alteration"
    CONTEXT_QUERY_ALTERATION = "context_query_alteration"
    NER_RETRIEVAL = "ner_retrieval"
    THEME_RETRIEVAL = "theme_retrieval"
    KEYPHRASE_RETRIEVAL = "keyphrase_retrieval"


def create_chatopenai_model(
    model_type: str,
    api_key: str,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI model from a model type and an api key.
    """
    return ChatOpenAI(
        model=model_type,
        api_key=api_key,
        base_url=MODEL_TYPES.LLM_MODELS[model_type],
    )


def create_openai_embeddings(
    model_type: str = MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key: str = "None"
) -> OpenAIEmbeddings:
    """
    Create OpenAI Embeddings.
    """
    return OpenAIEmbeddings(
        model=model_type,
        api_key=api_key,
        base_url=(
            MODEL_TYPES.DEFAULT_EMBED_LOC
            if model_type == MODEL_TYPES.DEFAULT_EMBED_MODEL
            else None
        ),
    )


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        [
            f"{i+1}. Document: \n"
            + re.sub(
                r"[\t\v]",
                "",
                re.sub(r"\n+", "\n", re.sub(r"[^\w\s!\.\?,]", "", d.page_content)),
            ).strip()
            for i, d in enumerate(docs)
        ]
    )
