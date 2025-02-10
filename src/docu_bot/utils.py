from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from docu_bot.constants import MODEL_TYPES


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
        base_url=MODEL_TYPES.LLM_MODELS.get(model_type),
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
