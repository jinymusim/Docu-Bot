from langchain_openai import ChatOpenAI
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
