from copy import deepcopy
from typing import List, Generator, Dict, Iterator

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import BaseMessage, BaseMessageChunk
from docu_bot.constants import PROMPTS
from docu_bot.stores.docstore import DocumentStore
from docu_bot.stores.utils import LoadedVectorStores
from docu_bot.stores.vectorstore import MultiVectorStore

from docu_bot.retrievals.empty_retrieval import EmptyRetrieval
from docu_bot.retrievals.document_retrival import DocumentRetrieval
from docu_bot.retrievals.generative_retrieval import GenerativeDocumentRetrieval
from docu_bot.retrievals.rerank_retrieval import RerankDocumentRetrieval
from docu_bot.retrievals.ner_retireval import NerRetrieval
from docu_bot.retrievals.theme_retrieval import ThemeRetrieval
from docu_bot.retrievals.keyphrase_retrieval import KeyphraseRetrieval
from docu_bot.retrievals.query_alteration_retrieval import (
    QueryAlterationDocumentRetrieval,
)
from docu_bot.retrievals.context_query_alteration_retrieval import (
    ContextQueryAlterationDocumentRetrieval,
)

from docu_bot.utils import create_chatopenai_model, format_docs, RETRIEVAL_TYPES


def answer_question(
    messages: List[Dict[str, str]],
    documents: List[Document],
    model: ChatOpenAI,
    prompt: str = PROMPTS.SYSTEM_PROMPT,
    temperature: float = 0.7,
    stream: bool = False,
) -> Iterator[BaseMessageChunk] | BaseMessage:
    """
    Answer a question using a given model and a list of documents.
    """
    # Get the best answer from the model
    prompt_template = ChatPromptTemplate.from_messages(
        [("user", prompt + PROMPTS.INPUT_PROMPT)]
        + [(message["role"], message["content"]) for message in messages]
    )
    model = model.bind(temperature=temperature)
    if stream:
        return model.stream(
            prompt_template.invoke(
                {
                    "context": format_docs(documents),
                    "query": messages[0]["content"],
                }
            )
        )
    else:
        return model.invoke(
            prompt_template.invoke(
                {
                    "context": format_docs(documents),
                    "query": messages[0]["content"],
                }
            )
        )


def get_documents(
    question: str,
    retriever: BaseRetriever,
) -> List[Document]:
    """
    Retrieve documents relevant to the question.
    """
    return retriever.invoke(input=question)


def prepare_retriever(
    long_branches: List[str],
    zip_files: List[str],
    docstore: DocumentStore,
    loaded_vectorstores: LoadedVectorStores,
    model_type: str,
    api_key: str,
    retrieval_type: RETRIEVAL_TYPES = RETRIEVAL_TYPES.DEFAULT,
    search_kwargs: Dict[str, str] = {},
) -> BaseRetriever:
    vectorstore_to_use = []
    for repo_or_files in long_branches + zip_files:
        vectorstore_to_use.extend(loaded_vectorstores.get_vectorstores(repo_or_files))

    vectorstore = MultiVectorStore(chroma_vectors=vectorstore_to_use)
    llm = create_chatopenai_model(
        model_type=model_type,
        api_key=api_key,
    )

    if retrieval_type == RETRIEVAL_TYPES.EMPTY:
        return EmptyRetrieval(
            vectorstore=vectorstore,
            docstore=docstore,
            search_kwargs=search_kwargs,
        )
    elif retrieval_type == RETRIEVAL_TYPES.GENERATIVE:
        return GenerativeDocumentRetrieval(
            vectorstore=vectorstore,
            docstore=docstore,
            llm=llm,
            search_kwargs=search_kwargs,
        )
    elif retrieval_type == RETRIEVAL_TYPES.CONTEXT_QUERY_ALTERATION:
        return ContextQueryAlterationDocumentRetrieval(
            vectorstore=vectorstore,
            docstore=docstore,
            llm=llm,
            search_kwargs=search_kwargs,
        )
    elif retrieval_type == RETRIEVAL_TYPES.QUERY_ALTERATION:
        return QueryAlterationDocumentRetrieval(
            vectorstore=vectorstore,
            docstore=docstore,
            llm=llm,
            search_kwargs=search_kwargs,
        )
    elif retrieval_type == RETRIEVAL_TYPES.RERANK:
        return RerankDocumentRetrieval(
            vectorstore=vectorstore,
            docstore=docstore,
            llm=llm,
            search_kwargs=search_kwargs,
        )
    elif retrieval_type == RETRIEVAL_TYPES.NER_RETRIEVAL:
        return NerRetrieval(
            vectorstore=vectorstore,
            docstore=docstore,
            llm=llm,
            search_kwargs=search_kwargs,
        )
    elif retrieval_type == RETRIEVAL_TYPES.THEME_RETRIEVAL:
        return ThemeRetrieval(
            vectorstore=vectorstore,
            docstore=docstore,
            llm=llm,
            search_kwargs=search_kwargs,
        )
    elif retrieval_type == RETRIEVAL_TYPES.KEYPHRASE_RETRIEVAL:
        return KeyphraseRetrieval(
            vectorstore=vectorstore,
            docstore=docstore,
            llm=llm,
            search_kwargs=search_kwargs,
        )
    else:
        return DocumentRetrieval(
            vectorstore=vectorstore,
            docstore=docstore,
            search_kwargs=search_kwargs,
        )


def rag(
    messages: List[Dict[str, str]],
    retrieved_documents: List[Document],
    model_type: str,
    api_key: str,
    temperature: float = 0.7,
    prompt: str = PROMPTS.SYSTEM_PROMPT,
) -> List[Dict[str, str]]:

    llm = create_chatopenai_model(
        model_type=model_type,
        api_key=api_key,
    )

    model_answer = answer_question(
        messages=messages,
        documents=retrieved_documents,
        model=llm,
        prompt=prompt,
        temperature=temperature,
        stream=False,
    )
    output = deepcopy(messages)
    output.append({"role": "assistant", "content": model_answer.content})
    return output


def stream_rag(
    messages: List[Dict[str, str]],
    retrieved_documents: List[Document],
    model_type: str,
    api_key: str,
    temperature: float = 0.7,
    prompt: str = PROMPTS.SYSTEM_PROMPT,
) -> Generator[List[Dict[str, str]], None, None]:

    llm = create_chatopenai_model(
        model_type=model_type,
        api_key=api_key,
    )

    model_answer = answer_question(
        messages=messages,
        documents=retrieved_documents,
        model=llm,
        prompt=prompt,
        temperature=temperature,
        stream=True,
    )

    output = deepcopy(messages)

    output.append({"role": "assistant", "content": ""})
    for output_chunk in model_answer:
        output[-1]["content"] += output_chunk.content
        yield output
