import uuid

from typing import List, Optional
from docu_bot.retrievals.document_retrival import DocumentRetrieval
from docu_bot.constants import PROMPTS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import BaseStore


class GenerativeDocumentRetrieval(DocumentRetrieval):

    llm: ChatOpenAI
    generate_prompt: str = PROMPTS.GENERATE_DOCUMENT_PROMPT

    vectorstore: Optional[VectorStore] = None
    docstore: Optional[BaseStore[str, Document]] = None

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:

        llm = self.llm.bind(stop=["context"], max_tokens=300)

        template = ChatPromptTemplate.from_messages(
            [
                ("user", self.generate_prompt),
            ]
        )
        k = self.search_kwargs.get("k", 1)

        documents = []
        for _ in range(k):
            prompt = template.invoke({"query": query})
            msg = llm.invoke(prompt)
            documents.append(
                Document(
                    page_content=msg.content,
                    metadata={
                        "ItemId": str(uuid.uuid4()),
                        "source": "llm",
                    },
                )
            )

        return documents
