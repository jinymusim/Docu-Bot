from typing import List, Optional
from docu_bot.retrievals.document_retrival import DocumentRetrieval

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.stores import BaseStore


class EmptyRetrieval(DocumentRetrieval):

    vectorstore: Optional[VectorStore] = None
    docstore: Optional[BaseStore[str, Document]] = None

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:

        return []
