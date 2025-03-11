import uuid
import asyncio
import logging
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain.retrievers import MultiVectorRetriever
from langchain_core.documents import Document, BaseDocumentTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from ragas.testset.graph import Node


class DocumentRetrieval(MultiVectorRetriever):

    parent_splitters: Dict[str, BaseDocumentTransformer] = {
        ".md": RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=2000,
            chunk_overlap=0,
            length_function=len,
        ),
        ".rst": RecursiveCharacterTextSplitter.from_language(
            Language.RST, chunk_size=2000, chunk_overlap=0, length_function=len
        ),
        ".pdf": RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=0, length_function=len
        ),
    }

    child_splitters: Dict[str, BaseDocumentTransformer] = {
        ".md": RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=500,
            chunk_overlap=150,
            length_function=len,
        ),
        ".rst": RecursiveCharacterTextSplitter.from_language(
            Language.RST, chunk_size=500, chunk_overlap=150, length_function=len
        ),
        ".pdf": RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=150, length_function=len
        ),
    }

    llm: Optional[ChatOpenAI] = None

    def __get_full_and_child_docs_with_metadata(
        self, documnets: List[Document], splitter: BaseDocumentTransformer
    ) -> Tuple[Tuple[str, List[Document]], List[Document]]:
        full_docs = []
        child_docs = []
        for doc in documnets:
            id = str(uuid.uuid4())
            full_docs.append((id, doc))
            split_documents = splitter.transform_documents([doc])
            for document in split_documents:
                document.metadata[self.id_key] = id
                child_docs.append(document)
        return full_docs, child_docs

    def __add_documents_to_docstore(self, documents: List[Document], chunk_size):
        for i in range(0, len(documents), chunk_size):
            self.docstore.mset(
                key_value_pairs=documents[i : i + chunk_size],
            )

    def __add_documents_to_vectorstore(self, documents: List[Document], chunk_size):
        for i in range(0, len(documents), chunk_size):
            self.vectorstore.add_documents(
                documents=documents[i : i + chunk_size],
            )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
        chunk_size: int = 256,
    ):
        for ending, text_splitter in self.parent_splitters.items():
            proper_filetype_docs = [
                doc for doc in documents if doc.metadata["ItemId"].endswith(ending)
            ]
            if proper_filetype_docs:
                proper_filetype_docs = text_splitter.transform_documents(
                    proper_filetype_docs
                )

            proper_filetype_docs = [
                doc for doc in proper_filetype_docs if len(doc.page_content.strip()) > 0
            ]

            child_splitter = self.child_splitters[ending]
            full_docs, docs = self.__get_full_and_child_docs_with_metadata(
                proper_filetype_docs, child_splitter
            )

            self.__add_documents_to_vectorstore(docs, chunk_size)
            self.__add_documents_to_docstore(full_docs, chunk_size)

        self.docstore.save()

    def __get_new_query_from_extractor(self, query, extractor):
        try:
            _, entities = asyncio.get_event_loop().run_until_complete(
                extractor.extract(Node(properties={"page_content": query}))
            )
            new_query = " ".join(entities) if entities else query
        except Exception as e:
            logging.warning(f"Failed to extract entities: {e}")
            new_query = query
        return new_query

    def __get_ful_documents_from_sub_docs(
        self, sub_docs: List[Document], top_ids: Optional[List[str]] = None
    ) -> List[Document]:
        full_docs = []
        for doc_id, docs in sub_docs.items():
            docstore_docs = self.docstore.mget([doc_id])
            if top_ids and doc_id not in top_ids:
                continue
            if docstore_docs:
                doc = docstore_docs[0]
                if doc:
                    doc.metadata["sub_docs"] = docs
                    full_docs.append(doc)
        return full_docs

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:
        min_score = self.search_kwargs.get("min_score", 0.0)
        results = self.vectorstore.similarity_search_with_score(
            query, k=self.search_kwargs.get("k", 5)
        )

        ids_doc = defaultdict(list)
        for doc, score in results:
            doc_id = doc.metadata.get(self.id_key)
            score = 1.0 - score
            if score > min_score:
                doc.metadata["score"] = score
                ids_doc[doc_id].append(doc)

        return self.__get_ful_documents_from_sub_docs(ids_doc)
