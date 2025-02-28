import uuid
import asyncio
import logging
from typing import Dict, Optional, List
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain.retrievers import MultiVectorRetriever
from langchain_core.documents import Document, BaseDocumentTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from ragas.llms import LangchainLLMWrapper
from ragas.testset.graph import Node
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor


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

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
        chunk_size: int = 256,
    ):
        extractors = (
            [
                NERExtractor(llm=LangchainLLMWrapper(self.llm)),
                ThemesExtractor(llm=LangchainLLMWrapper(self.llm)),
            ]
            if self.llm
            else []
        )
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
            docs = []
            full_docs = []
            for norm_page in proper_filetype_docs:
                id = str(uuid.uuid4())
                full_docs.append((id, norm_page))
                split_documents = child_splitter.transform_documents([norm_page])
                split_documents = [
                    doc for doc in split_documents if len(doc.page_content.strip()) > 0
                ]
                for document in split_documents:
                    document.metadata[self.id_key] = id
                    for extractor in extractors:
                        try:
                            (
                                _,
                                extraxted_entities,
                            ) = asyncio.get_event_loop().run_until_complete(
                                extractor.extract(
                                    Node(
                                        properties={
                                            "page_content": document.page_content
                                        }
                                    )
                                )
                            )
                            if extraxted_entities:
                                for entity in extraxted_entities:
                                    document.metadata[entity] = True
                        except Exception as e:
                            logging.warning(f"Failed to extract entities: {e}")
                    docs.append(document)

            for i in range(0, len(docs), chunk_size):
                self.vectorstore.add_documents(
                    documents=docs[i : i + chunk_size],
                )
            for i in range(0, len(full_docs), chunk_size):
                self.docstore.mset(
                    key_value_pairs=full_docs[i : i + chunk_size],
                )
        self.docstore.save()

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

        full_docs = []
        for doc_id, docs in ids_doc.items():
            docstore_docs = self.docstore.mget([doc_id])
            if docstore_docs:
                doc = docstore_docs[0]
                if doc:
                    doc.metadata["sub_docs"] = docs
                    full_docs.append(doc)

        return full_docs
