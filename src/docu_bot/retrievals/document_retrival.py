import uuid

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from typing import List, Optional, Tuple
from collections import defaultdict

from langchain.retrievers import MultiVectorRetriever
from langchain_core.documents import Document, BaseDocumentTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class DocumentRetrieval(MultiVectorRetriever):

    splitters: List[Tuple[str, BaseDocumentTransformer]] = [
        (
            ".md",
            RecursiveCharacterTextSplitter.from_language(
                Language.MARKDOWN,
                chunk_size=300,
                chunk_overlap=100,
                length_function=len,
            ),
        ),
        (
            ".rst",
            RecursiveCharacterTextSplitter.from_language(
                Language.RST, chunk_size=300, chunk_overlap=100, length_function=len
            ),
        ),
        # (
        #    ".py",
        #    RecursiveCharacterTextSplitter.from_language(
        #        Language.PYTHON, chunk_size=300, chunk_overlap=100, length_function=len
        #    ),
        # ),
        (
            ".pdf",
            RecursiveCharacterTextSplitter(
                chunk_size=300, chunk_overlap=100, length_function=len
            ),
        ),
    ]

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
        chunk_size: int = 256,
    ):
        for ending, text_splitter in self.splitters:
            proper_filetype_docs = [
                doc for doc in documents if doc.metadata["ItemId"].endswith(ending)
            ]
            if proper_filetype_docs:
                split_documents = text_splitter.transform_documents(
                    proper_filetype_docs
                )
                # Filter Empty Documents
                index = 0
                while index < len(split_documents):
                    if len(split_documents[index].page_content.strip()) == 0:
                        split_documents.pop(index)
                        index -= 1
                    index += 1
                ids = [str(uuid.uuid4()) for _ in range(len(split_documents))]
                # Embed Documents
                docs = []
                full_docs = []

                for i, doc in enumerate(split_documents):
                    _id = ids[i]
                    doc.metadata[self.id_key] = _id
                    docs.append(doc)
                    full_docs.append((_id, doc))

                for i in range(0, len(docs), chunk_size):
                    self.vectorstore.add_documents(
                        documents=docs[i : i + chunk_size],
                    )
                    if add_to_docstore:
                        self.docstore.mset(
                            key_value_pairs=full_docs[i : i + chunk_size],
                        )

    def _get_relevant_documents(self, query, *, run_manager):
        min_score = self.search_kwargs.get("min_score", 0.5)
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
