import asyncio
from collections import defaultdict
from typing import List
from docu_bot.retrievals.document_retrival import DocumentRetrieval
from docu_bot.constants import PROMPTS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from ragas.llms import LangchainLLMWrapper

from ragas.testset.transforms.extractors.llm_based import ThemesExtractor


class ThemeRetrieval(DocumentRetrieval):

    llm: ChatOpenAI
    query_prompt: str = PROMPTS.QUERY_PROMPT

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:

        from ragas.testset.graph import Node

        min_score = self.search_kwargs.get("min_score", 0.0)

        extractor = ThemesExtractor(llm=LangchainLLMWrapper(self.llm))
        _, entities = asyncio.get_event_loop().run_until_complete(
            extractor.extract(Node(properties={"page_content": query}))
        )
        new_query = " ".join(entities) if entities else query

        results = self.vectorstore.similarity_search_with_score(
            new_query,
            k=self.search_kwargs.get("k", 5),
        )

        ids_doc = defaultdict(list)
        ids_score = defaultdict(list)
        for doc, score in results:
            doc_id = doc.metadata.get(self.id_key)
            score = 1.0 - score
            if score > min_score:
                doc.metadata["score"] = score
                ids_doc[doc_id].append(doc)
                ids_score[doc_id].append(score)

        top_ids = sorted(
            ids_score, key=lambda x: sum(ids_score[x]) / len(ids_score[x]), reverse=True
        )[: self.search_kwargs.get("k", 5)]

        full_docs = []
        for doc_id, docs in ids_doc.items():
            if doc_id not in top_ids:
                continue
            docstore_docs = self.docstore.mget([doc_id])
            if docstore_docs:
                doc = docstore_docs[0]
                if doc:
                    doc.metadata["sub_docs"] = docs
                    full_docs.append(doc)

        return full_docs
