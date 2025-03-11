import asyncio
import logging
from collections import defaultdict
from typing import List
from docu_bot.retrievals.document_retrival import DocumentRetrieval
from docu_bot.constants import PROMPTS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from ragas.llms import LangchainLLMWrapper

from ragas.testset.transforms.extractors.llm_based import NERExtractor


class NerRetrieval(DocumentRetrieval):

    llm: ChatOpenAI
    query_prompt: str = PROMPTS.QUERY_PROMPT

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:

        from ragas.testset.graph import Node

        min_score = self.search_kwargs.get("min_score", 0.0)

        extractor = NERExtractor(
            llm=LangchainLLMWrapper(self.llm),
            max_num_entities=self.search_kwargs.get("max_num_entities", 5),
        )

        new_query = self.__get_new_query_from_extractor(extractor, query)

        results = self.vectorstore.similarity_search_with_score(
            new_query,
            k=self.search_kwargs.get("k", 5),
        ) + self.vectorstore.similarity_search_with_score(
            query,
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

        return self.__get_ful_documents_from_sub_docs(ids_doc, top_ids)
