from collections import defaultdict

from typing import List
from docu_bot.retrievals.query_alteration_retrieval import (
    QueryAlterationDocumentRetrieval,
)
from docu_bot.constants import PROMPTS
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document


class ContextQueryAlterationDocumentRetrieval(QueryAlterationDocumentRetrieval):

    query_prompt: str = PROMPTS.CONTEXT_QUERY_PROMPT

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:

        llm = self.llm.bind(stop=["query"], max_tokens=30)

        template = ChatPromptTemplate.from_messages(
            [
                ("user", self.query_prompt),
            ]
        )

        min_score = self.search_kwargs.get("min_score", 0.0)
        results = self.vectorstore.similarity_search(
            query, k=self.search_kwargs.get("k", 5)
        )

        queries = []
        for context in results:
            prompt = template.invoke({"query": query, "context": context.page_content})
            msg = llm.invoke(prompt)
            queries.append(msg.content)

        ids_doc = defaultdict(list)
        ids_score = defaultdict(list)
        for sythetic_query in queries:
            results = self.vectorstore.similarity_search_with_score(
                sythetic_query, k=self.search_kwargs.get("k", 5)
            )
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
