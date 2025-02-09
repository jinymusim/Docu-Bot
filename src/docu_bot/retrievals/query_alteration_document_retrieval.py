from collections import defaultdict

from docu_bot.retrievals.document_retrival import DocumentRetrieval
from docu_bot.constants import PROMPTS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class QueryAlterationDocumentRetrieval(DocumentRetrieval):

    llm: ChatOpenAI
    query_prompt: str = PROMPTS.QUERY_PROMPT

    def _get_relevant_documents(self, query, *, run_manager):

        template = ChatPromptTemplate(
            [
                ("system", self.query_prompt),
            ]
        )
        num_custom_queires = self.search_kwargs.get("num_custom_queires", 5)
        min_score = self.search_kwargs.get("min_score", 0.5)

        queries = [query]
        for _ in range(num_custom_queires):
            prompt = template.invoke({"query": query})
            msg = self.llm.invoke(prompt)
            queries.append(msg.content)

        ids_doc = defaultdict(list)
        ids_score = defaultdict(list)
        for query in queries:
            results = self.vectorstore.similarity_search_with_score(
                query, k=self.search_kwargs.get("k", 5)
            )
            for doc, score in results:
                doc_id = doc.metadata.get(self.id_key)
                score = 1.0 - score
                if score > min_score:
                    doc.metadata["score"] = score
                    ids_doc[doc_id].append(doc)
                    ids_score[doc_id].append(score)

        top_ids = sorted(ids_score, key=lambda x: sum(ids_score[x]), reverse=True)[
            : self.search_kwargs.get("k", 5)
        ]

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
