from collections import defaultdict

from typing import List
from docu_bot.retrievals.document_retrival import DocumentRetrieval
from docu_bot.constants import PROMPTS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document


class QueryAlterationDocumentRetrieval(DocumentRetrieval):

    llm: ChatOpenAI
    query_prompt: str = PROMPTS.QUERY_PROMPT

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:

        llm = self.llm.bind(stop=["query"], max_tokens=30)

        template = ChatPromptTemplate.from_messages(
            [
                ("user", self.query_prompt),
            ]
        )
        num_custom_queires = self.search_kwargs.get("num_custom_queires", 5)
        min_score = self.search_kwargs.get("min_score", 0.0)

        queries = []
        for _ in range(num_custom_queires):
            prompt = template.invoke({"query": query})
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

        return self.__get_ful_documents_from_sub_docs(ids_doc, top_ids)
