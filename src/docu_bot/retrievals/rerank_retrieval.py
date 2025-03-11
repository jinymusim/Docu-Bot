from math import exp
from collections import defaultdict

from typing import List
from docu_bot.retrievals.document_retrival import DocumentRetrieval
from docu_bot.constants import PROMPTS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document


class RerankDocumentRetrieval(DocumentRetrieval):

    llm: ChatOpenAI
    rerank_prompt: str = PROMPTS.RERANK_PROMPT

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:

        template = ChatPromptTemplate(
            [
                ("user", self.rerank_prompt),
            ]
        )

        llm = self.llm.bind(logprobs=True, max_tokens=3)

        min_score = self.search_kwargs.get("min_score", 0.0)
        results = self.vectorstore.similarity_search_with_score(
            query, k=self.search_kwargs.get("ask", 10)
        )

        ids_doc = defaultdict(list)
        ids_score = defaultdict(list)
        for doc, sim_score in results:
            doc_id = doc.metadata.get(self.id_key)

            prompt = template.invoke({"query": query, "context": doc.page_content})
            msg = llm.invoke(prompt)
            score = -1.0
            log_probs = msg.response_metadata.get("logprobs", {})
            if log_probs and "Yes" in log_probs.keys():
                score = exp(log_probs["Yes"])
            elif "Yes" in msg.content:
                score = 1.0 - sim_score

            if score > min_score:
                doc.metadata["score"] = score
                ids_doc[doc_id].append(doc)
                ids_score[doc_id].append(score)

        top_ids = sorted(
            ids_score, key=lambda x: sum(ids_score[x]) / len(ids_score[x]), reverse=True
        )[: self.search_kwargs.get("k", 5)]

        return self.__get_ful_documents_from_sub_docs(ids_doc, top_ids)
