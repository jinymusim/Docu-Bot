from math import exp
from collections import defaultdict

from docu_bot.retrievals.document_retrival import DocumentRetrieval
from docu_bot.constants import PROMPTS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class RerankDocumentRetrieval(DocumentRetrieval):

    llm: ChatOpenAI
    rerank_prompt: str = PROMPTS.RERANK_PROMPT

    def _get_relevant_documents(self, query, *, run_manager):

        template = ChatPromptTemplate(
            [
                ("system", self.rerank_prompt),
            ]
        )

        llm = self.llm.bind(logprobs=True, max_tokens=2)

        min_score = self.search_kwargs.get("min_score", 0.1)
        results = self.vectorstore.similarity_search(
            query, k=self.search_kwargs.get("k", 5)
        )

        ids_doc = defaultdict(list)
        for doc in results:
            doc_id = doc.metadata.get(self.id_key)

            prompt = template.invoke({"query": query, "document": doc.page_content})
            msg = llm.invoke(prompt)
            score = 0
            if msg.response_metadata["logprobs"].get("Yes", None):
                score = exp(msg.response_metadata["logprobs"]["Yes"])

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
