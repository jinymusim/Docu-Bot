from typing import List

from langchain_core.vectorstores.base import VectorStore
from langchain_core.documents import Document
from langchain_chroma import Chroma


class MultiVectorStore(VectorStore):
    def __init__(self, chroma_vectors: List[Chroma]):
        self.chroma_vectors = chroma_vectors

    def similarity_search_with_score(
        self, query, k, filter: dict | None = None, **kwargs
    ):
        results = []
        for chroma in self.chroma_vectors:
            results.extend(
                chroma.similarity_search_with_score(query, k, filter, **kwargs)
            )

        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results[:k]

    def similarity_search(self, query, k=4, filter: dict | None = None, **kwargs):
        return [
            doc
            for doc, _ in self.similarity_search_with_score(query, k, filter, kwargs)
        ]

    ## The following methods are not implemented and are here so the class is not abstract

    async def aadd_documents(self, documents, **kwargs):
        return await super().aadd_documents(documents, **kwargs)

    async def aadd_texts(self, texts, metadatas=None, *, ids=None, **kwargs):
        return await super().aadd_texts(texts, metadatas, ids=ids, **kwargs)

    def add_documents(self, documents, **kwargs):
        return super().add_documents(documents, **kwargs)

    def add_texts(self, texts, metadatas=None, *, ids=None, **kwargs):
        return super().add_texts(texts, metadatas, ids=ids, **kwargs)

    async def adelete(self, ids=None, **kwargs):
        return super().adelete(ids, **kwargs)

    async def afrom_documents(self, documents, **kwargs):
        return await super().afrom_documents(documents, **kwargs)

    async def afrom_texts(self, texts, metadatas=None, *, ids=None, **kwargs):
        return await super().afrom_texts(texts, metadatas, ids=ids, **kwargs)

    async def aget_by_ids(self, ids):
        return await super().aget_by_ids(ids)

    async def amax_marginal_relevance_search(
        self, query, k=4, fetch_k=20, lambda_mult=0.5, **kwargs
    ):
        return await super().amax_marginal_relevance_search(
            query, k, fetch_k, lambda_mult, **kwargs
        )

    async def amax_marginal_relevance_search_by_vector(
        self, embedding, k=4, fetch_k=20, lambda_mult=0.5, **kwargs
    ):
        return await super().amax_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, **kwargs
        )

    def as_retriever(self, **kwargs):
        return super().as_retriever(**kwargs)

    async def asearch(self, query, search_type, **kwargs):
        return await super().asearch(query, search_type, **kwargs)

    async def asimilarity_search(self, query, k=4, **kwargs):
        return await super().asimilarity_search(query, k, **kwargs)

    async def asimilarity_search_by_vector(self, embedding, k=4, **kwargs):
        return await super().asimilarity_search_by_vector(embedding, k, **kwargs)

    async def asimilarity_search_with_relevance_scores(self, query, k=4, **kwargs):
        return await super().asimilarity_search_with_relevance_scores(
            query, k, **kwargs
        )

    async def asimilarity_search_with_score(self, *args, **kwargs):
        return await super().asimilarity_search_with_score(*args, **kwargs)

    def delete(self, ids=None, **kwargs):
        return super().delete(ids, **kwargs)

    def from_documents(self, documents, **kwargs):
        return super().from_documents(documents, **kwargs)

    def from_texts(self, texts, metadatas=None, *, ids=None, **kwargs):
        return super().from_texts(texts, metadatas, ids=ids, **kwargs)

    def get_by_ids(self, ids):
        return super().get_by_ids(ids)

    def max_marginal_relevance_search(
        self, query, k=4, fetch_k=20, lambda_mult=0.5, **kwargs
    ):
        return super().max_marginal_relevance_search(
            query, k, fetch_k, lambda_mult, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self, embedding, k=4, fetch_k=20, lambda_mult=0.5, **kwargs
    ):
        return super().max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, **kwargs
        )

    def search(self, query, search_type, **kwargs):
        return super().search(query, search_type, **kwargs)

    def similarity_search_by_vector(self, embedding, k=4, **kwargs):
        return super().similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
        return super().similarity_search_with_relevance_scores(query, k, **kwargs)
