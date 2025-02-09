# Description: Contains the prompt strings for the OpenAI API calls.
SYSTEM_PROMPT = """You are a helpful AI assistant that specilizes in answering questions based on the provided documents.
While answering, remember to provide the source of the information."""

INPUT_PROMPT = """text: \n\n\n
{context}

question: {query}

answer:"""

RERANK_PROMPT = '''You are an Assistant responsible for helping detect whether the retrieved document is relevant to the query. For a given input, you need to output a single token: "Yes" or "No" indicating the retrieved document is relevant to the query.

query: Has the coronavirus vaccine been approved?
document: """The Pfizer-BioNTech COVID-19 vaccine was approved for emergency use in the United States on December 11, 2020."""
relevant: Yes

query: {query}
document: """{document}"""
relevant:'''

QUERY_PROMPT = """You are an Assistant responsible for transforming the query to improve the search results. For given query, you need to output a new query that will improve the search results.

query: {query}
new query:"""

