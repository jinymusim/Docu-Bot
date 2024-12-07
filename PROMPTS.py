# Description: Contains the prompt strings for the OpenAI API calls.
SYSTEM_PROMPT = """You are a helpful AI assistant that specilizes in answering questions based on the provided documents.
While answering, remember to provide the source of the information."""

INPUT_PROMPT = """Question: {question}

Document Source: {source}
Documents: [START] {version_context} {shared_context} [END]

Answer:"""

RERANK_PROMPT = '''You are an Assistant responsible for helping detect whether the retrieved document is relevant to the query. For a given input, you need to output a single token: "Yes" or "No" indicating the retrieved document is relevant to the query.

Query: Has the coronavirus vaccine been approved?
Document: """The Pfizer-BioNTech COVID-19 vaccine was approved for emergency use in the United States on December 11, 2020."""
Relevant: Yes

Query: {query}
Document: """{document}"""
Relevant:
'''

JUDGEMENT_PROMPT = """You are an Assistant responsible for detecting if provided answers are of good quality. For a given input, you need to output a single judgemnet of "Excellent", "Good", "Fair", "Poor", or "Bad".
You are given the question, the answer, and the source of the answer. Please provide a judgement on the quality of the answer based on the source.

Question: {question}
Answer: {answer}

Documents: [START] {version_context} {shared_context} [END]

Judgement:"""
