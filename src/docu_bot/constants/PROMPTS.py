# Description: Contains the prompt strings for the OpenAI API calls.
SYSTEM_PROMPT = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

"""

INPUT_PROMPT = """text: \n\n\n
{context}

question: {query}

answer:"""

RERANK_PROMPT = '''You are an assistant responsible for helping detect whether the retrieved document is relevant to the query.
For a given input, you need to output a single token: "Yes" or "No" indicating the relevance of the document to the query.

query: Does the application support file transfer protocols?
document: """dCache provides a GSI-FTP door, which is in effect a GSI authenticated FTP access point to dCache
listing a directory
To list the content of a dCache directory, the GSI-FTP protocol can be used;

[user] $ edg-gridftp-ls gsiftp://gridftp-door.example.org/pnfs/example.org/data/dteam/
"""
relevant: Yes

query: {query}
document: """{context}"""
relevant:'''

QUERY_PROMPT = """You are an assistant responsible for rewriting the query to improve the search results. 
For given query, you need to alter the query to improve the search results. In general the query should be more specific or more general.
The altered query needs to be question of lenght max 30 tokens.

query: What protocols are supported?
altered query: What are the file protocols supported by the application?

query: {query}
altered query: """

CONTEXT_QUERY_PROMPT = """You are an assistant responsible for rewriting the query to improve the search results.
You are given a context to the original query and you need to output am altered query that will improve the search results.
In general the query should be more specific or more general. The altered query needs to be question of lenght max 30 tokens.

query: {query}
context: {context}
altered query: """

GENERATE_DOCUMENT_PROMPT = """You are an Assistant that for given query generates context of how could retrieved document look like.
The generated context must be max 150 tokens long. It is important for the context to be coherent and relevant to the query.

query: {query}
context:"""
