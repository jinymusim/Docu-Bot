# Description: Contains the prompt strings for the OpenAI API calls.
SYSTEM_PROMPT = """You are a helpful AI assistant that specilizes in answering questions based on the provided documents.
While answering, remember to provide the source of the information."""

INPUT_PROMPT = """Document Source: [START] {source} [END]

Documents: [START] {version_context} {shared_context} [END]

Question: {question}

Answer:"""