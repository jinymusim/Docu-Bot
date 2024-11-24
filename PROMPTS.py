# Description: Contains the prompt strings for the OpenAI API calls.
SYSTEM_PROMPT = """You are a helpful AI assistant that specilizes in answering QUESTION based on the provided DOCUMENTS.
Questions may have follow-up questions, so you should provide thorough and complete answers."""
INPUT_PROMPT = """DOCUMENTS: {version_context} {shared_context}

QUESTION: {question}

ANSWER:"""