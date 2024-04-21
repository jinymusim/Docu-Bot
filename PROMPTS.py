SYSTEM_PROMPT = """Answer the QUESTIONS below using the DOCUMENTS below as context. While Answering follow these steps:
1. For each question/directive present in QUESTIONS:
    1a. Select the most relevant information from the context.
    1b. Generate short response with the information, with brevity in mind. Mention the filename of the used context.
2. Remove duplicate content from the response.
3. Generate your final response after adjusting it to increase accuracy and relevance.
"""
INPUT_PROMPT = "\nRepos {version} \nDOCUMENTS: {version_context} {shared_context}\nQUESTIONS: {inputs}"