from collections import defaultdict
from docu_bot.constants import PROMPTS
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    SemanticSimilarity,
    ContextEntityRecall,
)
from datasets import Dataset
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from ragas.evaluation import evaluate


def evaluate(
    dataset,
    llm,
    document_retriever,
    prompt=PROMPTS.INPUT_PROMPT,
    metrics = [ 
        FactualCorrectness(),
        Faithfulness(),
        SemanticSimilarity(),
        ContextEntityRecall(),   
    ]
):
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",PROMPTS.SYSTEM_PROMPT),
            ("user", prompt)
        ]
    )
            
    document_chain = create_stuff_documents_chain(llm, qa_prompt) 
    retrival_chain = create_retrieval_chain(document_retriever, document_chain)
    
    output_dataset = defaultdict(list)
    for sample in dataset.samples:
        output_dataset["user_input"].append(sample.eval_sample.user_input)
        output_dataset["reference"].append(sample.eval_sample.reference)
        output_dataset["response"].append(
            retrival_chain.invoke({"query": sample.eval_sample.user_input})['answer']
        )
        output_dataset["retreived_contexts"].append(
            [document.page_content for document in document_retriever.invoke({"query": sample.eval_sample.user_input})]
        )
        
    output_dataset = Dataset.from_dict(output_dataset)
    return evaluate(dataset= output_dataset, metrics=metrics)