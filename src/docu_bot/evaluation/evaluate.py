from collections import defaultdict
from docu_bot.constants import PROMPTS
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
    SemanticSimilarity,
    NonLLMContextRecall,
    LLMContextPrecisionWithReference,
    NonLLMContextPrecisionWithReference,
    ContextEntityRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from ragas.evaluation import evaluate
from docu_bot.utils import format_docs


class Evaluator:

    def __init__(
        self,
        evaluator_llm: ChatOpenAI,
        evaluator_embedding_model: OpenAIEmbeddings,
        metrics=[
            FactualCorrectness(),
            Faithfulness(),
            LLMContextRecall(),
            SemanticSimilarity(),
            NonLLMContextRecall(),
            LLMContextPrecisionWithReference(),
            NonLLMContextPrecisionWithReference(),
            ContextEntityRecall(),
        ],
    ):
        self.evaluator_llm = LangchainLLMWrapper(evaluator_llm)
        self.evaluator_embedding_model = LangchainEmbeddingsWrapper(
            evaluator_embedding_model
        )
        self.metrics = metrics

    def evaluate_configuration(
        self,
        dataset,
        rag_llm,
        document_retriever,
        prompt=PROMPTS.SYSTEM_PROMPT,
    ):
        chat_prompt = ChatPromptTemplate.from_messages(
            [("user", prompt + PROMPTS.INPUT_PROMPT)]
        )

        rag_chain = (
            {
                "context": document_retriever | format_docs,
                "query": RunnablePassthrough(),
            }
            | chat_prompt
            | rag_llm
            | StrOutputParser()
        )

        output_dataset = defaultdict(list)
        for sample in dataset.samples:
            output_dataset["user_input"].append(sample.eval_sample.user_input)
            output_dataset["reference"].append(sample.eval_sample.reference)
            output_dataset["reference_contexts"].append(
                sample.eval_sample.reference_contexts
            )
            output_dataset["response"].append(
                rag_chain.invoke(sample.eval_sample.user_input)
            )
            context = document_retriever.invoke(sample.eval_sample.user_input)
            if len(context) == 0:
                output_dataset["retrieved_contexts"].append([""])
            else:
                output_dataset["retrieved_contexts"].append(
                    [document.page_content for document in context]
                )

        output_dataset = Dataset.from_dict(output_dataset)
        return evaluate(
            dataset=output_dataset,
            metrics=self.metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embedding_model,
        )
