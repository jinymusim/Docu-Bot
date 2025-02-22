import time
from typing import Tuple, List
from tqdm import tqdm
from collections import defaultdict
from datasets import Dataset
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
from ragas.evaluation import evaluate, EvaluationResult
from ragas.testset.synthesizers.testset_schema import Testset
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import MultiVectorRetriever
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from docu_bot.constants import PROMPTS
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
        dataset: Testset,
        rag_llm: ChatOpenAI,
        document_retriever: MultiVectorRetriever,
        prompt=PROMPTS.SYSTEM_PROMPT,
        use_tqdm=True,
    ) -> Tuple[EvaluationResult, List]:
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

        time_data = []
        output_dataset = defaultdict(list)
        if use_tqdm:
            sample_iterate = tqdm(dataset.samples)
        else:
            sample_iterate = dataset.samples

        for sample in sample_iterate:
            gen_ts_start = time.time()
            output_dataset["user_input"].append(sample.eval_sample.user_input)
            output_dataset["reference"].append(sample.eval_sample.reference)
            output_dataset["reference_contexts"].append(
                sample.eval_sample.reference_contexts
            )
            output_dataset["response"].append(
                rag_chain.invoke(sample.eval_sample.user_input)
            )
            context = document_retriever.invoke(sample.eval_sample.user_input)
            gen_ts_end = time.time()
            if len(context) == 0:
                output_dataset["retrieved_contexts"].append([""])
            else:
                output_dataset["retrieved_contexts"].append(
                    [document.page_content for document in context]
                )
            time_data.append(gen_ts_end - gen_ts_start)

        output_dataset = Dataset.from_dict(output_dataset)

        eval_result = evaluate(
            dataset=output_dataset,
            metrics=self.metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embedding_model,
        )
        return eval_result, time_data
