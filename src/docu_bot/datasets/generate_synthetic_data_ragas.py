from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor


def _get_personas() -> list[Persona]:
    return [
        Persona(
            name="New Developer",
            role_description="A developer who is new to the codebase and wants to understand the codebase structure.",
        ),
        Persona(
            name="Product user",
            role_description="A user who wants to understand the product's features.",
        ),
    ]


def _get_transforms(llm: LangchainLLMWrapper) -> list:
    ner = NERExtractor(llm=llm)

    return [ner]


def _get_qurery_distribution(llm: LangchainLLMWrapper) -> list:
    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 1.0),
    ]

    return distribution


def create_generator(
    llm: LangchainLLMWrapper, embedding_model: LangchainEmbeddingsWrapper, personas=None
) -> TestsetGenerator:
    if personas is None:
        personas = _get_personas()

    return TestsetGenerator(
        llm=llm,
        embedding_model=embedding_model,
        persona_list=personas,
    )


def generate_dataset(
    generator: TestsetGenerator,
    documents,
    dataset_size,
    transforms=None,
    query_distribution=None,
):
    if transforms is None:
        transforms = _get_transforms(generator.llm)

    if query_distribution is None:
        query_distribution = _get_qurery_distribution(generator.llm)

    dataset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=dataset_size,
        transforms=transforms,
        query_distribution=query_distribution,
    )

    return dataset
