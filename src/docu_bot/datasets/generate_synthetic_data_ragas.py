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
            name="Curious Developer",
            role_description="A developer who is curious about the codebase and wants to understand the codebase structure.",
        ),
        Persona(
            name="Code Reviewer",
            role_description="A developer who is reviewing the codebase for quality assurance.",
        ),
        Persona(
            name="New Developer",
            role_description="A developer who is new to the codebase and wants to understand the codebase structure.",
        ),
        Persona(
            name="Package Maintainer",
            role_description="A developer who is maintaining the codebase.",
        ),
    ]


def _get_transforms(llm: LangchainLLMWrapper) -> list:
    ner = NERExtractor()
    ner.llm = llm

    return [ner]


async def _get_qurery_distribution(llm: LangchainLLMWrapper) -> list:

    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm), 1.0),
    ]

    for query, _ in distribution:
        prompts = await query.adapt_prompts("english", llm=llm)
        query.set_prompts(**prompts)

    return distribution


def create_generator(
    llm: LangchainLLMWrapper, embeddings: LangchainEmbeddingsWrapper, personas=None
) -> TestsetGenerator:
    if personas is None:
        personas = _get_personas()

    return TestsetGenerator(
        llm=llm,
        embeddings=embeddings,
        persona_list=personas,
    )


async def generate_dataset(
    generator: TestsetGenerator,
    documents,
    dataset_size,
    transforms=None,
    query_distribution=None,
):
    if transforms is None:
        transforms = _get_transforms(generator.llm)

    if query_distribution is None:
        query_distribution = await _get_qurery_distribution(generator.llm)

    dataset = generator.generate_with_langchain_docs(
        documents=documents,
        dataset_size=dataset_size,
        transforms=transforms,
        query_distribution=query_distribution,
    )

    return dataset
