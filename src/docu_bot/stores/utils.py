import os
import json
import logging
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.document_loaders import BaseLoader
from docu_bot.constants import MODEL_TYPES
from docu_bot.stores.docstore import DocumentStore
from docu_bot.retrievals.document_retrival import DocumentRetrieval


class LoadedVectorStores:
    def __init__(
        self,
        json_file=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "data",
            "vector_stores.json",
        ),
        embedding_model=MODEL_TYPES.DEFAULT_EMBED_MODEL,
        api_key="None",
    ):

        self._json_file = json_file
        self._vectorstores = {}
        if not os.path.exists(json_file):
            self._json_data = {}
        else:
            with open(json_file, "r") as json_fp:
                self._json_data = json.load(json_fp)
        for repo_or_files, data_path in self._json_data.items():
            if not os.path.exists(data_path):
                logging.warning(
                    f"Data path {data_path} for {repo_or_files} does not exist. Removing from loaded repositories and files."
                )
                self._json_data.pop(repo_or_files)
            else:
                if ".git" in repo_or_files:
                    collection_name = f"{os.path.basename(os.path.dirname(repo_or_files)).split('.git')[0]}-{os.path.basename(repo_or_files)}"
                else:
                    collection_name = os.path.basename(repo_or_files).split(".git")[0]

                self._vectorstores[repo_or_files] = Chroma(
                    collection_name=collection_name,
                    persist_directory=data_path,
                    embedding_function=OpenAIEmbeddings(
                        model=embedding_model,
                        api_key=api_key,
                        base_url=(
                            MODEL_TYPES.DEFAULT_EMBED_LOC
                            if embedding_model == MODEL_TYPES.DEFAULT_EMBED_MODEL
                            else None
                        ),
                    ),
                )

    def add_vectorstore(self, repo_or_files, data_path, vectorstore: Chroma):
        self._json_data[repo_or_files] = data_path
        self._vectorstores[repo_or_files] = vectorstore

        os.makedirs(os.path.dirname(self._json_file), exist_ok=True)

        with open(self._json_file, "w") as f:
            json.dump(self._json_data, f)

    def get_vectorstores(self, repo_or_files) -> List[Chroma]:
        return [
            vector_store
            for cache_repo_file, vector_store in self._vectorstores.items()
            if repo_or_files in cache_repo_file
        ]


def create_vector_store_from_document_loader(
    document_loader: BaseLoader,
    docstore: DocumentStore,
    vectorstores: LoadedVectorStores,
    embedding_model=MODEL_TYPES.DEFAULT_EMBED_MODEL,
    api_key="None",
) -> Chroma:
    if hasattr(document_loader, "filename"):
        collection_name = document_loader.filename
        collection_nameshort = document_loader.filename.split(".git")[0]
    else:
        collection_name = os.path.join(
            document_loader.repo_path, document_loader.branch
        )
        collection_nameshort = f'{os.path.basename(document_loader.repo_path).split(".git")[0]}-{document_loader.branch}'
    if collection_name in vectorstores._json_data:
        return vectorstores._vectorstores[collection_name]

    retrieval = DocumentRetrieval(
        vectorstore=Chroma(
            collection_name=collection_nameshort,
            persist_directory=f"{document_loader.save_path}-vectorstore",
            embedding_function=OpenAIEmbeddings(
                model=embedding_model,
                api_key=api_key,
                base_url=(
                    MODEL_TYPES.DEFAULT_EMBED_LOC
                    if embedding_model == MODEL_TYPES.DEFAULT_EMBED_MODEL
                    else None
                ),
            ),
        ),
        docstore=docstore,
    )

    retrieval.add_documents(document_loader.load())

    vectorstores.add_vectorstore(
        collection_name,
        f"{document_loader.save_path}-vectorstore",
        retrieval.vectorstore,
    )

    return retrieval.vectorstore
