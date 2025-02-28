import os
import git
import pypdf
import logging
from pathlib import Path
from typing import Optional, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from docu_bot.document_loaders.utils import (
    LoadedRepositoriesAndFiles,
    get_file_link,
    PERMITED_FILE_EXTENSIONS,
)


class GitDocumentLoader(BaseLoader):
    def __init__(
        self,
        repo_path: str,
        branch: str,
        loaded_repositories_and_files: LoadedRepositoriesAndFiles,
        cache_dir: Optional[str] = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "document_cache"
        ),
        save_path: Optional[str] = None,
    ):
        self.repo_path = repo_path
        self.branch = branch
        self.loaded_repositories_and_files = loaded_repositories_and_files

        save_path = self._get_save_path_if_cached()
        if save_path is None and cache_dir is None:
            raise ValueError("No save path provided and no cache directory provided.")

        self.save_path = self._create_save_path(save_path, cache_dir, repo_path, branch)
        self._clone_repository()

    def _clone_repository(self):
        if (
            os.path.join(self.repo_path, self.branch)
            in self.loaded_repositories_and_files._json_data
        ):
            pass
        elif self.repo_path.endswith(".git"):
            os.makedirs(self.save_path, exist_ok=True)
            try:
                git.Repo.clone_from(
                    self.repo_path,
                    self.save_path,
                    branch=self.branch,
                )
            except git.exc.GitCommandError as e:
                logging.warning(
                    f"Error while cloning repository {self.repo_path}: {e}. Trying to clone through os.system."
                )
                os.system(f"git clone {self.repo_path} {self.save_path}")
            self.loaded_repositories_and_files.add_directory(
                os.path.join(self.repo_path, self.branch), self.save_path
            )
        else:
            raise ValueError(
                "Repo path must be a git repository or a path to a git repository."
            )

    def _get_save_path_if_cached(self) -> Optional[str]:
        if (
            os.path.join(self.repo_path, self.branch)
            in self.loaded_repositories_and_files._json_data
        ):
            return self.loaded_repositories_and_files._json_data[
                os.path.join(self.repo_path, self.branch)
            ]
        return None

    def _create_save_path(self, save_path, cache_dir, repo_path, branch) -> str:
        if save_path:
            return save_path
        return os.path.abspath(
            os.path.join(
                cache_dir, os.path.basename(repo_path).split(".git")[0], branch
            )
        )

    def lazy_load(self) -> Iterator[Document]:
        # Get only files
        files_path = Path(self.save_path).rglob("*.*")
        for file_path in files_path:
            if (
                file_path.exists()
                and file_path.is_file()
                and file_path.suffix in PERMITED_FILE_EXTENSIONS
            ):
                if file_path.suffix == ".pdf":
                    reader = pypdf.PdfReader(file_path)
                    for i, page in enumerate(reader.pages):
                        yield Document(
                            page_content=page.extract_text(),
                            metadata={
                                "ItemId": get_file_link(
                                    os.path.join(self.repo_path, self.branch),
                                    file_path,
                                    self.save_path,
                                ),
                                "source": "git",
                                "page_number": i + 1,
                            },
                        )
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    yield Document(
                        page_content=text,
                        metadata={
                            "ItemId": get_file_link(
                                os.path.join(self.repo_path, self.branch),
                                file_path,
                                self.save_path,
                            ),
                            "source": "git",
                        },
                    )
