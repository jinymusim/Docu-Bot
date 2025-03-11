import os
import zipfile
import shutil
import pypdf
from pathlib import Path
from typing import Optional, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from docu_bot.document_loaders.utils import (
    LoadedRepositoriesAndFiles,
    PERMITED_FILE_EXTENSIONS,
)


class ZipDocumentLoader(BaseLoader):
    def __init__(
        self,
        temp_file: str,
        loaded_repositories_and_files: LoadedRepositoriesAndFiles,
        cache_dir: Optional[str] = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "document_cache"
        ),
        save_path: Optional[str] = None,
    ):
        self.temp_file = temp_file
        self.filename = os.path.basename(temp_file)
        self.loaded_repositories_and_files = loaded_repositories_and_files

        save_path = self._get_save_path_if_cached()
        if save_path is None and cache_dir is None:
            raise ValueError("No save path provided and no cache directory provided.")

        self.save_path = self._create_save_path(save_path, cache_dir, self.filename)
        self._extract_zip()

    def _get_save_path_if_cached(self) -> Optional[str]:
        if self.filename in self.loaded_repositories_and_files._json_data:
            return self.loaded_repositories_and_files._json_data[self.filename]
        return None

    def _create_save_path(self, save_path, cache_dir, filename) -> str:
        if save_path:
            return save_path
        return os.path.abspath(os.path.join(cache_dir, filename.split(".")[0]))

    def _extract_zip(self):
        os.makedirs(self.save_path, exist_ok=True)
        if self.temp_file.endswith(".zip"):
            zf = zipfile.ZipFile(self.temp_file, "r")
            zf.extractall(self.save_path)
        else:
            shutil.copy(self.temp_file, os.path.join(self.save_path, self.filename))

        self.loaded_repositories_and_files.add_directory(self.filename, self.save_path)

    def __load_pdf(self, file_path: Path) -> Iterator[Document]:
        reader = pypdf.PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            yield Document(
                page_content=page.extract_text(),
                metadata={
                    "ItemId": str(file_path),
                    "source": "user",
                    "page_number": i + 1,
                },
            )

    def __load_generic_textfile(self, file_path: Path) -> Iterator[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        yield Document(
            page_content=text,
            metadata={"ItemId": str(file_path), "source": "user"},
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
                    yield from self.__load_pdf(file_path)
                else:
                    yield from self.__load_generic_textfile(file_path)
