import os
import pickle
from typing import Optional
from langchain_core.stores import BaseStore, InMemoryBaseStore
from langchain_core.documents import Document


class DocumentStore(BaseStore[str, Document]):
    def __init__(
        self,
        store_dir: Optional[str] = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "documentstore.pickle"
        ),
    ):
        self.store_dir = store_dir
        super().__init__()
        if store_dir is None:
            self.docstore = InMemoryBaseStore()
        elif os.path.exists(store_dir):
            with open(store_dir, "rb") as f:
                self.docstore: InMemoryBaseStore = pickle.load(f)
        else:
            self.docstore = InMemoryBaseStore()

    def save(self):
        if self.store_dir:
            with open(self.store_dir, "wb") as f:
                pickle.dump(self.docstore, f)
        else:
            raise ValueError("No store directory provided.")

    def mget(self, keys):
        return self.docstore.mget(keys)

    async def amget(self, keys):
        return await self.docstore.amget(keys)

    def mset(self, key_value_pairs):
        self.docstore.mset(key_value_pairs)

    async def amset(self, key_value_pairs):
        await self.docstore.amset(key_value_pairs)

    def mdelete(self, keys):
        return self.docstore.mdelete(keys)

    async def amdelete(self, keys):
        return await self.docstore.amdelete(keys)

    def yield_keys(self, prefix=None):
        return self.docstore.yield_keys(prefix)

    async def ayield_keys(self, prefix=None):
        return await self.docstore.ayield_keys(prefix)
