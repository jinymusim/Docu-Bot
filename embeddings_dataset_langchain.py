from torch.utils.data import Dataset

#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import DirectoryLoader, TextLoader
#from langchain.document_loaders.directory import DirectoryLoader
#from langchain.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
#from langchain.vectorstores.chroma import Chroma
from langchain_openai import  OpenAIEmbeddings
from fuzzywuzzy import fuzz

import os
import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

EMBED_STEP = 256


#class MyEmbeddingFunction(EmbeddingFunction):
#    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2') -> None:
#        super().__init__()
#        self.transformer = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
#        
#    def __call__(self, input: Documents) -> Embeddings:
#        # embed the documents somehow
#        return self.transformer(input)
#    
#    def embed_documents(self, input: Documents):
#        return self.__call__(input)

class EmbeddingsDataset(Dataset):
    def __init__(self, datasource_directory, transformer_model: OpenAIEmbeddings, cache_dir =os.path.join(os.path.dirname(__file__), "chroma-embed-cache") ):
        self.cache_dir = os.path.join(os.path.dirname(__file__), cache_dir )
        self.embedd_function = transformer_model
        self.source_dir = datasource_directory
        self.branch = datasource_directory.split(os.sep)[-1]
        
        exist = os.path.exists(self.cache_dir)
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.vectordb = Chroma(
            collection_name='embedding-data',
            persist_directory=self.cache_dir,
            embedding_function=self.embedd_function   
        )
        if not exist:
        
            data_sources_splitter_pairs = [
                ('*.md',  {'language': Language.MARKDOWN ,'chunk_size' : 400, 'chunk_overlap'  : 150, 'length_function' : len,}),
                ('*.rst',  {'language': Language.RST, 'chunk_size' : 400, 'chunk_overlap'  : 150, 'length_function' : len,}),
                ('*.txt',  {'chunk_size' : 400, 'chunk_overlap'  : 150, 'length_function' : len,}),
                ('*.py', {'language': Language.PYTHON, 'chunk_size' : 100, 'chunk_overlap'  : 0, 'length_function' : len,}),
                ('*.html', {'language': Language.HTML, 'chunk_size' : 400, 'chunk_overlap'  : 150, 'length_function' : len,}),
                ('*.tex', {'language': Language.LATEX, 'chunk_size' : 400, 'chunk_overlap'  : 150, 'length_function' : len,}),
            ]
            

            for ending, kwargs_splitter in data_sources_splitter_pairs:
            
                text_loader_kwargs={'autodetect_encoding': True, "encoding": 'utf-8'}
                loader = DirectoryLoader(datasource_directory, show_progress=True, recursive=True, glob=ending,
                        loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

                if 'language' in kwargs_splitter:
                    text_splitter = RecursiveCharacterTextSplitter.from_language(
                        **kwargs_splitter
                    )
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        **kwargs_splitter
                    )

                documents = loader.load()

                if len(documents) > 0:
                    docs = text_splitter.split_documents(documents)  
                    for i in range(0,len(docs), EMBED_STEP):
                        self.vectordb.add_documents(
                        documents=docs[i:i+EMBED_STEP],
                        ) 
        
        
    
    def __getitem__(self, index):
        return self.vectordb[index]
    
    def __len__(self):
        return self.vectordb._collection.count()
    
    def __remove_full_overhead(self, document:Document):
        filename_rel_path = os.path.relpath(document.metadata['source'], self.source_dir)
        doc_dict = {
            'filename' : filename_rel_path,
            'branch' : self.branch,
            'data' : document.page_content
        }
        
        return doc_dict
    
    def __call__(self, query, k=7, fetch_k=50):
        max_marginal = self.vectordb.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        max_similar = self.vectordb.similarity_search(query, k=k)
        for doc in max_similar:
            is_different = True
            for sim_doc in max_marginal:
                if fuzz.partial_ratio(doc.page_content, sim_doc.page_content) > 95:
                    is_different = False
                    break
            if is_different:
                max_marginal.append(doc)
        
        return list(map(lambda x: self.__remove_full_overhead(x), max_marginal))
    
    def relevant_docs_filename(self, query, k=7, fetch_k=50):
        max_marginal = self.vectordb.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k) + self.vectordb.similarity_search(query, k=k)
        filename_reference_list = []
        filepaths = []
        for doc in max_marginal:
            filename_rel_path = os.path.relpath(doc.metadata['source'], self.source_dir)
            if filename_rel_path in filename_reference_list:
                continue
            else:
                filename_reference_list.append(filename_rel_path)
                filepaths.append(doc.metadata['source'])
        return filepaths
    
    def querry_documents(self, query, k=5, fetch_k=30):  
        documents = self.vectordb.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        docuemnt_filenames = []
        i=0
        while i < len(documents):
            if documents[i].metadata['source'] in docuemnt_filenames:
                documents.pop(i)
                i-=1
            else:
                docuemnt_filenames.append(documents[i].metadata['source'])
            i+=1
        full_documents = []
        for document in documents:
            with open(document.metadata['source'], 'r', encoding='utf-8') as file:
                filename = os.path.relpath(document.metadata['source'], self.source_dir)
                full_documents.append({'filename': filename, 'data' : file.read()})
        
        return full_documents
    
    def querry_documents_small(self, query, k=5, fetch_k=30): 
        documents = self.vectordb.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        full_documents = []
        for document in documents:
            filename = os.path.relpath(document.metadata['source'], self.source_dir)
            full_documents.append({'filename': filename, 'data' : document.page_content})
        
        return full_documents
        
    