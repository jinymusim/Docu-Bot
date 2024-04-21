from embeddings_dataset_langchain import EmbeddingsDataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TextIteratorStreamer
from langchain_openai import  OpenAIEmbeddings
from openai import OpenAI
from threading import Thread
from zipfile import ZipFile, BadZipFile
import PROMPTS
import MODEL_TYPES
import CONTEXT_SIZE
import os
import torch
import json
import git
import subprocess
import importlib.util
import argparse
import shutil


def supports_flash_attention():
    """Check if a GPU supports FlashAttention."""
    major, minor = torch.cuda.get_device_capability(0)
    
    flash_attention = False if  importlib.util.find_spec('flash_attn') is None else True
    
    
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.x)
    is_sm8x = major == 8 and minor >= 0
    is_sm9x = major == 9 and minor >= 0

    return (is_sm8x or is_sm9x) and flash_attention

class RetrivalAugment:
    
    def __init__(self, cache_repo_list = os.path.join(os.path.dirname(__file__), 'cached_repos.json'), cache_dir= os.path.join(os.path.dirname(__file__), 'py_cache'), args:argparse.Namespace = None) -> None:
        # Embedding Model to be Used in Document and Querry Embeddings
        
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Store requested cache list location
        self.cache_repo_list = cache_repo_list
        # Create cache list if not present, otherwise load it
        if not os.path.exists(self.cache_repo_list):
            self.cached = {'cached_repos': {}, 'cached_shared' : []}
            json.dump(self.cached, open(self.cache_repo_list, 'w+'), indent=6)
        else:
            self.cached = json.load(open(self.cache_repo_list, 'r'))
            
        # Template variable for shared documents that could be uploaded
        self.shared_documents = {}
        self.version_specific_documents = {}
        
    
    def __load_cached_branches(self, git_repos, branches,api_key:str):
        for repo in git_repos:
            self.version_specific_documents[repo] = {}
            normalized_github_path = repo.removesuffix('.git')
            _ ,repo_rel_name = os.path.split(normalized_github_path)
            for branch in branches:
                brach_repo, true_ver = os.path.split(branch)
                if brach_repo in repo:
                    self.version_specific_documents[repo][true_ver] = EmbeddingsDataset(os.path.join(self.cache_dir , repo_rel_name, true_ver), 
                                                                cache_dir=os.path.join(self.cache_dir , f'{repo_rel_name}-{true_ver}-embed'), 
                                                                transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key=api_key))
        
    def __load_cached_shared(self, shared, api_key:str):
                
        for zip_name in shared:
            self.shared_documents[zip_name] = EmbeddingsDataset(os.path.join(self.cache_dir, zip_name.removesuffix('.zip')), 
                                                    cache_dir=os.path.join(self.cache_dir, f'{zip_name.removesuffix(".zip")}-embed'), 
                                                    transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key=api_key))
      
    def _get_repo_branches(self, base_repo: str):
        # Check if proper git repo format
        if not base_repo.endswith('.git'):
            return []
        branches = []
        g = git.cmd.Git()
        # Try to lookup all of git branches
        try:
            for ref in g.ls_remote('--heads',base_repo).split('\n'):
                branches.append(ref.split('/')[-1])
            return branches
        except Exception as e:
            print(e)
            return []
         
    def _get_cached_repos(self):
        # Return all cached repos
        return list(self.cached['cached_repos'].keys())
    
    def _get_cached_shared(self):
        # Return all caches secondary directories
        return list(self.cached['cached_shared'])
      
    def _check_branch_cache_short(self, base_repo: str):
        # Return all cached branches to given repo
        if not base_repo.endswith('.git'):
            return []
        if base_repo in self.cached['cached_repos'].keys():
            return self.cached['cached_repos'][base_repo]
        return []
    
    def _check_branch_cache(self, base_repos: list[str]):
        # Return all cached branches to given repos
        if not isinstance(base_repos, list):
            _, repo_rel_name = os.path.split(base_repos.removesuffix('.git'))
            repo_dir = os.path.dirname(base_repos)
            _, repo_rel_dir =  os.path.split(repo_dir)
            if base_repos in self.cached['cached_repos'].keys():
                return list(map(lambda x: f'{repo_rel_dir}/{repo_rel_name}/{x}', self.cached['cached_repos'][base_repos])) 
            return []
        cache_branches = []
        for repo in base_repos:

            _, repo_rel_name = os.path.split(repo.removesuffix('.git'))
            repo_dir = os.path.dirname(repo)
            _, repo_rel_dir =  os.path.split(repo_dir)
            
            if repo in self.cached['cached_repos'].keys():
                cache_branches += list(map(lambda x: f'{repo_rel_dir}/{repo_rel_name}/{x}', self.cached['cached_repos'][repo])) 
        return cache_branches
            
    def _add_following_repo_branches(self, base_repo:str, repo_branches: list[str], api_key:str = None):
        if not base_repo.endswith('.git'):
            return
        if api_key.strip() == '' and 'OPENAI_API_KEY' in os.environ.keys() and os.getenv('OPENAI_API_KEY').strip() != '':
            api_key = os.getenv('OPENAI_API_KEY')
        elif api_key.strip() == '':
            return
        normalized_github_path = base_repo.removesuffix('.git')
        _ ,repo_rel_name = os.path.split(normalized_github_path)
        if not base_repo in self.cached['cached_repos'].keys():
            self.cached['cached_repos'][base_repo] = []
        if not base_repo in self.version_specific_documents.keys():
            self.version_specific_documents[base_repo] = {}
        for requested_branch in repo_branches:
            if requested_branch in self.cached['cached_repos'][base_repo]:
                if not requested_branch in self.version_specific_documents[base_repo].keys():
                    self.version_specific_documents[base_repo][requested_branch] = EmbeddingsDataset(os.path.join(self.cache_dir , repo_rel_name, requested_branch), 
                                                                                      cache_dir=os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed'), 
                                                                                      transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key=api_key))
            else:
                # Try also https://code.it4i.cz/sccs/docs.it4i.cz/-/archive/master/docs.it4i.cz-master.zip
                subprocess.run(f'curl -L -o {os.path.abspath(os.path.join(self.cache_dir , requested_branch + ".zip"))} {normalized_github_path}/zipball/{requested_branch}', shell=True)
                try:
                    zf = ZipFile(os.path.join(self.cache_dir , f'{requested_branch}.zip'), 'r') 
                except BadZipFile as e:
                    os.remove(os.path.join(self.cache_dir , f'{requested_branch}.zip'))
                    subprocess.run(f'curl -L -o {os.path.abspath(os.path.join(self.cache_dir , requested_branch + ".zip"))} {normalized_github_path}/-/archive/{requested_branch}/zipfile.zip', shell=True)
                
                if os.path.exists(os.path.join(self.cache_dir , f'{requested_branch}.zip')):
                    try:
                        zf = ZipFile(os.path.join(self.cache_dir , f'{requested_branch}.zip'), 'r') 
                        
                        os.makedirs(os.path.join(self.cache_dir , repo_rel_name), exist_ok=True)
                        if os.path.exists(os.path.join(self.cache_dir , repo_rel_name, requested_branch)):
                            shutil.rmtree(os.path.join(self.cache_dir , repo_rel_name, requested_branch))
                        
                        filenames = list(filter(lambda x: x.endswith('.txt') or x.endswith('.md') or x.endswith('.rst'), zf.namelist()) )

                        zf.extractall(os.path.join(self.cache_dir , repo_rel_name), members=filenames)
                        shutil.move(os.path.join(self.cache_dir , repo_rel_name, zf.namelist()[0]),
                                  os.path.join(self.cache_dir , repo_rel_name, requested_branch))
                        if os.path.exists(os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed')):
                            shutil.rmtree(os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed'))
                        
                        self.version_specific_documents[base_repo][requested_branch] = EmbeddingsDataset(os.path.join(self.cache_dir , repo_rel_name, requested_branch), 
                                                                                          cache_dir=os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed'), 
                                                                                          transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key=api_key))
                        self.cached['cached_repos'][base_repo].append(requested_branch)
                        zf.close()
                        os.remove(os.path.join(self.cache_dir , f'{requested_branch}.zip'))
                    except Exception as e:
                        print(e)
                        os.remove(os.path.join(self.cache_dir , f'{requested_branch}.zip'))
        # Remove Key entry if embedding failed
        if len(self.cached['cached_repos'][base_repo]) == 0:
            self.cached['cached_repos'].pop(base_repo)
            self.version_specific_documents.pop(base_repo)
        
        json.dump(self.cached, open(self.cache_repo_list, 'w+'), indent=6)
        
    def _add_following_zip(self, zip_info:str, api_key:str = None):
        if api_key.strip() == '' and 'OPENAI_API_KEY' in os.environ.keys() and os.getenv('OPENAI_API_KEY').strip() != '':
            api_key = os.getenv('OPENAI_API_KEY')
        elif api_key.strip() == '':
            return
        _, zip_name = os.path.split(zip_info)
        if not zip_name.endswith('.zip') or zip_name in self.cached['cached_shared']:
            os.remove(zip_info)
            return
        else:
            shutil.move(zip_info, os.path.join(self.cache_dir , zip_name))
            try:
                zf = ZipFile(os.path.join(self.cache_dir , zip_name), 'r') 
                
                if os.path.exists(os.path.join(self.cache_dir , zip_name.removesuffix('.zip'))):
                    shutil.rmtree(os.path.join(self.cache_dir , zip_name.removesuffix('.zip')))
                    
                os.makedirs(os.path.join(self.cache_dir , zip_name.removesuffix('.zip')), exist_ok=True)
                    
                filenames = list(filter(lambda x: x.endswith('.txt') or x.endswith('.md') or x.endswith('.rst'), zf.namelist()) )
                zf.extractall(os.path.join(self.cache_dir , zip_name.removesuffix('.zip')), members=filenames)
                
                if os.path.exists(os.path.join(self.cache_dir , f'{zip_name.removesuffix(".zip")}-embed')):
                    shutil.rmtree(os.path.join(self.cache_dir , f'{zip_name.removesuffix(".zip")}-embed'))
                    
                self.shared_documents[zip_name] = EmbeddingsDataset(os.path.join(self.cache_dir , zip_name.removesuffix('.zip')), 
                                                    cache_dir=os.path.join(self.cache_dir , f'{zip_name.removesuffix(".zip")}-embed'), 
                                                    transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key=api_key))
                self.cached['cached_shared'].append(zip_name)
                zf.close()
                os.remove(os.path.join(self.cache_dir , zip_name))   
            except Exception as e:
                print(e)
                os.remove(os.path.join(self.cache_dir , zip_name))
                
        json.dump(self.cached, open(self.cache_repo_list, 'w+'), indent=6)
        
    def _get_relevant_docs(self, git_repos: list[str], versions: list[str], inputs, api_key:str = None) -> str:
        if api_key.strip() == '' and 'OPENAI_API_KEY' in os.environ.keys() and os.getenv('OPENAI_API_KEY').strip() != '':
            api_key = os.getenv('OPENAI_API_KEY')
        elif api_key.strip() == '':
            return "No API Key Provided"
        
        self.__load_cached_branches(git_repos,versions,api_key=api_key)
        
        result_string = "### Most Relevant Documents"
        for repo in git_repos:
            
            _, repo_rel_name = os.path.split(repo.removesuffix('.git'))
            repo_dir = os.path.dirname(repo)
            _, repo_rel_dir =  os.path.split(repo_dir)
            
            for version in versions:
                if f"{repo_rel_dir}/{repo_rel_name}" in version:
                    _, true_ver = os.path.split(version)    
                    relevant_docs = (self.version_specific_documents[repo][true_ver]).relevant_docs_filename(inputs, 
                                                                                                                k=max(1, CONTEXT_SIZE.GIT_DOCUMENTS//len(versions)), 
                                                                                                                fetch_k=max(1, CONTEXT_SIZE.GIT_DIVERSE_K//len(versions)))
                    full_paths = []
                    for path in relevant_docs:
                        
                        rel_file_path:str = path.split(true_ver)[-1].replace(os.sep, '/')
                        rel_file_path_norm = rel_file_path.removeprefix('/')
                        repo_name = repo.removesuffix('.git')
                        full_paths.append(f'[{rel_file_path_norm}]({repo_name}/blob/{true_ver}{rel_file_path})' )
                    full_paths = sorted(full_paths)
                    result_string += f'\n #### Repo {version} \n' + '  \n'.join(full_paths)
            
        return result_string
                           
            
    def __call__(self, git_repos: list[str] = None, versions= None, inputs = '', shared=None, temperature: float= 0.2, api_key:str = None, model:str = 'gpt-3.5-turbo', system_prompt:str = PROMPTS.SYSTEM_PROMPT):    
        if len(git_repos) == 0 and len(shared) == 0:
            yield 'I was not given any documentation path to work from.'
            return
        
        if api_key.strip() == '' and 'OPENAI_API_KEY' in os.environ.keys() and os.getenv('OPENAI_API_KEY').strip() != '':
            api_key = os.getenv('OPENAI_API_KEY')
        elif api_key.strip() == '':
            yield "No API Key Provided"
            return
        
        self.__load_cached_shared(shared, api_key=api_key)
        
        
        version_context = []
        for repo in git_repos:
            
            _, repo_rel_name = os.path.split(repo.removesuffix('.git'))
            repo_dir = os.path.dirname(repo)
            _, repo_rel_dir =  os.path.split(repo_dir)
            
            for version in versions:
                if f"{repo_rel_dir}/{repo_rel_name}" in version:
                    _, true_ver = os.path.split(version)
                    version_context += self.version_specific_documents[repo][true_ver](inputs, 
                                                    k=max(1, CONTEXT_SIZE.GIT_DOCUMENTS//len(versions)), 
                                                    fetch_k=max(1, CONTEXT_SIZE.GIT_DIVERSE_K//len(versions)))
         
        shared_context = []
        for share in shared:
            shared_context += self.shared_documents[share].querry_documents(f"{'' if (versions==None or len(versions) == 0)  else versions}\n{inputs}", 
                                                                            k=max(1, CONTEXT_SIZE.SHARED_DOCUMENTS//len(shared)), 
                                                                            fetch_k=max(1, CONTEXT_SIZE.SHARED_DIVERSE_K//len(shared)))      
        messages = [
            {
                'role' : 'system',
                'content' : system_prompt
            },
            {
                "role": "user",
                "content":  PROMPTS.INPUT_PROMPT.format(version=versions, 
                                                            version_context=version_context, 
                                                            shared_context=shared_context, 
                                                            inputs=inputs)
            },
        ] 
        open_api = OpenAI(api_key=api_key)
        completion = open_api.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=float(temperature) if (temperature != None and temperature > 0)  else 0.2,
        )
        
        partial_message = ""
        for chunk in completion:
            partial_message += chunk.choices[0].delta.content if chunk.choices[0].delta.content != None else ''
            yield partial_message

    