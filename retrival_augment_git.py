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
    
    def __init__(
        self, 
        cache_repo_list = os.path.join(os.path.dirname(__file__), 'cached_repos.json'), 
        cache_dir= os.path.join(os.path.dirname(__file__), 'py_cache'), 
        args:argparse.Namespace = None
    ) -> None:
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
        self.__load_all_cached()
        
    
    def __load_all_cached(self):
        """Load all cached repositories and shared documents.
        """
        # Load each git repo
        for key in self.cached['cached_repos'].keys():
            # Embedding storage variable
            self.version_specific_documents[key] = {}
            normalized_github_path = key.removesuffix('.git')
            _, repo_rel_name = os.path.split(normalized_github_path)
            # Load all cached branches of the git repo
            for branch in self.cached['cached_repos'][key].keys():
                    self.version_specific_documents[key][branch] = EmbeddingsDataset(
                        os.path.join(self.cache_dir , repo_rel_name, branch), 
                        cache_dir=os.path.join(self.cache_dir , f'{repo_rel_name}-{branch}-embed'), 
                        transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key='None', base_url=MODEL_TYPES.DEFAULT_EMBED_LOC)
                    )
        
                
        for zip_name in self.cached['cached_shared']:
            self.shared_documents[zip_name] = EmbeddingsDataset(
                os.path.join(self.cache_dir, zip_name.removesuffix('.zip')), 
                cache_dir=os.path.join(self.cache_dir, f'{zip_name.removesuffix(".zip")}-embed'), 
                transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key='None', base_url=MODEL_TYPES.DEFAULT_EMBED_LOC)
            )
      
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
        
    def _get_branches_redirects(self, base_repo: str, branches: list[str]):
        """Retrieves the redirects for the specified branches in a given base repository.

        Args:
            base_repo (str): The name of the base repository.
            branches (list[str]): A list of branch names.

        Returns:
            list[str]: A list of redirects corresponding to the branches. If a branch doesn't have a redirect, an empty string is returned for that branch.
        """
        # Empty redirects when repo not cached
        if not base_repo in self.cached['cached_repos'].keys():
            return [''] * len(branches)
        redirects = []
        # Retrive all redirects for branches
        for branch in branches:
            # Check if branch is cached
            if branch in self.cached['cached_repos'][base_repo].keys():
                redirects.append(self.cached['cached_repos'][base_repo][branch]['path'])
            # If not cached, return empty string
            else:
                redirects.append('')
        return redirects
         
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
            return list(self.cached['cached_repos'][base_repo].keys())
        return []
    
    def _check_branch_cache(self, base_repos: list[str]):
        # Return all cached branches to given repos
        if not isinstance(base_repos, list):
            _, repo_rel_name = os.path.split(base_repos.removesuffix('.git'))
            repo_dir = os.path.dirname(base_repos)
            _, repo_rel_dir =  os.path.split(repo_dir)
            if base_repos in self.cached['cached_repos'].keys():
                return list(map(lambda x: f'{repo_rel_dir}/{repo_rel_name}/{x}', self.cached['cached_repos'][base_repos].keys())) 
            return []
        cache_branches = []
        for repo in base_repos:

            _, repo_rel_name = os.path.split(repo.removesuffix('.git'))
            repo_dir = os.path.dirname(repo)
            _, repo_rel_dir =  os.path.split(repo_dir)
            
            if repo in self.cached['cached_repos'].keys():
                cache_branches += list(map(lambda x: f'{repo_rel_dir}/{repo_rel_name}/{x}', self.cached['cached_repos'][repo].keys())) 
        return cache_branches
            
    def _add_following_repo_branches(self, base_repo:str, repo_branches: list[str], api_key:str = None, *args):
        if not base_repo.endswith('.git'):
            return
        #if api_key.strip() == '' and 'OPENAI_API_KEY' in os.environ.keys() and os.getenv('OPENAI_API_KEY').strip() != '':
        #    api_key = os.getenv('OPENAI_API_KEY')
        #elif api_key.strip() == '':
        #    return
        api_key = 'API_KEY'
        normalized_github_path = base_repo.removesuffix('.git')
        _ ,repo_rel_name = os.path.split(normalized_github_path)
        if not base_repo in self.cached['cached_repos'].keys():
            self.cached['cached_repos'][base_repo] = {}
        if not base_repo in self.version_specific_documents.keys():
            self.version_specific_documents[base_repo] = {}
        # Check if no redirects are given (Quick submission)
        if len(args) == 0:
            redirect_mindfully_inputted = False
            requested_redirects = [''] * len(repo_branches)
        # Collect all redirects
        else:
            requested_redirects = list(args)
            redirect_mindfully_inputted = True
        for requested_branch, redirect in zip(repo_branches, requested_redirects):
            if requested_branch in self.cached['cached_repos'][base_repo].keys():
                if redirect_mindfully_inputted:
                    self.cached['cached_repos'][base_repo][requested_branch] = {'path': redirect.strip().rstrip('/')}
                if not requested_branch in self.version_specific_documents[base_repo].keys():
                    self.version_specific_documents[base_repo][requested_branch] = EmbeddingsDataset(
                        os.path.join(self.cache_dir , repo_rel_name, requested_branch), 
                        cache_dir=os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed'), 
                        transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key=api_key, base_url=MODEL_TYPES.DEFAULT_EMBED_LOC)
                    )
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
                        
                        filenames =  zf.namelist()

                        zf.extractall(os.path.join(self.cache_dir , repo_rel_name), members=filenames)
                        shutil.move(os.path.join(self.cache_dir , repo_rel_name, zf.namelist()[0]),
                                  os.path.join(self.cache_dir , repo_rel_name, requested_branch))
                        if os.path.exists(os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed')):
                            shutil.rmtree(os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed'))
                        
                        self.version_specific_documents[base_repo][requested_branch] = EmbeddingsDataset(
                            os.path.join(self.cache_dir , repo_rel_name, requested_branch), 
                            cache_dir=os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed'), 
                            transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key=api_key, base_url=MODEL_TYPES.DEFAULT_EMBED_LOC)
                        )
                        
                        self.cached['cached_repos'][base_repo][requested_branch] = {'path': redirect.strip().rstrip('/')}
                        zf.close()
                        os.remove(os.path.join(self.cache_dir , f'{requested_branch}.zip'))
                    except Exception as e:
                        print(e)
                        os.remove(os.path.join(self.cache_dir , f'{requested_branch}.zip'))
        # Remove Key entry if embedding failed
        if len(self.cached['cached_repos'][base_repo].keys()) == 0:
            self.cached['cached_repos'].pop(base_repo)
            self.version_specific_documents.pop(base_repo)
        
        json.dump(self.cached, open(self.cache_repo_list, 'w+'), indent=6)
        
    def _add_following_zip(self, zip_info:str, api_key:str = None):
        #if api_key.strip() == '' and 'OPENAI_API_KEY' in os.environ.keys() and os.getenv('OPENAI_API_KEY').strip() != '':
        #    api_key = os.getenv('OPENAI_API_KEY')
        #elif api_key.strip() == '':
        #    return
        api_key = 'API_KEY'
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
                    
                filenames = zf.namelist()
                zf.extractall(os.path.join(self.cache_dir , zip_name.removesuffix('.zip')), members=filenames)
                
                if os.path.exists(os.path.join(self.cache_dir , f'{zip_name.removesuffix(".zip")}-embed')):
                    shutil.rmtree(os.path.join(self.cache_dir , f'{zip_name.removesuffix(".zip")}-embed'))
                    
                self.shared_documents[zip_name] = EmbeddingsDataset(
                    os.path.join(self.cache_dir , zip_name.removesuffix('.zip')), 
                    cache_dir=os.path.join(self.cache_dir , f'{zip_name.removesuffix(".zip")}-embed'), 
                    transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key=api_key, base_url=MODEL_TYPES.DEFAULT_EMBED_LOC)
                )
                self.cached['cached_shared'].append(zip_name)
                zf.close()
                os.remove(os.path.join(self.cache_dir , zip_name))   
            except Exception as e:
                print(e)
                os.remove(os.path.join(self.cache_dir , zip_name))
                
        json.dump(self.cached, open(self.cache_repo_list, 'w+'), indent=6)
        
    def _get_relevant_docs(self, git_repos: list[str], versions: list[str], inputs) -> str:
        
        result_string = "### Most Relevant Documents"
        for repo in git_repos:
            
            _, repo_rel_name = os.path.split(repo.removesuffix('.git'))
            repo_dir = os.path.dirname(repo)
            _, repo_rel_dir =  os.path.split(repo_dir)
            
            for version in versions:
                if f"{repo_rel_dir}/{repo_rel_name}" in version:
                    _, true_ver = os.path.split(version)    
                    relevant_docs = (self.version_specific_documents[repo][true_ver]).relevant_docs_filename(
                        inputs, 
                        k=max(1, CONTEXT_SIZE.GIT_DOCUMENTS//len(versions)), 
                        fetch_k=max(1, CONTEXT_SIZE.GIT_DIVERSE_K//len(versions))
                    )
                    full_paths = []
                    for path in relevant_docs:
                        # Get the relative file path
                        rel_file_path:str = path.split(true_ver)[-1].replace(os.sep, '/')
                        rel_file_path_norm = rel_file_path.removeprefix('/')
                        # Check if the file is in a subdirectory
                        repo_name = repo.removesuffix('.git')
                        # Check if the branch has a redirect
                        if self.cached['cached_repos'][repo][true_ver]['path'].strip() != '':
                            # Get the redirect name
                            redirect_name = self.cached['cached_repos'][repo][true_ver]['path']
                            # Add the full path to the list
                            full_paths.append(f'[{rel_file_path_norm}]({redirect_name}{rel_file_path})' )   
                        # If no redirect
                        else:
                            # Add Git path to the list
                            full_paths.append(f'[{rel_file_path_norm}]({repo_name}/blob/{true_ver}{rel_file_path})' )
                    # Sort the paths
                    full_paths = sorted(full_paths)
                    # Add the paths to the result string
                    result_string += f'\n #### {"Repo" if len(git_repos) > 1 else "Branch"} {version} \n' + '  \n'.join(full_paths)
            
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
        open_api = OpenAI(api_key=api_key, base_url=MODEL_TYPES.LLM_MODELS[model])
        completion = open_api.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            stream=True,
            temperature=float(temperature) if (temperature != None and temperature > 0)  else 0.2,
        )
        
        partial_message = ""
        for chunk in completion:
            partial_message += chunk.choices[0].delta.content if chunk.choices[0].delta.content != None else ''
            yield partial_message

    