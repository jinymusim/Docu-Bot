from embeddings_dataset_langchain import EmbeddingsDataset
from langchain_openai import  OpenAIEmbeddings
from openai import OpenAI
from zipfile import ZipFile, BadZipFile
import PROMPTS
import MODEL_TYPES
import CONTEXT_SIZE
import os
import json
import git
import subprocess
import argparse
import shutil

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
        
                
        for filename in self.cached['cached_shared']:
            self.shared_documents[filename] = EmbeddingsDataset(
                os.path.join(self.cache_dir, filename.split('.')[0]), 
                cache_dir=os.path.join(self.cache_dir, f'{filename.split(".")[0]}-embed'), 
                transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key='None', base_url=MODEL_TYPES.DEFAULT_EMBED_LOC)
            )
      
    def _get_repo_branches(self, base_repo: str):
        """Retrieves the branches of a given git repository.
        Args:
            base_repo (str): The URL of the git repository.
        Returns:
            list: A list of branch names in the git repository.

        """
        # Check if proper git repo format
        if not base_repo.endswith('.git'):
            return []
        branches = []
        g = git.cmd.Git()
        # Try to lookup all of git branches
        try:
            # Utilize git command to list all branches
            for ref in g.ls_remote('--heads', base_repo).split('\n'):
                branches.append(ref.split('/')[-1])
            return branches
        # Execption if git repo is not found or not accessible
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
        """Get a list of all cached repositories.

        Returns:
            A list of repository names that are currently cached.
        """
        # Return all cached repos
        return list(self.cached['cached_repos'].keys())

    def _get_cached_shared(self):
        """Get the list of all cached secondary directories.

        Returns:
            A list of cached secondary directories.
        """
        # Return all cached shared directories
        return list(self.cached['cached_shared'])
      
    def _check_branch_cache_short(self, base_repo: str):
        # Return all cached branches to given repo
        if not base_repo.endswith('.git'):
            return []
        if base_repo in self.cached['cached_repos'].keys():
            return list(self.cached['cached_repos'][base_repo].keys())
        return []
    
    def _check_branch_cache(self, base_repos: list[str]):
        """Check the branch cache for the given base repositories.

        Args:
            base_repos (list[str]): A list of base repository paths.

        Returns:
            list[str]: A list of cached branches for the given repositories.
        """
        # Check if single repo or multiple repos
        if not isinstance(base_repos, list):
            # Split the repo path
            _, repo_rel_name = os.path.split(base_repos.removesuffix('.git'))
            # Get the directory of the repo
            repo_dir = os.path.dirname(base_repos)
            # Get the relative directory of the repo
            _, repo_rel_dir = os.path.split(repo_dir)
            # Check if repo is cached
            if base_repos in self.cached['cached_repos'].keys():
                return list(map(lambda x: f'{repo_rel_dir}/{repo_rel_name}/{x}', self.cached['cached_repos'][base_repos].keys())) 
            return []
        # Multiple repos
        cache_branches = []
        # Iterate over all repos
        for repo in base_repos:
            # Split the repo path
            _, repo_rel_name = os.path.split(repo.removesuffix('.git'))
            # Get the directory of the repo
            repo_dir = os.path.dirname(repo)
            # Get the relative directory of the repo
            _, repo_rel_dir = os.path.split(repo_dir)
            # Check if repo is cached
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
        # Split the repo path
        _, repo_rel_name = os.path.split(normalized_github_path)
        # Check if repo is cached
        if not base_repo in self.cached['cached_repos'].keys():
            self.cached['cached_repos'][base_repo] = {}
        if not base_repo in self.version_specific_documents.keys():
            # Add repo to loaded repos
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
                # Try to download the branch
                subprocess.run(f'curl -L -o {os.path.abspath(os.path.join(self.cache_dir, requested_branch + ".zip"))} {normalized_github_path}/zipball/{requested_branch}', shell=True)
                # Check if branch is downloaded
                try:
                    # Try to open the zip file
                    zf = ZipFile(os.path.join(self.cache_dir, f'{requested_branch}.zip'), 'r')
                # If zip file is corrupted
                except BadZipFile as e:
                    # Remove the zip file
                    os.remove(os.path.join(self.cache_dir, f'{requested_branch}.zip'))
                    # Try alternative download method (https://code.it4i.cz/sccs/docs.it4i.cz/-/archive/master/docs.it4i.cz-master.zip)
                    subprocess.run(f'curl -L -o {os.path.abspath(os.path.join(self.cache_dir, requested_branch + ".zip"))} {normalized_github_path}/-/archive/{requested_branch}/zipfile.zip', shell=True)
                # Try to open the zip file
                if os.path.exists(os.path.join(self.cache_dir , f'{requested_branch}.zip')):
                    try:
                        # Open the zip file
                        zf = ZipFile(os.path.join(self.cache_dir , f'{requested_branch}.zip'), 'r') 
                        # Create the directory for the branch
                        os.makedirs(os.path.join(self.cache_dir , repo_rel_name), exist_ok=True)
                        # Remove the branch directory if it exists
                        if os.path.exists(os.path.join(self.cache_dir , repo_rel_name, requested_branch)):
                            shutil.rmtree(os.path.join(self.cache_dir , repo_rel_name, requested_branch))
                        
                        filenames =  zf.namelist()

                        zf.extractall(os.path.join(self.cache_dir , repo_rel_name), members=filenames)
                        # Move files to the branch directory
                        shutil.move(os.path.join(self.cache_dir , repo_rel_name, zf.namelist()[0]),
                                  os.path.join(self.cache_dir , repo_rel_name, requested_branch))
                        # Remove the branch cache directory if it exists
                        if os.path.exists(os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed')):
                            shutil.rmtree(os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed'))
                        
                        self.version_specific_documents[base_repo][requested_branch] = EmbeddingsDataset(
                            os.path.join(self.cache_dir , repo_rel_name, requested_branch), 
                            cache_dir=os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed'), 
                            transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key=api_key, base_url=MODEL_TYPES.DEFAULT_EMBED_LOC)
                        )
                        
                        self.cached['cached_repos'][base_repo][requested_branch] = {'path': redirect.strip().rstrip('/')}
                        zf.close()
                        # Remove the zip file
                        os.remove(os.path.join(self.cache_dir , f'{requested_branch}.zip'))
                    # If zip file is corrupted
                    except Exception as e:
                        # Print the error
                        print(e)
                        # Remove the zip file
                        os.remove(os.path.join(self.cache_dir , f'{requested_branch}.zip'))
        # Remove Key entry if embedding failed
        if len(self.cached['cached_repos'][base_repo].keys()) == 0:
            self.cached['cached_repos'].pop(base_repo)
            # Remove the repo from the loaded repos
            self.version_specific_documents.pop(base_repo)
        # Save the cache list
        json.dump(self.cached, open(self.cache_repo_list, 'w+'), indent=6)
        
    def _add_following_file(self, file_info:str, api_key:str = None):
        #if api_key.strip() == '' and 'OPENAI_API_KEY' in os.environ.keys() and os.getenv('OPENAI_API_KEY').strip() != '':
        #    api_key = os.getenv('OPENAI_API_KEY')
        #elif api_key.strip() == '':
        #    return
        api_key = 'API_KEY'
        _, filename = os.path.split(file_info)
        # Check if file and not already cached
        if filename in self.cached['cached_shared']:
            # Return if already cached
            os.remove(file_info)
            return
        # Process the file
        else:
            # Move file to cache directory
            shutil.move(file_info, os.path.join(self.cache_dir , filename))
            try:
                # Remove cache directory if it exists
                if os.path.exists(os.path.join(self.cache_dir , filename.split('.')[0])):
                    shutil.rmtree(os.path.join(self.cache_dir , filename.split('.')[0]))
                # Create the cache directory for the file
                os.makedirs(os.path.join(self.cache_dir , filename.split('.')[0]), exist_ok=True)
                
                # Open the zip file
                if filename.endswith('.zip'):
                    zf = ZipFile(os.path.join(self.cache_dir , filename), 'r') \
                    
                    filenames = zf.namelist()
                    zf.extractall(os.path.join(self.cache_dir , filename.removesuffix('.zip')), members=filenames)
                else:
                    shutil.move(os.path.join(self.cache_dir , filename), os.path.join(self.cache_dir , filename.split('.')[0], filename))
                    filenames = [filename]
                    
                self.shared_documents[filename] = EmbeddingsDataset(
                    os.path.join(self.cache_dir , filename.split('.')[0]), 
                    cache_dir=os.path.join(self.cache_dir , f'{filename.split(".")[0]}-embed'), 
                    transformer_model=OpenAIEmbeddings(model=MODEL_TYPES.DEFAULT_EMBED_MODEL, api_key=api_key, base_url=MODEL_TYPES.DEFAULT_EMBED_LOC)
                )
                self.cached['cached_shared'].append(filename)
                # Close the zip file
                if filename.endswith('.zip'):
                    zf.close()
                    # Remove the file
                    os.remove(os.path.join(self.cache_dir , filename))
                
            # If file is corrupted  
            except Exception as e:
                # Print the error
                print(e)
                # Remove the file
                os.remove(os.path.join(self.cache_dir , filename))
        # Save the cache list     
        json.dump(self.cached, open(self.cache_repo_list, 'w+'), indent=6)
        
    def _get_relevant_docs(self, git_repos: list[str], versions: list[str], inputs: list[str]) -> str:
        
        result_string = "### Most Relevant Documents"
        # Iterate over all repositories
        for repo in git_repos:
            # Split the repo path
            _, repo_rel_name = os.path.split(repo.removesuffix('.git'))
            # Get the directory of the repo
            repo_dir = os.path.dirname(repo)
            # Get the relative directory of the repo
            _, repo_rel_dir =  os.path.split(repo_dir)
            # Iterate over all versions
            for version in versions:
                # Check if version is in the repo
                if f"{repo_rel_dir}/{repo_rel_name}" in version:
                    _, true_ver = os.path.split(version)    
                    relevant_docs = (self.version_specific_documents[repo][true_ver]).relevant_docs_filename(
                        inputs[-1]['content'], 
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
                           
            
    def __call__(self, git_repos: list[str] = None, versions= None, inputs: list[str] = None, shared=None, temperature: float= 0.7, api_key:str = None, model:str = 'gpt-3.5-turbo', system_prompt:str = PROMPTS.SYSTEM_PROMPT):    
        if len(git_repos) == 0 and len(shared) == 0:
            yield 'I was not given any documentation path to work from.'
            return
        
        if api_key.strip() == '' and 'OPENAI_API_KEY' in os.environ.keys() and os.getenv('OPENAI_API_KEY').strip() != '':
            api_key = os.getenv('OPENAI_API_KEY')
        elif api_key.strip() == '':
            yield "No API Key Provided"
            return
        
        copied_inputs = inputs.copy()
        
        # Get most relevant documents for given repos and versions
        for i, user_in in enumerate(copied_inputs):
            if user_in['role'] == 'user':
                copied_inputs[i]['docs'] = []
                
                for repo in git_repos:
            
                    _, repo_rel_name = os.path.split(repo.removesuffix('.git'))
                    repo_dir = os.path.dirname(repo)
                    _, repo_rel_dir =  os.path.split(repo_dir)

                    for version in versions:
                        if f"{repo_rel_dir}/{repo_rel_name}" in version:
                            _, true_ver = os.path.split(version)
                            # Get the relevant documents
                            copied_inputs[i]['docs'] += self.version_specific_documents[repo][true_ver](
                                user_in['content'], 
                                k=max(1, CONTEXT_SIZE.GIT_DOCUMENTS//len(versions)), 
                                fetch_k=max(1, CONTEXT_SIZE.GIT_DIVERSE_K//len(versions))
                            )
                            
        # Get most relevant documents for shared documents
        for i, user_in in enumerate(copied_inputs):
            if user_in['role'] == 'user':
                copied_inputs[i]['shared'] = []
                
                for share in shared:
                    # Get the relevant documents
                    copied_inputs[i]['shared'] += self.shared_documents[share].querry_documents_small(
                        f"{'' if (versions==None or len(versions) == 0) else versions}\n{user_in['content']}", 
                        k=max(1, CONTEXT_SIZE.SHARED_DOCUMENTS//len(shared)), 
                        fetch_k=max(1, CONTEXT_SIZE.SHARED_DIVERSE_K//len(shared))
                    )
        
        messages = [
            {
                'role' : 'system',
                'content' : system_prompt
            }]
        for one_input in copied_inputs:
            role = one_input['role']
            messages.append({
                "role": one_input['role'],
                "content": one_input['content'] if role == 'assistant' else PROMPTS.INPUT_PROMPT.format( 
                    version_context="\n".join(one_input['docs']), 
                    shared_context="\n".join(one_input['shared']), 
                    question=one_input['content']
                )
            })
        open_api = OpenAI(api_key=api_key, base_url=MODEL_TYPES.LLM_MODELS[model])
        completion = open_api.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            stream=True,
            temperature=float(temperature) if (temperature != None and temperature > 0)  else 0.7,
        )
        
        inputs.append({'role': 'assistant', 'content': ''})
        for chunk in completion:
            inputs[-1]['content'] += chunk.choices[0].delta.content if chunk.choices[0].delta.content != None else ''
            yield inputs
    