from embeddings_dataset_langchain import EmbeddingsDataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TextIteratorStreamer
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from threading import Thread
from zipfile import ZipFile, BadZipFile
import MODEL_TYPES
import PROMPTS
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
    def __init__(self, cache_repo_list=os.path.join(os.path.dirname(__file__), 'cached_repos.json'), cache_dir=os.path.join(os.path.dirname(__file__), 'py_cache'), args: argparse.Namespace = None) -> None:
        """
        Initializes the RetrivalAugment class.

        Parameters:
            cache_repo_list (str, optional): The path to the cache repository list. Defaults to 'cached_repos.json' in the same directory as this file.
            cache_dir (str, optional): The path to the cache directory. Defaults to 'py_cache' in the same directory as this file.
            args (argparse.Namespace, optional): Command-line arguments. Defaults to None.
        """
        # Embedding Model to be Used in Document and Querry Embeddings
        self.base_embedding_model = HuggingFaceEmbeddings(model_name=MODEL_TYPES.DEFAULT_EMBED_MODEL,
                                                          model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Store requested cache list location
        self.cache_repo_list = cache_repo_list
        # Create cache list if not present, otherwise load it
        if not os.path.exists(self.cache_repo_list):
            self.cached = {'cached_repos': {}, 'cached_shared': []}
            json.dump(self.cached, open(self.cache_repo_list, 'w+'), indent=6)
        else:
            self.cached = json.load(open(self.cache_repo_list, 'r'))

        model_type = MODEL_TYPES.DEFAULT_LM_MODEL if args == None or not args.use_mixtral else MODEL_TYPES.MIXTRAL_MODEL

        # Sample device to be used by models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load Language Model to be used in RAG
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        if torch.cuda.is_available():
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            # Load Attention based on GPU params
            self.model = AutoModelForCausalLM.from_pretrained(model_type, quantization_config=nf4_config,
                                                              device_map="auto",
                                                              attn_implementation="flash_attention_2" if supports_flash_attention() else "sdpa")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_type, device_map="auto").to(self.device)
        # Template variable for shared documents that could be uploaded
        self.shared_documents = {}
        # Load Cached Embeddings from cache list
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
                    os.path.join(self.cache_dir, repo_rel_name, branch),
                    cache_dir=os.path.join(self.cache_dir, f'{repo_rel_name}-{branch}-embed'),
                    transformer_model=self.base_embedding_model
                )

        for zip_name in self.cached['cached_shared']:
            self.shared_documents[zip_name] = EmbeddingsDataset(
                os.path.join(self.cache_dir, zip_name.removesuffix('.zip')),
                cache_dir=os.path.join(self.cache_dir, f'{zip_name.removesuffix(".zip")}-embed'),
                transformer_model=self.base_embedding_model
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
        """Check the branch cache for a given repository.

        Args:
            base_repo (str): The base repository to check.

        Returns:
            list: A list of cached branches for the given repository.
        """
        # Repo not in proper format
        if not base_repo.endswith('.git'):
            return []
        # Check if repo is cached
        if base_repo in self.cached['cached_repos'].keys():
            # Return all cached branches
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
                # Return all cached branches
                return list(map(lambda x: f'{repo_rel_dir}/{repo_rel_name}/{x}', self.cached['cached_repos'][base_repos].keys()))
            # Return empty list if repo not cached
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
                # Add all cached branches
                cache_branches += list(map(lambda x: f'{repo_rel_dir}/{repo_rel_name}/{x}', self.cached['cached_repos'][repo].keys()))
        # Return all cached branches
        return cache_branches

    def _add_following_repo_branches(self, base_repo: str, repo_branches: list[str], *args):
        """Add following repository branches to the cache and version-specific documents.

        Args:
            base_repo (str): The base repository URL.
            repo_branches (list[str]): List of repository branches.
            *args: Additional arguments. Should be the redirects for the branches.

        """
        # Check if repo is in proper format
        if not base_repo.endswith('.git'):
            return
        # Normalize the repo path
        normalized_github_path = base_repo.removesuffix('.git')
        # Split the repo path
        _, repo_rel_name = os.path.split(normalized_github_path)
        # Check if repo is cached
        if not base_repo in self.cached['cached_repos'].keys():
            # Add repo to cache
            self.cached['cached_repos'][base_repo] = {}
        # Check if repo loaded
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
        # Iterate over all branches
        for requested_branch, redirect in zip(repo_branches, requested_redirects):
            # Check if branch is cached
            if requested_branch in self.cached['cached_repos'][base_repo].keys():
                # Add redirect to cache
                if redirect_mindfully_inputted:
                    self.cached['cached_repos'][base_repo][requested_branch] = {'path': redirect.strip().rstrip('/')}
                # Skip if branch is already cached loaded
                if not requested_branch in self.version_specific_documents[base_repo].keys():
                    # Load branch embeddings
                    self.version_specific_documents[base_repo][requested_branch] = EmbeddingsDataset(
                        os.path.join(self.cache_dir, repo_rel_name, requested_branch),
                        cache_dir=os.path.join(self.cache_dir, f'{repo_rel_name}-{requested_branch}-embed'),
                        transformer_model=self.base_embedding_model
                    )
            # If branch is not cached
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
                        # Select only the text files for extraction
                        filenames = list(filter(lambda x: x.endswith('.txt') or x.endswith('.md') or x.endswith('.rst'), zf.namelist()) )
                        # Extract the files
                        zf.extractall(os.path.join(self.cache_dir , repo_rel_name), members=filenames)
                        # Move files to the branch directory
                        shutil.move(os.path.join(self.cache_dir , repo_rel_name, zf.namelist()[0]),
                                  os.path.join(self.cache_dir , repo_rel_name, requested_branch))
                        # Remove the branch cache directory if it exists
                        if os.path.exists(os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed')):
                            shutil.rmtree(os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed'))
                        # Load the branch embeddings
                        self.version_specific_documents[base_repo][requested_branch] = EmbeddingsDataset(os.path.join(self.cache_dir , repo_rel_name, requested_branch), 
                                                                                          cache_dir=os.path.join(self.cache_dir , f'{repo_rel_name}-{requested_branch}-embed'), 
                                                                                          transformer_model=self.base_embedding_model)
                        # Add branch to cache
                        self.cached['cached_repos'][base_repo][requested_branch] = {'path': redirect.strip().rstrip('/')}
                        # Close the zip file
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
            # Remove the repo from the cache
            self.cached['cached_repos'].pop(base_repo)
            # Remove the repo from the loaded repos
            self.version_specific_documents.pop(base_repo)
        # Save the cache list
        json.dump(self.cached, open(self.cache_repo_list, 'w+'), indent=6)
        
    def _add_following_zip(self, zip_info:str):
        """Adds a zip file to the cache directory and extracts relevant files from it.

        Args:
            zip_info (str): The path to the zip file.

        """
        # Check if zip file is in proper format
        _, zip_name = os.path.split(zip_info)
        # Check if zip file is a zip file and not already cached
        if not zip_name.endswith('.zip') or zip_name in self.cached['cached_shared']:
            # Return if not a zip file or already cached
            os.remove(zip_info)
            return
        # Process the zip file
        else:
            # Move zip file to cache directory
            shutil.move(zip_info, os.path.join(self.cache_dir , zip_name))
            try:
                # Open the zip file
                zf = ZipFile(os.path.join(self.cache_dir , zip_name), 'r') 
                # Remove zip cache directory if it exists (Bad Zip File Exception)
                if os.path.exists(os.path.join(self.cache_dir , zip_name.removesuffix('.zip'))):
                    shutil.rmtree(os.path.join(self.cache_dir , zip_name.removesuffix('.zip')))
                # Create the cache directory for the zip file
                os.makedirs(os.path.join(self.cache_dir , zip_name.removesuffix('.zip')), exist_ok=True)
                # Select only the text files for extraction
                filenames = list(filter(lambda x: x.endswith('.txt') or x.endswith('.md') or x.endswith('.rst'), zf.namelist()) )
                # Extract the files
                zf.extractall(os.path.join(self.cache_dir , zip_name.removesuffix('.zip')), members=filenames)
                # Remove the zip cache directory if it exists
                if os.path.exists(os.path.join(self.cache_dir , f'{zip_name.removesuffix(".zip")}-embed')):
                    shutil.rmtree(os.path.join(self.cache_dir , f'{zip_name.removesuffix(".zip")}-embed'))
                # Load the zip embeddings
                self.shared_documents[zip_name] = EmbeddingsDataset(os.path.join(self.cache_dir , zip_name.removesuffix('.zip')), 
                                                    cache_dir=os.path.join(self.cache_dir , f'{zip_name.removesuffix(".zip")}-embed'), 
                                                    transformer_model=self.base_embedding_model)
                # Add zip file to cache
                self.cached['cached_shared'].append(zip_name)
                # Close the zip file
                zf.close()
                # Remove the zip file
                os.remove(os.path.join(self.cache_dir , zip_name)) 
            # If zip file is corrupted  
            except Exception as e:
                # Print the error
                print(e)
                # Remove the zip file
                os.remove(os.path.join(self.cache_dir , zip_name))
        # Save the cache list     
        json.dump(self.cached, open(self.cache_repo_list, 'w+'), indent=6)
        
    def _get_relevant_docs(self, git_repos: list[str], versions: list[str], inputs):
        """Retrieves the most relevant documents from the specified git repositories and versions.

        Args:
            git_repos (list[str]): A list of git repository URLs.
            versions (list[str]): A list of versions or branches to search within the repositories.
            inputs: The input data used to determine the relevance of the documents.

        Returns:
            str: A string containing the most relevant documents.
        """
        # String to store the result
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
                    # Split the version
                    _, true_ver = os.path.split(version)   
                    # Get the relevant documents 
                    relevant_docs = (self.version_specific_documents[repo][true_ver]).relevant_docs_filename(inputs, 
                                        k=max(1, CONTEXT_SIZE.GIT_DOCUMENTS//len(versions)), fetch_k=max(1, CONTEXT_SIZE.GIT_DIVERSE_K//len(versions)))
                    # Iterate over all relevant documents
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
                           
            
    def __call__(self, git_repos: list[str] = None, versions= None, inputs = '', shared=None, temperature: float= 0.2, system_prompt =PROMPTS.SYSTEM_PROMPT ):
        """Generates a response based on the given inputs.

        Args:
            git_repos (list[str], optional): List of git repositories to search for version-specific documents. Defaults to None.
            versions (optional): List of versions to consider. Defaults to None.
            inputs (str, optional): User input or query. Defaults to ''.
            shared (optional): List of shared documents to consider. Defaults to None.
            temperature (float, optional): Controls the randomness of the generated response. Defaults to 0.2.
            system_prompt (str, optional): System prompt for the conversation. Defaults to PROMPTS.SYSTEM_PROMPT.

        Returns:
            str: Generated response based on the given inputs.
        """
        # Check if any documents are loaded
        if len(self.version_specific_documents.keys()) == 0 and len(self.shared_documents.keys()) == 0:
            return 'I was not given any documents from which to answer.'
        # Create Text steam for the model
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        version_context = []
        # Get most relevant documents for given repos and versions
        for repo in git_repos:
            
            _, repo_rel_name = os.path.split(repo.removesuffix('.git'))
            repo_dir = os.path.dirname(repo)
            _, repo_rel_dir =  os.path.split(repo_dir)
            
            for version in versions:
                if f"{repo_rel_dir}/{repo_rel_name}" in version:
                    _, true_ver = os.path.split(version)
                    # Get the relevant documents
                    version_context += self.version_specific_documents[repo][true_ver](inputs, 
                                            k=max(1, CONTEXT_SIZE.GIT_DOCUMENTS//len(versions)), fetch_k=max(1, CONTEXT_SIZE.GIT_DIVERSE_K//len(versions)))
         
        shared_context = []
        # Get most relevant documents for shared documents
        for share in shared:
            # Get the relevant documents
            shared_context += self.shared_documents[share].querry_documents(f"{'' if (versions==None or len(versions) == 0)  else versions}\n{inputs}", 
                                    k=max(1, CONTEXT_SIZE.SHARED_DOCUMENTS//len(shared)), fetch_k=max(1, CONTEXT_SIZE.SHARED_DIVERSE_K//len(shared)))
        # Create Message from template
        messages = [
            {
                "role": "user",
                "content": system_prompt + PROMPTS.INPUT_PROMPT.format(version=versions, 
                                                                        version_context=version_context, 
                                                                        shared_context=shared_context, 
                                                                        inputs=inputs)
            },

        ]
        
        # Apply chat template to the messages
        chatted = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to(self.device)
        # Create argument list for model generation
        generate_kwargs = dict(
            chatted,
            streamer=streamer,
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.99,
            top_k=500,
            temperature=float(temperature) if (temperature != None and temperature > 0)  else 0.2,
            num_beams=1,
        )
        # Start the model generation in a separate thread
        t = Thread(target=self.model.generate, kwargs=generate_kwargs) 
        t.start()
        # Collect partially generated messages for display
        partial_message = ""
        for new_token in streamer:
            partial_message += new_token
            yield partial_message

    
if __name__ == '__main__':
    augment = RetrivalAugment()
    augment._add_following_repo_branches('https://github.com/jinymusim/GPT-Czech-Poet.git', ['main'])
    print(augment._get_cached_repos())
    print(augment._check_branch_cache('https://github.com/jinymusim/GPT-Czech-Poet.git'))