# RAG on Git repositories
This project is a solution to utilize Retrieval Augmented Generation on Git Repositories.  

## Requirements

Python >= 3.10

### Windows

Windows requires additional Visual C++ 2015 build tools to be installed.  
They are available on Microsoft website [Build Tools](https://visualstudio.microsoft.com/downloads/)

## Selected models

Following model was selected as Default models for Embedding.

- Embedding Model: [all-mpnet-base-v1](https://huggingface.co/sentence-transformers/all-mpnet-base-v1)
- [serve-model](https://github.com/jinymusim/serve-model) project is needed for standalone model serving.

To change it, rewrite the default value in ***MODEL_TYPES.py***. For example to `embedding-3-small`

The Default LLM model was seleted as:

- LLM Model: [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [serve-model](https://github.com/jinymusim/serve-model)

But this is changable in the ***Config*** section, allowing user to choose OpenAI LLM of their choice.


## Number of Retrieved Files

The approximate number of files retrived durring Document Retrival is defined in ***CONTEXT_SIZE.py***.

- ***GIT_DOCUMENTS*** defines approximate number of documents that will be taken from all branches of git repo together
- ***GIT_DIVERSE_K*** defines the divese field from with most disimilar documents will be taken
- ***SHARED_DOCUMENTS*** defines approximate number of documents that will be taken from all requested zips
- ***SHARED_DIVERSE_K*** is same as ***GIT_DIVERSE_K***

These numbers can be altered. The selected values were observed to be good midleground in information correctness and size.

## How to Run
To run, clone this repository and install dependancies in following steps
```
pip install  -r requirements.txt
```
To run the solution, run python on script gradio_app_git_update.py

```
python gradio_app_git_update.py
```
This will produce a link to a frontend UI.  
To utilize OpenAI models it is required to provide API key in the ***Config*** section.

### Provide Companny API key

If maneged behined authetication system, API key can be set as a default option.  
The user inputed API key will be used if inputed.  

To set company wide key, set eviromental variable OPENAI_API_KEY that will be utilized by the application.
```Windows
# Windows
$env:OPENAI_API_KEY=Company API key
```
```Linux
# Linux
export OPENAI_API_KEY=Company API key
```

### Custom Models

If custom models are to be utilized, follow the [serve-model](https://github.com/jinymusim/serve-model) project and run custom Huggingface models.
Custom models can be inserted to `MODEL_TYPES.py` with proper `base_url` where the model is served from.
