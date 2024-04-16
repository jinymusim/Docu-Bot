# RAG on Git repositories
This project is a solution to utilize Retrieval Augmented Generation on Git Repositories.  

## Requirements

Python >= 3.10

### Windows

Windows requires additional Visual C++ 2015 build tools to be installed.  
They are available on Microsoft website [Build Tools](https://visualstudio.microsoft.com/downloads/)

## Selected models

Following models were selected as Default models for RAG.

- Embedding Model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Language Model: [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- Large Language Model (Mixtral): [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

To change them, rewrite the default values in ***MODEL_TYPES.py***

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
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install  -r requirements.txt
```

To run the solution, run python on script gradio_app_git_update.py

```
python gradio_app_git_update.py
```
This will produce a link to a frontend UI where user can input questions.


### GPU acceleration

If the utilized GPU is of newer desing (A40, A100, H100), flash attention is good acceleration for better inference.  
To activate, install following package.
```
pip install  flash-attn --no-build-isolation
```

### Run With Large Language Model

***IMPORTANT*** The model must be downloaded and loaded to ***RAM*** at original size of ***100GB***!  
Make sure you have enough disk space and RAM.

If the GPU contains enough onboard memory (>= 30GB), user can utilize Mixtral-8x7B model.  

To start the app with Mixtral use the ***--use-mixtral=True*** argument

```
python gradio_app_git_update.py --use-mixtral=True
```


