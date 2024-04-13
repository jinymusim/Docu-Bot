# RAG on Git repositories
This project is a solution to utilize Retrieval Augmented Generation on Git Repositories.  

## Requirements

Python >= 3.10

### Windows

Windows requires additional Visual C++ 2015 build tools to be installed.  
They are available on Microsoft website [Build Tools](https://visualstudio.microsoft.com/downloads/)

## How to Run
To run clone this repository and install dependancies in following steps
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install  -r requirements.txt
```

To run the solution, run python on script gradio_app_git_update.py

```
python gradio_app_git_update.py
```
This will produce a link to a frontedn UI where user can input questions.

### GPU acceleration

If the utilized GPU is of newer desing (A40, A100, H100), flash attention is good acceleration for better inference.  
To activate, install following package.
```
pip install  flash-attn --no-build-isolation
```
