# RAG on Git repositories
This project is a solution to utilize Retrieval Augmented Generation on Git Repositories.  

## How to Run
To run clone this repository and install dependancies in following steps
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install  -r requirements.txt
pip install  flash-attn --no-build-isolation
```

To run the solution, run python on script gradio_app_git_update.py

```
python gradio_app_git_update.py
```
This will produce a link to a frontedn UI where user can input questions.
