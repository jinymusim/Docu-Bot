#!/usr/bin/env python3
import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


import argparse
import torch
import gradio as gr
from retrival_augment_git import RetrivalAugment
from contextlib import redirect_stdout


def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t", 'True']:
        return True
    elif value in ["false", "no", "n", "0", "f", 'False']:
        return False

    return False


parser = argparse.ArgumentParser()

parser.add_argument("--use-mixtral", default=False, type=parse_boolean, help='Use Mixtral model for generation')


def main(args):
    
    retrival_class =RetrivalAugment(args=args)
    retrival_class._add_following_repo_branches('https://github.com/dCache/dcache.git', ['9.2'])
    
    demo = gr.Blocks(title='Document Bot', theme=gr.themes.Default(primary_hue=gr.themes.colors.orange, secondary_hue=gr.themes.colors.blue) )
    callback = gr.CSVLogger()
    
    def changed_repo(repo):
        choices = retrival_class._check_branch_cache(repo)
        return gr.update(choices=choices, value=choices[-1]) 
    
    def selected_repo(repo):
        choices = retrival_class._get_repo_branches(repo)
        if len(choices) == 0:
            return gr.update(visible=True), gr.update(visible=True, interactive=False)
        already_selected = retrival_class._check_branch_cache(repo)
        if len(already_selected) == 0:
            return gr.update(choices=choices, value=[], visible=True), gr.update(visible=True, interactive=False)
        return gr.update(choices=choices, value=already_selected, visible=True), gr.update(visible=True, interactive=True)
    
    def changed_branches(branches):
        if len(branches) == 0:
            return gr.update(visible=True, interactive=False)
        return gr.update(visible=True, interactive=True)    
        
    
    def update_repo():
        return gr.update(choices=retrival_class._get_cached_repos(), 
                            value=retrival_class._get_cached_repos()[0], interactive=True)
        
    def update_shared():
        return gr.update(choices=retrival_class._get_cached_shared(), 
                            value=[], interactive=True)
    
    with demo:
        gr.Markdown(
        """
        # Document Bot
        Problem analysis with Retrieval Augmented Generation.
        """)
        
        # =================================================================== 
        # Add Git Repo/Branch Section
        git_update_box = gr.Dropdown(choices=retrival_class._get_cached_repos(), allow_custom_value=True, visible=False, interactive=True, label='Git Repository Input')
        with gr.Row():
            with gr.Column():
                return_button = gr.Button('Return', variant='seconday', visible=False)
            with gr.Column():
                git_submit_button = gr.Button('Submit Repo', variant='primary', visible=False)
        branch_update_box = gr.Dropdown(visible=False, multiselect=True, interactive=True, label='Branches to Cache')
        branch_submit_button = gr.Button('Submit Branches', variant='primary', visible=False)
        
        ## Git Section UI controll
        
        git_submit_button.click(selected_repo, [git_update_box], [branch_update_box, branch_submit_button])
        
        
        
        # ==================================================================================
        # Add Files Section
        file_add_box = gr.File(file_count='single', file_types=['file'], interactive=True, visible=False, label='Zip Uploader')
        file_return_button = gr.Button('Return', variant='secondary', visible=False)
        
        ## Add Section UI controll
        
        
        
        # ===================================================================
        # Main section
        with gr.Row():
            
            with gr.Column():
                git_box = gr.Dropdown( choices=retrival_class._get_cached_repos(), label='Git Repos', 
                                      value=retrival_class._get_cached_repos()[-1], interactive=True, visible=False)
                version_box = gr.Dropdown(choices=retrival_class._check_branch_cache(retrival_class._get_cached_repos()[-1]), label='Branches', 
                                          value=retrival_class._check_branch_cache(retrival_class._get_cached_repos()[-1])[-1], interactive=True, multiselect=True, visible=False)
                shared_box = gr.Dropdown(choices=retrival_class._get_cached_shared(), interactive=True, visible=False, multiselect=True, label='Additional Files')
                
                question_box = gr.Textbox(label='Question about documents', lines=12, interactive=True, visible=False)
                submit_button = gr.Button('Submit Question', variant='primary', interactive=True, visible=False)
                add_repo = gr.Button('Add Git Repository/Branch', variant='secondary', interactive=True, visible=False)
                add_file = gr.Button('Add Additional Directory', variant='secondary', interactive=True, visible=False)
            with gr.Column():
                submited_question_box = gr.Textbox(label='Submited Question', lines=3, interactive=False, visible=False)
                answer_box = gr.Textbox(label='Answer', lines=9, interactive=False, visible=False) 
                documents = gr.Markdown(visible=False)
                
        
        ## Main section UI controll
        git_box.change(fn=changed_repo, inputs=[git_box], outputs=[version_box])
        
        branch_update_box.change(fn=changed_branches, inputs=[branch_update_box], outputs=[branch_submit_button])
        
        submit_button.click(retrival_class.__call__, inputs=[git_box,version_box, question_box, shared_box], outputs=[answer_box]).then(
            lambda *args: callback.flag(args), [version_box, git_box, submited_question_box, answer_box, shared_box], []
            )
        submit_button.click(retrival_class._get_relevant_docs, inputs=[git_box, version_box, question_box], outputs=[documents])
        submit_button.click(lambda x: x, [question_box], [submited_question_box]).then(lambda _: gr.update(value=''), [],[question_box])
        
        add_repo.click(lambda _:10 * [gr.update(visible=False)], [], [shared_box,add_file ,git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents]).then(
            fn=update_repo, inputs=[], outputs=[git_update_box]
        ).then(
            lambda _: 3 * [gr.update(visible=True)], [], [git_update_box, git_submit_button, return_button]
        )
        
        add_file.click(lambda _:10 * [gr.update(visible=False)], [], [shared_box, add_file ,git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents]).then(
            lambda _: 2 * [gr.update(visible=True)], [], [file_add_box, file_return_button]
        )
        
        # ==================================================================================
        # Returns
        return_button.click(lambda _: 5* [gr.update(visible=False)], [], [git_update_box, git_submit_button, return_button, branch_submit_button, branch_update_box]).then(
            fn=update_repo, inputs=[], outputs=[git_box]   
        ).then(
            lambda _:10* [gr.update(visible=True)], [], [shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]
        )
        
        file_return_button.click(lambda _: 2* [gr.update(visible=False)], [], [file_add_box, file_return_button]).then(
            fn=update_repo, inputs=[], outputs=[git_box]   
        ).then(
            lambda _ : 10*[gr.update(visible=True)], [], [shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]
        )
        
        # ==================================================================================
        # Submits with Returns
        
        branch_submit_button.click(lambda _: 3*[gr.update(interactive=False)], [], [git_submit_button, return_button, branch_submit_button]).then(
            retrival_class._add_following_repo_branches, [git_update_box, branch_update_box], [] 
        ).then(
            lambda _: 5* [gr.update(visible=False,interactive=True)], [], [git_update_box, git_submit_button, return_button, branch_submit_button, branch_update_box]
        ).then(
            fn=update_repo, inputs=[], outputs=[git_box]
        ).then(
            fn=changed_repo, inputs=[git_box], outputs=[version_box]
        ).then(
            lambda _: 10 *[gr.update(visible=True)], [], [shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]
        )
        
        file_add_box.upload(lambda _: 2*[gr.update(interactive=False)], [], [file_add_box, file_return_button]).then(
            retrival_class._add_following_zip, [file_add_box], []
        ).then(
            lambda _: 2* [gr.update(visible=False,interactive=True)], [], [file_add_box, file_return_button]
        ).then(
            fn=update_shared, inputs=[], outputs=[shared_box]
        ).then(
            lambda _: 10 *[gr.update(visible=True)], [], [shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]
        )
        
        # ==================================================================================
        # Setup Section  
        callback.setup([version_box, git_box, submited_question_box, answer_box, shared_box], "flagged_data_points_all")

        ## Setup section controll
        demo.load(fn=update_repo, inputs=[], outputs=[git_box]).then(
            fn=changed_repo, inputs=[git_box], outputs=[version_box]
        ).then(
            fn=update_shared, inputs=[], outputs=[shared_box]
        ).then(
            lambda _: 10 *[gr.update(visible=True)], [], [shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]
        )

        
    with redirect_stdout(sys.stderr):
        app, local, shared = demo.launch(share=False, debug=True)
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    with torch.no_grad():
        main(args)

