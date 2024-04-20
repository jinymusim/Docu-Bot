#!/usr/bin/env python3
import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import MODEL_TYPES

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
    
    demo = gr.Blocks(title='Document Bot', theme=gr.themes.Default(primary_hue=gr.themes.colors.orange, secondary_hue=gr.themes.colors.blue) )
    callback = gr.CSVLogger()
    
    def get_good_branches(git_repos: list[str]):
        branches = []
        for repo in git_repos:
            branches.append(retrival_class._check_branch_cache(repo)[-1])
        return branches
    
    def changed_repo(repos):
        choices = retrival_class._check_branch_cache(repos)
        preset = get_good_branches(repos)
        return gr.update(choices=choices, value=preset) 
    
    def selected_repo(repo):
        choices = retrival_class._get_repo_branches(repo)
        if len(choices) == 0:
            return gr.update(visible=True), gr.update(visible=True, interactive=False)
        already_selected = retrival_class._check_branch_cache_short(repo)
        if len(already_selected) == 0:
            return gr.update(choices=choices, value=[], visible=True), gr.update(visible=True, interactive=False)
        return gr.update(choices=choices, value=already_selected, visible=True), gr.update(visible=True, interactive=True)
    
    def changed_branches(branches):
        if len(branches) == 0:
            return gr.update(visible=True, interactive=False)
        return gr.update(visible=True, interactive=True)    
        
    
    def update_repo():
        cashed_repos = retrival_class._get_cached_repos()
        if len(cashed_repos) == 0:
            return gr.update(choices=[], value=[], interactive=True)
        return gr.update(choices=cashed_repos, 
                        value=cashed_repos[0], interactive=True)
        
    def update_shared():
        return gr.update(choices=retrival_class._get_cached_shared(), 
                            value=[], interactive=True)
        
    
    
    with demo:
        gr.Markdown(
        f"""
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
        
        # ==================================================================================
        # Config Section
        change_temperature = gr.Slider(minimum=0.05, maximum=3, step=0.05, value=0.2, label='Temperature, Default = 0.2', interactive=True, visible=False)
        with gr.Row():
            with gr.Column():
                temperature_return_button = gr.Button('Return', variant='secondary', visible=False)
            with gr.Column():
                temperature_reset_button = gr.Button('Reset', variant='primary', visible=False)
        open_ai_model = gr.Dropdown(choices=MODEL_TYPES.LLM_MODELS, value=MODEL_TYPES.LLM_MODELS[0], label='OpenAI Model', visible=False, interactive=True)
        open_ai_key = gr.Textbox(label='OpenAI API Key', visible=False, interactive=True, lines=1, max_lines=1)
                
        ## Config Section UI controll
        temperature_reset_button.click(lambda : gr.update(value=0.2), [], [change_temperature])    
        
                
        # ==================================================================================
        # Main section
        with gr.Row():
            
            with gr.Column():
                git_box = gr.Dropdown( choices=retrival_class._get_cached_repos(), label='Git Repos', 
                                      value=[] if len(retrival_class._get_cached_repos()) == 0 else [retrival_class._get_cached_repos()[-1]], 
                                      interactive=True, visible=False, multiselect=True)
                version_box = gr.Dropdown(choices=[] if len(retrival_class._get_cached_repos()) == 0 else retrival_class._check_branch_cache([retrival_class._get_cached_repos()[-1]]), 
                                          label='Branches', 
                                          value=[] if len(retrival_class._get_cached_repos()) == 0 else get_good_branches([retrival_class._get_cached_repos()[-1]]), 
                                          interactive=True, multiselect=True, visible=False)
                shared_box = gr.Dropdown(choices=retrival_class._get_cached_shared(), interactive=True, visible=False, multiselect=True, label='Additional Files')
                
                question_box = gr.Textbox(label='Question about documents', lines=12, interactive=True, visible=False)
                submit_button = gr.Button('Submit Question', variant='primary', interactive=True, visible=False)
                with gr.Row():
                    with gr.Column():
                        add_repo = gr.Button('Add Git Repository/Branch', variant='secondary', interactive=True, visible=False)
                    with gr.Column():
                        add_file = gr.Button('Add Additional Directory', variant='secondary', interactive=True, visible=False)
                config_button = gr.Button('Config', variant='secondary', interactive=True, visible=False)
            with gr.Column():
                submited_question_box = gr.Textbox(label='Submited Question', lines=3, interactive=False, visible=False)
                answer_box = gr.Textbox(label='Answer', lines=9, interactive=False, visible=False) 
                documents = gr.Markdown(visible=False)
                
        
        ## Main section UI controll
        git_box.change(fn=changed_repo, inputs=[git_box], outputs=[version_box])
        
        branch_update_box.change(fn=changed_branches, inputs=[branch_update_box], outputs=[branch_submit_button])
        
        submit_button.click(lambda x: x, [question_box], [submited_question_box]).then(
            lambda : gr.update(value=''), [],[question_box]
        ).then(
            retrival_class._get_relevant_docs, inputs=[git_box, version_box, submited_question_box, open_ai_key], outputs=[documents]
        ).then(
            retrival_class.__call__, inputs=[git_box,version_box, submited_question_box, shared_box, change_temperature, open_ai_key, open_ai_model], outputs=[answer_box]
        ).then(
            lambda *args: callback.flag(args), [version_box, git_box, submited_question_box, answer_box, shared_box, open_ai_model, change_temperature], []
        )
        
        add_repo.click(lambda :11 * [gr.update(visible=False)], [], [config_button,shared_box,add_file ,git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents]).then(
            fn=update_repo, inputs=None, outputs=[git_update_box]
        ).then(
            lambda : 3 * [gr.update(visible=True)], inputs=[], outputs=[git_update_box, git_submit_button, return_button]
        )
        
        add_file.click(lambda :11 * [gr.update(visible=False)], [], [config_button, shared_box, add_file ,git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents]).then(
            lambda : 2 * [gr.update(visible=True)], [], [file_add_box, file_return_button]
        )
        
        config_button.click(lambda : 11 * [gr.update(visible=False)], [], [config_button, shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]).then(
            lambda  : 5 * [gr.update(visible=True)], [], [open_ai_model, open_ai_key, change_temperature, temperature_return_button, temperature_reset_button]
        )
        # ==================================================================================
        # Returns
        return_button.click(lambda : 5* [gr.update(visible=False)], [], [git_update_box, git_submit_button, return_button, branch_submit_button, branch_update_box]).then(
            fn=update_repo, inputs=[], outputs=[git_box]   
        ).then(
            lambda :11* [gr.update(visible=True)], [], [config_button, shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]
        )
        
        file_return_button.click(lambda : 2* [gr.update(visible=False)], [], [file_add_box, file_return_button]).then(
            fn=update_repo, inputs=[], outputs=[git_box]   
        ).then(
            lambda  : 11*[gr.update(visible=True)], [], [config_button, shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]
        )
        
        temperature_return_button.click(lambda : 5*[gr.update(visible=False)], [], [open_ai_model, open_ai_key, change_temperature, temperature_return_button, temperature_reset_button]).then(
            lambda : 11*[gr.update(visible=True)], [], [config_button, shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]
        )
        
        # ==================================================================================
        # Submits with Returns
        
        branch_submit_button.click(lambda : 3*[gr.update(interactive=False)], [], [git_submit_button, return_button, branch_submit_button]).then(
            retrival_class._add_following_repo_branches, [git_update_box, branch_update_box, open_ai_key], [] 
        ).then(
            lambda : 5* [gr.update(visible=False,interactive=True)], [], [git_update_box, git_submit_button, return_button, branch_submit_button, branch_update_box]
        ).then(
            fn=update_repo, inputs=[], outputs=[git_box]
        ).then(
            fn=changed_repo, inputs=[git_box], outputs=[version_box]
        ).then(
            lambda : 11 *[gr.update(visible=True)], [], [config_button, shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]
        )
        
        file_add_box.upload(lambda : 2*[gr.update(interactive=False)], [], [file_add_box, file_return_button]).then(
            retrival_class._add_following_zip, [file_add_box, open_ai_key], []
        ).then(
            lambda : 2* [gr.update(visible=False,interactive=True)], [], [file_add_box, file_return_button]
        ).then(
            fn=update_shared, inputs=[], outputs=[shared_box]
        ).then(
            lambda : 11 *[gr.update(visible=True)], [], [config_button, shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]
        )
        
        # ==================================================================================
        # Setup Section  
        callback.setup([version_box, git_box, submited_question_box, answer_box, shared_box, open_ai_model, change_temperature], "flagged_data_points_all")

        ## Setup section controll
        demo.load(fn=update_repo, inputs=[], outputs=[git_box]).then(
            fn=changed_repo, inputs=[git_box], outputs=[version_box]
        ).then(
            fn=update_shared, inputs=[], outputs=[shared_box]
        ).then(
            lambda : 11 *[gr.update(visible=True)], [], [config_button, shared_box, git_box, version_box, question_box, submit_button, add_repo, submited_question_box, answer_box, documents, add_file]
        )

        
    with redirect_stdout(sys.stderr):
        app, local, shared = demo.launch(share=False, server_name='0.0.0.0', server_port=7860)
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    with torch.no_grad():
        main(args)

