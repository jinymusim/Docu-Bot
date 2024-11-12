#!/usr/bin/env python3
import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import MODEL_TYPES
import PROMPTS
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
parser.add_argument("--max-branch-boxes", default=10, type=int, help='Maximal number of branches to cache at once')


def main(args):
    
    retrival_class =RetrivalAugment(args=args)
    
    demo = gr.Blocks(title='Document Bot', theme=gr.themes.Default(primary_hue=gr.themes.colors.orange, secondary_hue=gr.themes.colors.blue) )
    callback = gr.CSVLogger()
    
    def get_good_branches(git_repos: list[str]):
        """Retrieves the latest branch for each given git repository.
        Args:
            git_repos (list[str]): A list of git repository URLs.
        Returns:
            list[str]: A list of the latest branches for each git repository.
        """
        branches = []
        for repo in git_repos:
            # Add the latest branch for repository.
            branches.append(retrival_class._check_branch_cache(repo)[-1])
        return branches
    
    def changed_repo(repos):
        choices = retrival_class._check_branch_cache(repos)
        preset = get_good_branches(repos)
        return gr.update(choices=choices, value=preset) 
    
    def changed_new_repo(repo:list[str],branches: list[str]):
        if len(repo) == 0:
            return gr.update(choices=[], value=[])
        choices = retrival_class._check_branch_cache_short(repo[0])
        good_branches = [branch for branch in branches if branch in choices]
        if len(good_branches) == 0:
            preset = get_good_branches(repo)
            return gr.update(choices=choices, value=preset)
        return gr.update(choices=choices, value=good_branches)
    
    def selected_repo(repo):
        choices = retrival_class._get_repo_branches(repo)
        if len(choices) == 0:
            return gr.update(visible=True, choices=[], value=[]), gr.update(visible=True, interactive=False), gr.update(visible=True, interactive=False)
        already_selected = retrival_class._check_branch_cache_short(repo)
        if len(already_selected) == 0:
            return gr.update(choices=choices, value=[], visible=True), gr.update(visible=True, interactive=False), gr.update(visible=True, interactive=False)
        return gr.update(choices=choices, value=already_selected, visible=True), gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True)
    
    def changed_branches(branches):
        if len(branches) == 0:
            return gr.update(visible=True, interactive=False), gr.update(visible=True, interactive=False)
        return gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True)  
       
    def display_branches_redirect(git_repo:str,braches:list[str]):
        redirects = retrival_class._get_branches_redirects(git_repo, braches)
        updates = []
        for branch, redirect in zip(braches,redirects):
            updates.append(gr.update(value=redirect, visible=True, label=f'{branch}', interactive=True))
        
        return [gr.update(visible=True)] + 2*[gr.update(visible=True, interactive=True)] + updates + (args.max_branch_boxes - len(updates)) * [gr.update(visible=False)]
           
    
    def update_repo():
        cashed_repos = retrival_class._get_cached_repos()
        if len(cashed_repos) == 0:
            return gr.update(choices=[], value=[], interactive=True)
        return gr.update(choices=cashed_repos, 
                        value=cashed_repos[0], interactive=True)
        
    def update_new_repo(repo:str):
        cashed_repos = retrival_class._get_cached_repos()
        if len(cashed_repos) == 0:
            return gr.update(choices=[], value=[], interactive=True)
        if repo in cashed_repos:
            return gr.update(choices=cashed_repos, value=repo, interactive=True)
        return gr.update(choices=cashed_repos, 
                        value=cashed_repos[0], interactive=True)
        
    def update_shared():
        return gr.update(choices=retrival_class._get_cached_shared(), 
                            value=[], interactive=True)
        
    
    
    with demo:
        
        
        # ===================================================================
        # Config Redirects Section
        branch_redirect_explain = gr.Markdown("""## Add Branch Redirects
                                              Input Redirects for Branches, If left empty, will redirect to repository.
                                              """, visible=False)
        branch_redirect_boxes = [gr.Textbox(label=f'TEXTBOX {i}', value='', visible=False, max_lines=1, interactive=True) for i in range(args.max_branch_boxes)]
        with gr.Row():
            with gr.Column():
                branch_submit_return_button = gr.Button('Return', variant='seconday', visible=False,  interactive=True)
            with gr.Column():
                branch_submit_button = gr.Button('Submit Branches', variant='primary', visible=False)
                
        config_redirects_boxes = [branch_redirect_explain, branch_submit_return_button, branch_submit_button] + branch_redirect_boxes
                
        ## Config Redirects Section UI controll
        
        
        # =================================================================== 
        # Add Git Repo/Branch Section
        
        add_repo_markdown = gr.Markdown("""## Add Git Repository/Branch
                                        Add a git repository and branch to the cache.
                                        """, visible=False)
        
        git_update_box = gr.Dropdown(choices=retrival_class._get_cached_repos(), allow_custom_value=True, visible=False, interactive=True, label='Git Repository Input')
        with gr.Row():
            with gr.Column():
                return_button = gr.Button('Return', variant='seconday', visible=False,  interactive=True)
            with gr.Column():
                git_submit_button = gr.Button('Submit Repo', variant='primary', visible=False)
        branch_update_box = gr.Dropdown(visible=False, multiselect=True, interactive=True, label='Branches to Cache',max_choices=args.max_branch_boxes)
        with gr.Row():
            with gr.Column():
                branch_quick_submit_button = gr.Button('Quick Submit', variant='primary', visible=False)
            with gr.Column():
                branch_redirect_update_button = gr.Button('Configure Branches', variant='secondary', visible=False)
                   
        
        git_add_select_initial_boxes = [git_update_box, return_button, git_submit_button, add_repo_markdown]
        git_add_select_boxes = git_add_select_initial_boxes + [branch_update_box, branch_redirect_update_button, branch_quick_submit_button]
        git_add_select_boxes_no_markdown = [git_update_box, return_button, git_submit_button] + [branch_update_box, branch_redirect_update_button, branch_quick_submit_button]
        
        ## Git Section UI controll
        
        git_update_box.change(lambda : 2 * [gr.update(visible=False)], [], [branch_update_box, branch_redirect_update_button])
        
        git_submit_button.click(selected_repo, [git_update_box], [branch_update_box, branch_redirect_update_button, branch_quick_submit_button])
        
        
        branch_redirect_update_button.click(lambda : len(git_add_select_boxes)*[gr.update(visible=False)], [], git_add_select_boxes).then(
            display_branches_redirect, [git_update_box, branch_update_box], [branch_redirect_explain, branch_submit_return_button, branch_submit_button]+ branch_redirect_boxes
        ) 
        
        
        # ==================================================================================
        # Add Files Section
        add_file_markdown = gr.Markdown("""## Add Files
                                        Add files to the shared directory.
                                        """, visible=False)
        
        file_add_box = gr.File(file_count='single', file_types=['file'], interactive=True, visible=False, label='Zip Uploader')
        file_return_button = gr.Button('Return', variant='secondary', visible=False)
        
        file_section_boxes = [file_add_box, file_return_button, add_file_markdown]
        file_section_boxes_no_markdown = [file_add_box, file_return_button]
        
        ## Add Section UI controll
        
        # ==================================================================================
        # Config Section
        model_page_markdown = gr.Markdown(f"""## Configurations
            Configure the OpenAI model, temperature and the system prompt.
            """, visible=False)
        
        change_temperature = gr.Slider(minimum=0.05, maximum=2.0, step=0.05, value=0.2, label='Temperature, Default = 0.2', interactive=True, visible=False)
        change_system_prompt = gr.Textbox(label='System Prompt',value=PROMPTS.SYSTEM_PROMPT, visible=False, interactive=True, lines=10, max_lines=10)
        open_ai_model = gr.Dropdown(choices=list(MODEL_TYPES.LLM_MODELS.keys()), value=list(MODEL_TYPES.LLM_MODELS.keys())[-1], label='Chosen Model', visible=False, interactive=True)
        open_ai_key = gr.Textbox(label='OpenAI API Key', visible=False, interactive=True, lines=1, max_lines=1)
        
        with gr.Row():
            with gr.Column():
                temperature_return_button = gr.Button('Return', variant='secondary', visible=False)
            with gr.Column():
                reset_button = gr.Button('Reset', variant='primary', visible=False)
                
                
        config_section_boxes = [model_page_markdown, temperature_return_button, reset_button, 
                                change_temperature, change_system_prompt, open_ai_model, open_ai_key]
                
        ## Config Section UI controll
        reset_button.click(lambda : [gr.update(value=0.2), gr.update(value=PROMPTS.SYSTEM_PROMPT)], [], [change_temperature, change_system_prompt])    
        
                
        # ==================================================================================
        # Main section
        
        main_page_markdown =  gr.Markdown(
        """# Document Bot
        Problem analysis with Retrieval Augmented Generation.""", visible=False)
        
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
        
        main_page_boxes = [main_page_markdown, git_box, version_box, shared_box, question_box, submit_button, add_repo, 
                           add_file, config_button, submited_question_box, answer_box, documents]    
        
        ## Main section UI controll
        git_box.change(fn=changed_repo, inputs=[git_box], outputs=[version_box])
        
        branch_update_box.change(fn=changed_branches, inputs=[branch_update_box], outputs=[branch_redirect_update_button, branch_quick_submit_button])
        
        submit_button.click(lambda x: x, [question_box], [submited_question_box]).then(
            lambda : gr.update(value=''), [],[question_box]
        ).then(
            retrival_class._get_relevant_docs, inputs=[git_box, version_box, submited_question_box, open_ai_key], outputs=[documents]
        ).then(
            retrival_class.__call__, inputs=[git_box,version_box, submited_question_box, shared_box, change_temperature, open_ai_key, open_ai_model, change_system_prompt], outputs=[answer_box]
        ).then(
            lambda *args: callback.flag(args), [version_box, git_box, submited_question_box, answer_box, shared_box, open_ai_model, change_temperature, change_system_prompt], []
        )
        
        add_repo.click(lambda :len(main_page_boxes) * [gr.update(visible=False)], [], main_page_boxes).then(
            fn=update_repo, inputs=[], outputs=[git_update_box]
        ).then(
            lambda : len(git_add_select_initial_boxes) * [gr.update(visible=True)], [], git_add_select_initial_boxes
        )
        
        add_file.click(lambda :len(main_page_boxes) * [gr.update(visible=False)], [], main_page_boxes).then(
            lambda : len(file_section_boxes) * [gr.update(visible=True)], [], file_section_boxes
        )
        
        config_button.click(lambda : len(main_page_boxes) * [gr.update(visible=False)], [], main_page_boxes).then(
            lambda  : len(config_section_boxes) * [gr.update(visible=True)], [], config_section_boxes
        )
        # ==================================================================================
        # Returns
        return_button.click(lambda : len(git_add_select_boxes)* [gr.update(visible=False)], [], git_add_select_boxes).then(
            fn=update_repo, inputs=[], outputs=[git_box]   
        ).then(
            lambda :len(main_page_boxes)* [gr.update(visible=True)], [], main_page_boxes
        )
        
        file_return_button.click(lambda : len(file_section_boxes)* [gr.update(visible=False)], [], file_section_boxes).then(
            fn=update_repo, inputs=[], outputs=[git_box]   
        ).then(
            lambda : len(main_page_boxes)*[gr.update(visible=True)], [], main_page_boxes
        )
        
        temperature_return_button.click(lambda : len(config_section_boxes)*[gr.update(visible=False)], [], config_section_boxes).then(
            lambda : len(main_page_boxes)*[gr.update(visible=True)], [], main_page_boxes
        )
        
        branch_submit_return_button.click(lambda : len(config_redirects_boxes)*[gr.update(visible=False)], [], config_redirects_boxes).then(
            lambda : len(git_add_select_boxes)*[gr.update(visible=True)], [], git_add_select_boxes
        )
        
        # ==================================================================================
        # Submits with Returns
        
        branch_submit_button.click(lambda : 2*[gr.update(interactive=False)], [], [branch_submit_return_button, branch_submit_button]).then(
            retrival_class._add_following_repo_branches, [git_update_box, branch_update_box, open_ai_key] + branch_redirect_boxes , [] 
        ).then(
            lambda : len(config_redirects_boxes)* [gr.update(visible=False)], [], config_redirects_boxes
        ).then(
            fn=update_new_repo, inputs=[git_update_box], outputs=[git_box]
        ).then(
            fn=changed_new_repo, inputs=[git_box, branch_update_box], outputs=[version_box]
        ).then(
            lambda : len(main_page_boxes) *[gr.update(visible=True)], [], main_page_boxes
        )
        
        file_add_box.upload(lambda : len(file_section_boxes_no_markdown)*[gr.update(interactive=False)], [], file_section_boxes_no_markdown).then(
            retrival_class._add_following_zip, [file_add_box], []
        ).then(
            lambda : len(file_section_boxes_no_markdown) *[gr.update(visible=False,interactive=True)], [], file_section_boxes_no_markdown
        ).then(
            lambda : gr.update(visible=False), [], add_file_markdown
        ).then(
            fn=update_shared, inputs=[], outputs=[shared_box]
        ).then(
            lambda : len(main_page_boxes) *[gr.update(visible=True)], [], main_page_boxes
        )
        
        branch_quick_submit_button.click(lambda : len(git_add_select_boxes_no_markdown)*[gr.update(interactive=False)], [], git_add_select_boxes_no_markdown).then(
            retrival_class._add_following_repo_branches, [git_update_box, branch_update_box,open_ai_key] + [] , [] 
        ).then(
            lambda : gr.update(visible=False), [], add_repo_markdown
        ).then(
            lambda : len(git_add_select_boxes_no_markdown)* [gr.update(visible=False, interactive=True)], [], git_add_select_boxes_no_markdown
        ).then(
            fn=update_new_repo, inputs=[git_update_box], outputs=[git_box]
        ).then(
            fn=changed_new_repo, inputs=[git_box, branch_update_box], outputs=[version_box]
        ).then(
            lambda : len(main_page_boxes) *[gr.update(visible=True)], [], main_page_boxes
        )
        
        # ==================================================================================
        # Setup Section  
        callback.setup([version_box, git_box, submited_question_box, answer_box, shared_box, open_ai_model, change_temperature, change_system_prompt], "flagged_data_points_all")

        ## Setup section controll
        demo.load(fn=update_repo, inputs=[], outputs=[git_box]).then(
            fn=changed_repo, inputs=[git_box], outputs=[version_box]
        ).then(
            fn=update_shared, inputs=[], outputs=[shared_box]
        ).then(
            lambda :len(main_page_boxes)* [gr.update(visible=True)], [], main_page_boxes
        )

        
    with redirect_stdout(sys.stderr):
        app, local, shared = demo.launch(share=False, server_name='0.0.0.0', server_port=7860)
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    with torch.no_grad():
        main(args)
