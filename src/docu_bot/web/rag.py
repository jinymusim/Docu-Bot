#!/usr/bin/env python3
import sys
import os

import argparse
import gradio as gr
from contextlib import redirect_stdout
from docu_bot.document_loaders.utils import (
    LoadedRepositoriesAndFiles,
    get_available_branches,
)
from docu_bot.document_loaders.git_document_loader import GitDocumentLoader
from docu_bot.document_loaders.zip_document_loader import ZipDocumentLoader
from docu_bot.stores.utils import (
    LoadedVectorStores,
    create_vector_store_from_document_loader,
)
from docu_bot.stores.docstore import DocumentStore
from docu_bot.constants import PROMPTS, MODEL_TYPES
from docu_bot.chat.answer_openai import (
    prepare_retriever,
    get_documents,
    stream_rag,
)


def main(args):

    repositories_cache = LoadedRepositoriesAndFiles()
    vectorstores_cache = LoadedVectorStores()
    document_store = DocumentStore()

    demo = gr.Blocks(
        title="Document Bot",
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.orange, secondary_hue=gr.themes.colors.blue
        ),
    )
    callback = gr.CSVLogger()

    # ==================================================================================
    # UI Functions

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
            branches.append(repositories_cache.get_cached_repo_branches(repo)[-1])
        return branches

    def changed_repo(repos):
        choices = []
        for repo in repos:
            choices += repositories_cache.get_cached_repo_branches(repo)
        preset = get_good_branches(repos)
        return gr.update(choices=choices, value=preset)

    def changed_new_repo(repo: list[str], branches: list[str]):
        if len(repo) == 0:
            return gr.update(choices=[], value=[])
        choices = repositories_cache.get_cached_repo_branches(repo[0])
        good_branches = [branch for branch in branches if branch in choices]
        if len(good_branches) == 0:
            preset = get_good_branches(repo)
            return gr.update(choices=choices, value=preset)
        return gr.update(choices=choices, value=good_branches)

    def selected_repo(repo):
        choices = get_available_branches(repo)
        if len(choices) == 0:
            return (
                gr.update(visible=True, choices=[], value=[]),
                gr.update(visible=True, interactive=False),
            )
        already_selected = repositories_cache.get_cached_repo_branches_short(repo)
        if len(already_selected) == 0:
            return (
                gr.update(choices=choices, value=[], visible=True),
                gr.update(visible=True, interactive=False),
            )
        return (
            gr.update(choices=choices, value=already_selected, visible=True),
            gr.update(visible=True, interactive=True),
        )

    def changed_branches(branches):
        if len(branches) == 0:
            return gr.update(visible=True, interactive=False)
        return gr.update(visible=True, interactive=True)

    def update_repo():
        cached_repos = repositories_cache.get_cached_repositories()
        if len(cached_repos) == 0:
            return gr.update(choices=[], value=[], interactive=True)
        return gr.update(choices=cached_repos, value=cached_repos[0], interactive=True)

    def update_new_repo(repo: str):
        cashed_repos = repositories_cache.get_cached_repositories()
        if len(cashed_repos) == 0:
            return gr.update(choices=[], value=[], interactive=True)
        if repo in cashed_repos:
            return gr.update(choices=cashed_repos, value=repo, interactive=True)
        return gr.update(choices=cashed_repos, value=cashed_repos[0], interactive=True)

    def update_shared():
        return gr.update(
            choices=repositories_cache.get_cached_files(), value=[], interactive=True
        )

    def display_document_links(documents):
        if len(documents) == 0:
            return ""
        markdown = "## Relevant Documents\n"
        stored_ids = []
        for doc in documents:
            if doc.metadata["ItemId"] in stored_ids:
                continue

            page_info = ""
            if "page_number" in doc.metadata:
                page_info = f" (Page {doc.metadata['page_number']})"

            if "http" in doc.metadata.get("ItemId", ""):
                markdown += f"[{doc.metadata['ItemId']}]({doc.metadata['ItemId']}) {page_info}\n"
            else:
                markdown += f"{doc.metadata['ItemId']} {page_info}\n"

            stored_ids.append(doc.metadata["ItemId"])
        return markdown

    # ==================================================================================
    # Document Load Functions

    def add_following_repo_branches(repo: str, branches: list[str], open_ai_key: str):
        for branch in branches:
            git_loader = GitDocumentLoader(
                repo_path=repo,
                branch=branch,
                loaded_repositories_and_files=repositories_cache,
            )
            create_vector_store_from_document_loader(
                git_loader, document_store, vectorstores_cache
            )

    def add_following_file(file: str, open_ai_key: str):
        zip_loader = ZipDocumentLoader(
            temp_file=file,
            loaded_repositories_and_files=repositories_cache,
        )
        create_vector_store_from_document_loader(
            zip_loader, document_store, vectorstores_cache
        )

    # ==================================================================================
    # RAG Functions

    def user_input(user_input: str, history: list):
        if not args.keep_history:
            history = []
        return "", history + [{"role": "user", "content": user_input}]

    def retrieve_documents_for_user(
        current_documents, files, branches, chatbot_box, model, api_key
    ):
        if len(chatbot_box) > 1:
            return current_documents

        retrieval = prepare_retriever(
            long_branches=branches,
            zip_files=files,
            docstore=document_store,
            loaded_vectorstores=vectorstores_cache,
            model_type=model,
            api_key=api_key,
            rerank=args.rerank,
            query_alteration=args.query_alteration,
        )
        documents = get_documents(chatbot_box[-1]["content"], retrieval)
        return documents

    def generate_answer(
        chatbot_box, current_documents, model, api_key, temperature, prompt
    ):
        for out in stream_rag(
            messages=chatbot_box,
            retrieved_documents=current_documents,
            model_type=model,
            api_key=api_key,
        ):
            yield out

    with demo:

        # ==================================================================================
        # State Variables
        current_documents_store = gr.State()

        # ===================================================================
        # Add Git Repo/Branch Section

        add_repo_markdown = gr.Markdown(
            """## Add Git Repository/Branch
            Add a git repository and branch to the cache.
            """,
            visible=False,
        )

        git_update_box = gr.Dropdown(
            choices=repositories_cache.get_cached_repositories(),
            allow_custom_value=True,
            visible=False,
            interactive=True,
            label="Git Repository Input",
        )
        with gr.Row():
            with gr.Column():
                return_button = gr.Button(
                    "Return", variant="secondary", visible=False, interactive=True
                )
            with gr.Column():
                git_submit_button = gr.Button(
                    "Submit Repo", variant="primary", visible=False
                )
        branch_update_box = gr.Dropdown(
            visible=False,
            multiselect=True,
            interactive=True,
            label="Branches to Cache",
            max_choices=args.max_branch_boxes,
        )
        with gr.Row():
            branch_quick_submit_button = gr.Button(
                "Quick Submit", variant="primary", visible=False
            )

        git_add_select_initial_boxes = [
            git_update_box,
            return_button,
            git_submit_button,
            add_repo_markdown,
        ]
        git_add_select_boxes = git_add_select_initial_boxes + [
            branch_update_box,
            branch_quick_submit_button,
        ]
        git_add_select_boxes_no_markdown = [
            git_update_box,
            return_button,
            git_submit_button,
        ] + [
            branch_update_box,
            branch_quick_submit_button,
        ]

        ## Git Section UI controll

        git_update_box.change(
            lambda: gr.update(visible=False),
            [],
            branch_update_box,
        )

        git_submit_button.click(
            selected_repo,
            [git_update_box],
            [
                branch_update_box,
                branch_quick_submit_button,
            ],
        )

        # ==================================================================================
        # Add Files Section
        add_file_markdown = gr.Markdown(
            """## Add Files
            Add files to the shared directory.
            """,
            visible=False,
        )

        file_add_box = gr.File(
            file_count="single",
            file_types=["file"],
            interactive=True,
            visible=False,
            label="Zip and File Uploader",
        )
        file_return_button = gr.Button("Return", variant="secondary", visible=False)

        file_section_boxes = [file_add_box, file_return_button, add_file_markdown]
        file_section_boxes_no_markdown = [file_add_box, file_return_button]

        ## Add Section UI controll

        # ==================================================================================
        # Config Section
        model_page_markdown = gr.Markdown(
            f"""## Configurations
            Configure the OpenAI model, temperature and the system prompt.
            """,
            visible=False,
        )

        change_temperature = gr.Slider(
            minimum=0.05,
            maximum=2.0,
            step=0.05,
            value=0.7,
            label="Temperature, Default = 0.7",
            interactive=True,
            visible=False,
        )
        change_system_prompt = gr.Textbox(
            label="System Prompt",
            value=PROMPTS.SYSTEM_PROMPT,
            visible=False,
            interactive=True,
            lines=10,
            max_lines=10,
        )
        open_ai_model = gr.Dropdown(
            choices=list(MODEL_TYPES.LLM_MODELS.keys()),
            value=list(MODEL_TYPES.LLM_MODELS.keys())[-1],
            label="Chosen Model",
            visible=False,
            interactive=True,
        )
        open_ai_key = gr.Textbox(
            label="OpenAI API Key",
            value="metacentrum",
            placeholder="metacentrum",
            visible=False,
            interactive=True,
            lines=1,
            max_lines=1,
        )

        with gr.Row():
            with gr.Column():
                temperature_return_button = gr.Button(
                    "Return", variant="secondary", visible=False
                )
            with gr.Column():
                reset_button = gr.Button("Reset", variant="primary", visible=False)

        config_section_boxes = [
            model_page_markdown,
            temperature_return_button,
            reset_button,
            change_temperature,
            change_system_prompt,
            open_ai_model,
            open_ai_key,
        ]

        ## Config Section UI controll
        reset_button.click(
            lambda: [gr.update(value=0.7), gr.update(value=PROMPTS.SYSTEM_PROMPT)],
            [],
            [change_temperature, change_system_prompt],
        )

        # ==================================================================================
        # Main section

        main_page_markdown = gr.Markdown(
            """# Document Bot
        Problem analysis with Retrieval Augmented Generation.""",
            visible=False,
        )

        with gr.Row():

            with gr.Column():
                git_box = gr.Dropdown(
                    choices=repositories_cache.get_cached_repositories(),
                    label="Git Repos",
                    value=(
                        []
                        if len(repositories_cache.get_cached_repositories()) == 0
                        else [repositories_cache.get_cached_repositories()[-1]]
                    ),
                    interactive=True,
                    visible=False,
                    multiselect=True,
                )
                version_box = gr.Dropdown(
                    choices=(
                        []
                        if len(repositories_cache.get_cached_repositories()) == 0
                        else repositories_cache.get_cached_repo_branches(
                            repositories_cache.get_cached_repositories()[-1]
                        )
                    ),
                    label="Branches",
                    value=(
                        []
                        if len(repositories_cache.get_cached_repositories()) == 0
                        else get_good_branches(
                            [repositories_cache.get_cached_repositories()[-1]]
                        )
                    ),
                    interactive=True,
                    multiselect=True,
                    visible=False,
                )
                shared_box = gr.Dropdown(
                    choices=repositories_cache.get_cached_files(),
                    interactive=True,
                    visible=False,
                    multiselect=True,
                    label="Additional Files",
                )

                question_box = gr.Textbox(
                    label="Question about documents",
                    lines=4,
                    interactive=True,
                    visible=False,
                )
                submit_button = gr.Button(
                    "Submit Question",
                    variant="primary",
                    interactive=True,
                    visible=False,
                )
                with gr.Row():
                    with gr.Column():
                        add_repo = gr.Button(
                            "Add Git Repository/Branch",
                            variant="secondary",
                            interactive=True,
                            visible=False,
                        )
                    with gr.Column():
                        add_file = gr.Button(
                            "Add Additional Directory",
                            variant="secondary",
                            interactive=True,
                            visible=False,
                        )
                config_button = gr.Button(
                    "Config", variant="secondary", interactive=True, visible=False
                )
            with gr.Column():
                chatbot_box = gr.Chatbot(
                    type="messages",
                    label="Answer",
                    visible=False,
                    height=None,
                    min_height=400,
                    max_height=1200,
                    show_label=False,
                )
                clear_history = gr.Button(
                    "Clear History",
                    variant="secondary",
                    interactive=True,
                    visible=False,
                )
                visible_documents = gr.Markdown(visible=False)

        main_page_boxes = [
            main_page_markdown,
            git_box,
            version_box,
            shared_box,
            question_box,
            submit_button,
            add_repo,
            add_file,
            config_button,
            chatbot_box,
            clear_history,
            visible_documents,
        ]

        ## Main section UI controll
        git_box.change(fn=changed_repo, inputs=[git_box], outputs=[version_box])

        branch_update_box.change(
            fn=changed_branches,
            inputs=[branch_update_box],
            outputs=[branch_quick_submit_button],
        )

        submit_button.click(
            user_input, [question_box, chatbot_box], [question_box, chatbot_box]
        ).then(
            retrieve_documents_for_user,
            inputs=[
                current_documents_store,
                shared_box,
                version_box,
                chatbot_box,
                open_ai_model,
                open_ai_key,
            ],
            outputs=[current_documents_store],
        ).then(
            display_document_links,
            [current_documents_store],
            [visible_documents],
        ).then(
            generate_answer,
            inputs=[
                chatbot_box,
                current_documents_store,
                open_ai_model,
                open_ai_key,
                change_temperature,
                change_system_prompt,
            ],
            outputs=[chatbot_box],
        ).then(
            lambda *args: callback.flag(args),
            [
                version_box,
                git_box,
                chatbot_box,
                shared_box,
                open_ai_model,
                change_temperature,
                change_system_prompt,
            ],
            [],
        )
        ############################################################################################################

        clear_history.click(lambda: gr.update(value=[]), [], chatbot_box)

        add_repo.click(
            lambda: len(main_page_boxes) * [gr.update(visible=False)],
            [],
            main_page_boxes,
        ).then(fn=update_repo, inputs=[], outputs=[git_update_box]).then(
            lambda: len(git_add_select_initial_boxes) * [gr.update(visible=True)],
            [],
            git_add_select_initial_boxes,
        )

        add_file.click(
            lambda: len(main_page_boxes) * [gr.update(visible=False)],
            [],
            main_page_boxes,
        ).then(
            lambda: len(file_section_boxes) * [gr.update(visible=True)],
            [],
            file_section_boxes,
        )

        config_button.click(
            lambda: len(main_page_boxes) * [gr.update(visible=False)],
            [],
            main_page_boxes,
        ).then(
            lambda: len(config_section_boxes) * [gr.update(visible=True)],
            [],
            config_section_boxes,
        )
        # ==================================================================================
        # Returns
        return_button.click(
            lambda: len(git_add_select_boxes) * [gr.update(visible=False)],
            [],
            git_add_select_boxes,
        ).then(fn=update_repo, inputs=[], outputs=[git_box]).then(
            lambda: len(main_page_boxes) * [gr.update(visible=True)],
            [],
            main_page_boxes,
        )

        file_return_button.click(
            lambda: len(file_section_boxes) * [gr.update(visible=False)],
            [],
            file_section_boxes,
        ).then(fn=update_repo, inputs=[], outputs=[git_box]).then(
            lambda: len(main_page_boxes) * [gr.update(visible=True)],
            [],
            main_page_boxes,
        )

        temperature_return_button.click(
            lambda: len(config_section_boxes) * [gr.update(visible=False)],
            [],
            config_section_boxes,
        ).then(
            lambda: len(main_page_boxes) * [gr.update(visible=True)],
            [],
            main_page_boxes,
        )

        # ==================================================================================
        # Submits with Returns

        file_add_box.upload(
            lambda: len(file_section_boxes_no_markdown)
            * [gr.update(interactive=False)],
            [],
            file_section_boxes_no_markdown,
        ).then(add_following_file, [file_add_box, open_ai_key], []).then(
            lambda: len(file_section_boxes_no_markdown)
            * [gr.update(visible=False, interactive=True)],
            [],
            file_section_boxes_no_markdown,
        ).then(
            lambda: gr.update(visible=False), [], add_file_markdown
        ).then(
            fn=update_shared, inputs=[], outputs=[shared_box]
        ).then(
            lambda: len(main_page_boxes) * [gr.update(visible=True)],
            [],
            main_page_boxes,
        )

        branch_quick_submit_button.click(
            lambda: len(git_add_select_boxes_no_markdown)
            * [gr.update(interactive=False)],
            [],
            git_add_select_boxes_no_markdown,
        ).then(
            add_following_repo_branches,
            [git_update_box, branch_update_box, open_ai_key] + [],
            [],
        ).then(
            lambda: gr.update(visible=False), [], add_repo_markdown
        ).then(
            lambda: len(git_add_select_boxes_no_markdown)
            * [gr.update(visible=False, interactive=True)],
            [],
            git_add_select_boxes_no_markdown,
        ).then(
            fn=update_new_repo, inputs=[git_update_box], outputs=[git_box]
        ).then(
            fn=changed_new_repo,
            inputs=[git_box, branch_update_box],
            outputs=[version_box],
        ).then(
            lambda: len(main_page_boxes) * [gr.update(visible=True)],
            [],
            main_page_boxes,
        )

        # ==================================================================================
        # Setup Section
        callback.setup(
            [
                version_box,
                git_box,
                chatbot_box,
                shared_box,
                open_ai_model,
                change_temperature,
                change_system_prompt,
            ],
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "data",
                "flagged_data_points_all",
            ),
        )

        ## Setup section controll
        demo.load(fn=update_repo, inputs=[], outputs=[git_box]).then(
            fn=changed_repo, inputs=[git_box], outputs=[version_box]
        ).then(fn=update_shared, inputs=[], outputs=[shared_box]).then(
            lambda: len(main_page_boxes) * [gr.update(visible=True)],
            [],
            main_page_boxes,
        )

    with redirect_stdout(sys.stderr):
        app, local, shared = demo.launch(
            share=False, server_name="0.0.0.0", server_port=args.port
        )


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-branch-boxes",
        default=10,
        type=int,
        help="Maximal number of branches to cache at once",
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Rerank the documents based on the query"
    )
    parser.add_argument(
        "--keep-history", action="store_true", help="Keep the history of the chatbot"
    )
    parser.add_argument(
        "--query-alteration", action="store_true", help="Alter query for better results"
    )
    parser.add_argument(
        "--port", default=7860, type=int, help="Port to run the Gradio server on"
    )
    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)
