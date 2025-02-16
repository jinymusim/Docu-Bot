import json
import os
import git
import logging

PERMITED_FILE_EXTENSIONS = [
    # ".txt",
    ".pdf",
    ".md",
    ".rst",
    # ".py",
]


class LoadedRepositoriesAndFiles:
    def __init__(
        self,
        json_file=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "data",
            "repositories_and_files.json",
        ),
    ):
        self._json_file = json_file
        if not os.path.exists(json_file):
            self._json_data = {}
        else:
            with open(json_file, "r") as json_fp:
                self._json_data = json.load(json_fp)
        for repo_or_files, data_path in self._json_data.items():
            if not os.path.exists(data_path):
                logging.warning(
                    f"Data path {data_path} for {repo_or_files} does not exist. Removing from loaded repositories and files."
                )
                self._json_data.pop(repo_or_files)

    def add_directory(self, repo_or_files, data_path):
        self._json_data[repo_or_files] = data_path

        os.makedirs(os.path.dirname(self._json_file), exist_ok=True)

        with open(self._json_file, "w") as f:
            json.dump(self._json_data, f)

    def get_cached_repositories(self):
        return [
            os.path.dirname(repo) for repo in self._json_data.keys() if ".git" in repo
        ]

    def get_cached_repo_branches_short(self, repo):
        return [
            os.path.basename(cache_repo)
            for cache_repo in self._json_data.keys()
            if repo in cache_repo
        ]

    def get_cached_repo_branches(self, repo):
        return [
            os.path.join(
                os.path.basename(os.path.dirname(cache_repo)),
                os.path.basename(cache_repo),
            )
            for cache_repo in self._json_data.keys()
            if repo in cache_repo
        ]

    def get_cached_files(self):
        return [repo for repo in self._json_data.keys() if ".git" not in repo]


def get_file_link(repo_path_or_key, file_path, save_path):
    if repo_path_or_key.endswith(".git"):
        branch = "master"
    else:
        branch = os.path.basename(repo_path_or_key)
        repo_path_or_key = os.path.dirname(repo_path_or_key)

    if repo_path_or_key.endswith(".git"):
        repo_path_or_key = repo_path_or_key.split(".git")[0]
    # Malfomed repo_path_or_key
    else:
        return None

    relative_path = os.path.relpath(file_path, save_path)

    return f"{repo_path_or_key}/blob/{branch}/{relative_path}"


def get_available_branches(repo):
    # Check if proper git repo format
    if not repo.endswith(".git"):
        return []
    branches = []
    g = git.cmd.Git()
    # Try to lookup all of git branches
    try:
        # Utilize git command to list all branches
        for ref in g.ls_remote("--heads", repo).split("\n"):
            branches.append(ref.split("/")[-1])
        return branches
    # Execption if git repo is not found or not accessible
    except Exception as e:
        print(e)
        return []
