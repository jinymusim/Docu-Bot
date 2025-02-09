import runpy

from setuptools import setup, find_packages

__version__ = runpy.run_path("src/docu_bot/version.py")["__version__"]

setup(
    name="Docu-Bot",
    package_dir={"": "src"},
    include_package_data=True,
    packages=find_packages("src"),
    version=__version__,
    author="jinymusim",
    url="https://github.com/jinymusim/Docu-Bot.git",
    license="CC0-1.0",
    install_requires=[
        "chardet==5.2.0",
        "fuzzywuzzy==0.18.0",
        "gradio==5.5.0",
        "langchain==0.3.7",
        "langchain-community==0.3.6",
        "langchain-openai==0.2.7",
        "langchain-chroma==0.1.4",
        "ninja==1.11.1.1",
        "openai==1.54.4",
        "packaging==24.2",
        "pypdf==5.1.0",
        "wheel==0.45.0",
        "ragas==0.2.13",
    ],
    scripts=[
        "scripts/web_rag.py",
    ],
)
