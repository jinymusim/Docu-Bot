#!/usr/bin/make -f
SHELL:=/bin/bash -O extglob
DOCKER_BUILD :=
JUP_PORT := 13385
PROJECT_SLUG := $(shell basename $(PWD))
PROJECT_SLUG_UNDERSCORE = $(subst -,_,$(PROJECT_SLUG))
DOCKER_NAME := $(PROJECT_SLUG)
CON_NAME := $(PROJECT_SLUG)-$(USER)
DOCKER_PATH := jinymusim/$(DOCKER_NAME):latest

jupyter-run:
	docker run -d $(DOCKER_ARGS) \
            --rm \
            --name $(CON_NAME) \
            -e PYTHONPATH=/home/jovyan/work/src \
            --network host \
            -v $(shell pwd):/home/jovyan/work \
            $(DOCKER_PATH) \
            jupyter lab --port $(JUP_PORT) /home/jovyan/work

build-docker:
	docker build $(DOCKER_BUILD) -t $(DOCKER_PATH) .

build-python-package:
	python setup.py bdist_wheel