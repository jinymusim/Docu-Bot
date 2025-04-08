FROM jupyter/base-notebook:x86_64-python-3.11.6

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY --chown=${NB_UID}:${NB_GID} . /tmp/docu-bot
WORKDIR /tmp/docu-bot

# Install the package
RUN pip install --no-cache-dir .

# Switch back to notebook user
USER ${NB_USER}

# Set the working directory to the notebook directory
WORKDIR /home/${NB_USER}/work
# Copy necessary project files to work directory if needed
# RUN cp -r /tmp/docu-bot/* /home/${NB_USER}/work/ # Currently is mounted as a volume

# Clean up
USER root
RUN rm -rf /tmp/docu-bot
USER ${NB_USER}


ARG BUILD_DATE=unknown-date
ARG VCS_REF=unknown-rev
ARG VERSION=unknown-version
ARG BUILD_TYPE=manual
ARG BUILD_HOSTNAME
ARG BUILD_JOB_NAME
ARG BUILD_NUMBER

LABEL maintainer="jinymusim" \
      org.label-schema.schema-version="1.0.0" \
      org.label-schema.description="Retrieval augmented system for conversation over git repositories" \
      org.label-schema.usage="git@github.com:jinymusim/Docu-Bot.git" \
      org.label-schema.version="$VERSION" \
      org.label-schema.build-date="$BUILD_DATE" \
      org.label-schema.build-type="$BUILD_TYPE" \
      org.label-schema.build-ci-job-name="$BUILD_JOB_NAME" \
      org.label-schema.build-ci-build-id="$BUILD_NUMBER" \
      org.label-schema.build-ci-host-name="$BUILD_HOSTNAME" \
      org.label-schema.url="git@github.com:jinymusim/Docu-Bot.git" \
      org.label-schema.vcs-url="git@github.com:jinymusim/Docu-Bot.git" \
      org.label-schema.vcs-ref="$VCS_REF"

# Expose the Jupyter port
EXPOSE 8888

# Set default command to start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0"]