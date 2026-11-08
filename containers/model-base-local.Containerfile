# Local-only model base bootstrap.
#
# Use this when the host already has the repo venv and a uv-managed Python.
# Build with scripts/build_base.sh model. The build mounts the local Python and
# site-packages read-only; no network is used.

ARG BASE_OS_IMAGE=docker.io/library/rust:1.95-bookworm
FROM ${BASE_OS_IMAGE}

RUN mkdir -p /opt/python && cp -a /host-python/. /opt/python/
RUN mkdir -p /opt/site && cp -a /host-site/. /opt/site/

ENV PATH=/opt/python/bin:$PATH
ENV PYTHONPATH=/opt/site

WORKDIR /app
