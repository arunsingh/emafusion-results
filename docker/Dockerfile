###############################################################################
# Stage 0 – Build the Python wheel-house on a tiny Debian image               #
###############################################################################
FROM python:3.11-slim AS builder
WORKDIR /app

# copy sources first to leverage docker-layer-cache
COPY pyproject.toml poetry.lock requirements.txt ./
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -U pip wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

###############################################################################
# Stage 1 – Flatcar runtime image                                             #
###############################################################################
# NOTE: the Luzilla images publish Flatcar rootfs as an OCI image             #
#       (≈ 45 MB compressed). It contains no package manager –                 #
#       we bring in a standalone Python build.                                #
FROM ghcr.io/luzilla/flatcar:stable AS runtime   # ← **flatcar swap** :contentReference[oaicite:0]{index=0}

# Install a relocatable Python build (musl/glibc-compatible) ------------------
ARG PY_VER=3.11.9
ARG PY_BASEURL=https://github.com/astral-sh/python-build-standalone/releases/download/20250311
ADD ${PY_BASEURL}/cpython-${PY_VER}+20250311-x86_64-unknown-linux-gnu-install_only.tar.gz /tmp/python.tgz :contentReference[oaicite:1]{index=1}
RUN mkdir -p /opt/python && \
    tar -xzf /tmp/python.tgz -C /opt/python --strip-components=1 && \
    ln -s /opt/python/bin/python3 /usr/bin/python3 && \
    ln -s /opt/python/bin/pip3    /usr/bin/pip3 && \
    rm /tmp/python.tgz

# Copy the virtual-env we built in Stage 0
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/app/src"

# Copy source for runtime execution (small)
WORKDIR /app
COPY src/ ./src/
ENTRYPOINT ["python3", "-m", "emafusion.cli"]
CMD ["--help"]
