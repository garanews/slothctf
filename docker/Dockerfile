# SlothCTF sandbox image: the minimum to compile and run model-written C and
# Python. Built locally with `python main.py sandbox-build` (never auto-pulled).
# Containers run with --user 65534 (nobody), --network none and a read-only root
# filesystem, so no build-time user or entrypoint setup is needed here.
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev python3 \
    && rm -rf /var/lib/apt/lists/*

# Default workdir matches the bind mount used by DockerBackend (/work).
WORKDIR /work
