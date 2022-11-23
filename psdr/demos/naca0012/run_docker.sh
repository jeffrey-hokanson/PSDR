#! /bin/bash
docker run -it --rm --entrypoint /bin/bash --mount type=bind,source="$PWD",target='/workdir' jeffreyhokanson/naca0012:latest
