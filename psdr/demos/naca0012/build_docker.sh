#! /bin/bash
#eval $(docker-machine env -u)
docker build -t naca0012:v1 .

# docker login

# docker tag oas:v1 jeffreyhokanson/oas:v1
# docker tag oas:v1 jeffreyhokanson/oas:latest
# docker push jeffreyhokanson/oas:v1
# docker push jeffreyhokanson/oas:latest
