#! /bin/bash
#eval $(docker-machine env -u)
docker build -t naca0012:v1 .

docker login
docker tag naca0012:v1 jeffreyhokanson/naca0012:latest
docker tag naca0012:v1 jeffreyhokanson/naca0012:v1
docker push jeffreyhokanson/naca0012:v1
docker push jeffreyhokanson/naca0012:latest
