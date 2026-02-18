FROM ubuntu:latest
LABEL authors="stanl"

ENTRYPOINT ["top", "-b"]