---
title: "Installation"
sequence: 1
---

This tutorial assumes you will use Python as language to work in. Though Tensorflow supports multiple languages, Python has the best documentation available.


# Python + Tensorflow

Support for Tensorflow differs per operating system, resulting in different recommended versions for Python:

- Ubuntu: 2.7 or 3.3+
- Mac OS X: 2.7 or 3.3+
- Windows: 3.5.x

Detailed instructions can be found on the [Tensorflow installation pages](https://www.tensorflow.org/install/). When in doubt, follow the instructions for Tensorflow using CPU support only, using virtualenv on Python 3.

# Docker

If you are familiar with Docker and don't want to install Python/Tensorflow locally you can Docker images.

## Ubuntu

Follow the [Tensorflow guide](https://www.tensorflow.org/install/install_linux#installing_with_docker).

## Mac OS X

Follow the [Tensorflow guide](https://www.tensorflow.org/install/install_mac#installing_with_docker).

## Windows

- Start a docker terminal.
- Run `docker-machine ip` in order to know the IP your docker environment is using.
- Run `docker run -it -p 8888:8888 --name tf tensorflow/tensorflow` to start the docker container.
- On your machine, navigate to `<docker-ip>:8888` to connect to the jupyter notebook server. Use the token provided in the docker terminal.
- Shutdown the docker container using `ctrl+c`, you can restart it using `docker start -i tf`, or permanently remove it using `docker rm tf`.