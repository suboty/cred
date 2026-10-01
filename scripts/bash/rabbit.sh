#!/bin/sh
set -e

CONTAINER_NAME="rabbitmq-cred"
IMAGE="rabbitmq:3-management"
HOST_AMQP_PORT="5672"
HOST_UI_PORT="15672"
CONTAINER_AMQP_PORT="5672"
CONTAINER_UI_PORT="15672"

if [ -n "$(docker ps -a -q -f name=^/${CONTAINER_NAME}$)" ]; then
    if [ -z "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
        echo "Starting existing container ${CONTAINER_NAME}..."
        docker start "${CONTAINER_NAME}"
    else
        echo "Container ${CONTAINER_NAME} is already running."
    fi
else
    echo "Creating container ${CONTAINER_NAME}..."
    docker run \
      --name "${CONTAINER_NAME}" \
      -p "${HOST_AMQP_PORT}:${CONTAINER_AMQP_PORT}" \
      -p "${HOST_UI_PORT}:${CONTAINER_UI_PORT}" \
      -e RABBITMQ_DEFAULT_USER=guest \
      -e RABBITMQ_DEFAULT_PASS=guest \
      -d "${IMAGE}"
fi