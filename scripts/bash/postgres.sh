#!/bin/sh
sudo docker run \
  --name postgres-db-cred \
  -p 15432:5432 \
  -e POSTGRES_USER=demo_user \
  -e POSTGRES_PASSWORD=demo_password \
  -e POSTGRES_DB=demo_db \
  -d postgres
