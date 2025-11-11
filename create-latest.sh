#!/usr/bin/env bash

image=$1

docker manifest rm "svanosselaer/python-sprint-zero-${image}:latest" 2>/dev/null || true

docker manifest create \
  "svanosselaer/python-sprint-zero-${image}:latest" \
  --amend "svanosselaer/python-sprint-zero-${image}:amd64" \
  --amend "svanosselaer/python-sprint-zero-${image}:arm64" &&
docker manifest push "svanosselaer/python-sprint-zero-${image}:latest"
