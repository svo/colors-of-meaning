#!/usr/bin/env bash

image=$1

docker manifest rm "svanosselaer/colors-of-meaning-${image}:latest" 2>/dev/null || true

docker manifest create \
  "svanosselaer/colors-of-meaning-${image}:latest" \
  --amend "svanosselaer/colors-of-meaning-${image}:amd64" \
  --amend "svanosselaer/colors-of-meaning-${image}:arm64" &&
docker manifest push "svanosselaer/colors-of-meaning-${image}:latest"
