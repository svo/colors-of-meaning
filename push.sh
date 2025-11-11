#!/usr/bin/env bash

image=$1 &&
architecture=$2 &&

if [ -z "$architecture" ]; then
  docker push "svanosselaer/colors-of-meaning-${image}" --all-tags
else
  docker push "svanosselaer/colors-of-meaning-${image}:${architecture}"
fi
