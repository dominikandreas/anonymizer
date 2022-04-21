#!/bin/bash

cd "$(dirname "$0")"
tag_name="uai_anonymizer"

if docker image history $tag_name > /dev/null; then
    echo Not rebuilding docker image as an image with the tag "$tag_name" already exists. 
    echo You can manually re-build it via "docker build . -t $tag_name"
else
    echo Rebuilding docker image $tag_name
    docker build . -t $tag_name
fi

docker run --gpus all -it --rm -v $PWD:/anonymizer anonym /bin/bash -c "cd /anonymizer && PYTHONPATH=$PWD /bin/bash"
