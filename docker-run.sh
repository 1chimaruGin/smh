#!/bin/bash

docker run -it --rm -p 7860:7860 -v $(pwd):/app --name conservative conservative_image bash
