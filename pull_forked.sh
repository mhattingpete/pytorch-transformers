#!/bin/bash
git checkout master
git pull git@github.com:huggingface/pytorch-transformers.git #-s recursive -X theirs
git push
