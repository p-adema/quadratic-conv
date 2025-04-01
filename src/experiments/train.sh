#!/bin/sh
screen -S 'training' sh -c "PYTHONUNBUFFERED=1 uv run python $1 2> train.err | tee train.log"