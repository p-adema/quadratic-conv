#!/bin/sh
screen -S 'training' bash -c "PYTHONUNBUFFERED=1 uv run python $1 > >(tee -a train_out.log) 2> >(tee -a train_err.log >&2)"