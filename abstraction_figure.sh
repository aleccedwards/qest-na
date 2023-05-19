#!/usr/bin/env bash

# Abstraction figures

python3 benchmarks.py -b nl2 -f experiments/nl2/nl2-concrete.pdf
# python3 main.py -r 1 -c experiments/nl2/pwc-nl2-config.yaml --output-type plot flowpipe --n-procs 1 
# python3 main.py -r 1 -c experiments/nl2/pwa-nl2-config.yaml --output-type plot flowpipe --n-procs 1 
python3 main.py -r 1 -c experiments/nl2/nl-nl2-config.yaml --output-type plot flowpipe --n-procs 1 
