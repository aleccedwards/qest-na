#!/usr/bin/env bash

# error refine

python3 main.py -c experiments/error-refine/pwa-refine-config.yaml
python3 main.py -c experiments/error-refine/pwa-refine-config.yaml --error-check True

python3 main.py -c experiments/error-refine/pwc-refine-config.yaml
python3 main.py -c experiments/error-refine/pwc-refine-config.yaml --error-check True