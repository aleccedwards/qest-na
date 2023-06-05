#!/usr/bin/env bash

set -e
set -x

DIR=$(mktemp -d)
PWD="$(pwd)"
PROJ="${PWD}"

cp -R "$PROJ" "$DIR"
cd "$DIR"

find . -name __pycache__ -exec rm -rf {} \; || :
find . -name '*.pyc' -delete || :

find . -name '.git' -exec rm -rf {} \; || :
find . -name '.gitignore' -delete || :
find . -name '.ps' -delete || :
find . -name '.gen' -delete || :
find . -name 'private' -exec rm -rf {} \; || :
find . -name 'counterexamples' -exec rm -rf {} \; || :
find . -name 'images' -exec rm -rf {} \; || :
find . -name 'outputs' -exec rm -rf {} \; || :
# find . -name 'spaceex_exe' -exec rm -rf {} \; || :
find . -name '.json' -exec rm -rf {} \; || :
find . -name '.vscode' -exec rm -rf {} \; || :
find . -name *.pkl -exec rm -rf {} \; || :
find . -name 'package.sh' -exec rm -rf {} \; || :
find . -name 'project.zip' -exec rm -rf {} \; || :


zip -r qest-na.zip *
cp qest-na.zip "$PROJ"

sha1sum qest-na.zip
# unzip -l project.zip