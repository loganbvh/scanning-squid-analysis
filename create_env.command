#! /bin/bash
source ~/opt/anaconda3/bin/activate
cd "$(dirname "$0")"
conda env create --file environment.yml
