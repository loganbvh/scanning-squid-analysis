#! /bin/bash
source /anaconda3/bin/activate scanning-squid-analysis
cd "$(dirname "$0")"
cd ..
python -m scanning-squid-analysis.gui.window