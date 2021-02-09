#!/bin/bash

python url_transition.py
sh scrape.sh
sh extract.sh
python document_clustering.py
python data_postprocess.py

