# utils/hm — H&M dataset preprocessing (raw_handm.csv → hm.csv). Full pipeline: docs/hm_pipeline.md
# Usage (always pass explicit paths, outside data/hm/: the class defaults overwrite data/hm/hm.csv):
#   (raw_handm.csv is not distributed with the repository; put your copy at <work>/raw_handm.csv)
#   python utils/hm/preprocessor.py --input <work>/raw_handm.csv --output <work>/data/hm/hm.csv   # step 1
#   python run.py --config "configs/system/system.yaml configs/hm/hm.yaml <work>/work.yaml" --build-index   # step 2
# The shipped data/hm/hm.zip already holds hm.csv with precomputed vectors (no API needed).
