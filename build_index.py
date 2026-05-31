"""Wrapper to invoke scripts/build_es_index.py from root."""
import sys
sys.argv[0] = 'scripts/build_es_index.py'
from scripts.build_es_index import build_index

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/imdb/imdb.yaml')
    args = parser.parse_args()
    config_list = args.config.strip().split(' ')
    build_index(config_list)
