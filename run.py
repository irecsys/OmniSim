"""
Command-line entry point: run the dialogue simulation, or build the local index (--build-index).

--config takes one or more YAML files separated by spaces; later files override earlier ones.
Exit code 3 means that an LLM or embedding call failed during the run (details in the run
summary file).

Usage:
    python run.py --config configs/imdb/imdb.yaml --build-index
    python run.py --config configs/imdb/imdb.yaml
    python run.py --config configs/imdb/imdb.yaml --mode adaptive --chats-per-entry 1 --num-workers 1
"""
import argparse
import logging
import sys

from utils.quick_start import check_retrieval_backend, run_simulation


def build_index(config_file_list):
    """--build-index: build the local index (index_dir, else index/{dataset}) from the items CSV."""
    from utils.retrieval.local_index import build_local_index, index_dir_for, load_config
    cfg = load_config(config_file_list)
    check_retrieval_backend(cfg, logging.getLogger())
    build_local_index(cfg, index_dir_for(cfg), config_files=config_file_list)


def exit_code(summary):
    """run.py's exit code from the run summary returned by run_simulation."""
    return int((summary or {}).get('exit_code', 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/imdb/imdb.yaml')
    parser.add_argument('--mode', type=str, choices=['free', 'static', 'adaptive'],
                        help='Override mode_refinement in config (free | static | adaptive)')
    parser.add_argument('--strategy', type=str,
                        help='Run only specific strategies, comma-separated (e.g. item_list,user_guest)')
    parser.add_argument('--build-index', action='store_true',
                        help='Build the local retrieval index from the items CSV, then exit')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override num_workers in config')
    parser.add_argument('--chats-per-entry', type=int, default=None,
                        help='Override chats_per_entry in config')
    parser.add_argument('--pairs-file', type=str, default=None,
                        help='Override input_pairs_file in config')

    args, _ = parser.parse_known_args()

    config_file_list = args.config.strip().split(' ') if args.config else None

    if args.build_index:
        build_index(config_file_list)
    else:
        strategies = [s.strip() for s in args.strategy.split(',')] if args.strategy else None

        summary = run_simulation(config_file_list=config_file_list, mode_override=args.mode,
                                 strategies_override=strategies,
                                 num_workers_override=args.num_workers,
                                 chats_per_entry_override=args.chats_per_entry,
                                 pairs_file_override=args.pairs_file)
        # 0, or 3 after any LLM / embedding error (fallback, aborted / dropped conversation, profile failure,
        # failed startup check); details in the run summary file
        sys.exit(exit_code(summary))
