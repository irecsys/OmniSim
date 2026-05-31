import argparse

from utils.quick_start import run_simulation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/imdb/imdb.yaml')
    parser.add_argument('--mode', type=str, choices=['free', 'static', 'adaptive'],
                        help='Override mode_refinement in config (free | static | adaptive)')
    parser.add_argument('--strategy', type=str,
                        help='Run only specific strategies, comma-separated (e.g. item_list,user_guest)')
    parser.add_argument('--build-index', action='store_true',
                        help='Build the Elasticsearch index from items CSV, then exit')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override num_workers in config')
    parser.add_argument('--chats-per-entry', type=int, default=None,
                        help='Override chats_per_entry in config')
    parser.add_argument('--pairs-file', type=str, default=None,
                        help='Override input_pairs_file in config')

    args, _ = parser.parse_known_args()

    config_file_list = args.config.strip().split(' ') if args.config else None

    if args.build_index:
        from scripts.build_es_index import build_index
        build_index(config_file_list)
    else:
        strategies = [s.strip() for s in args.strategy.split(',')] if args.strategy else None

        run_simulation(config_file_list=config_file_list, mode_override=args.mode,
                       strategies_override=strategies,
                       num_workers_override=args.num_workers,
                       chats_per_entry_override=args.chats_per_entry,
                       pairs_file_override=args.pairs_file)
