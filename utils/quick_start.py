"""
utils.quick_start
=================
Main simulation orchestrator for OmniSim.

Entry point:  ``run_simulation(config_file_list, ...)``

Workflow
--------
1. Load and merge configuration (system defaults + dataset overrides).
2. Connect to Elasticsearch and verify the index exists.
3. Load dataset (items, users, interactions) and build/restore user profiles.
4. Dispatch conversation generation across strategy queues
   (user_item_pairs / item_list / user_guest / random) using a thread pool.
5. Write completed dialogues to ``chats/{dataset}/{mode}/{strategy}/{run_ts}/``.
"""

import warnings
import time
import os
import json
import hashlib
import random
import traceback
import threading
import pandas as pd
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed

from logging import getLogger
from datetime import datetime
from openai import AuthenticationError, APIConnectionError, OpenAIError, RateLimitError
from elasticsearch import Elasticsearch
from elasticsearch import AuthenticationException, AuthorizationException, ConnectionError, TransportError

from utils.logger import init_logger
from utils.utils import init_seed, get_openai_clients
from utils.configurator import Config
from utils.dataset import Dataset
from utils.simulator import simulate_free, simulate_static, simulate_adaptive


def _file_md5(filepath: str) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _needs_profile_rebuild(config, data_path: str) -> bool:
    """
    Return True if user_profiles.csv needs to be (re)built:
      - user_profiles.csv doesn't exist, OR
      - hash file is missing, OR
      - any source file hash has changed since last build.
    """
    profiles_path = os.path.join(data_path, 'user_profiles.csv')
    hash_path = os.path.join(data_path, '.profile_hash.json')

    if not os.path.exists(profiles_path):
        return True
    if not os.path.exists(hash_path):
        return True

    try:
        with open(hash_path, 'r') as f:
            stored = json.load(f)
    except Exception:
        return True

    for cfg_key in ('file_users', 'file_interactions'):
        fname = config.get(cfg_key) or ''
        if str(fname).strip().lower() in ('', 'none', '~', 'null'):
            continue
        fpath = os.path.join(data_path, fname)
        if not os.path.exists(fpath):
            continue
        current_hash = _file_md5(fpath)
        if stored.get(fname) != current_hash:
            return True

    return False


def _cleanup_tmp_file(chat_folder: str, item_id):
    """Delete any leftover tmp file for item_id in case of a mid-conversation error."""
    import glob
    pattern = os.path.join(chat_folder, f"{item_id}-tmp-*.txt")
    for f in glob.glob(pattern):
        try:
            os.remove(f)
        except Exception:
            pass


def _save_profile_hashes(config, data_path: str):
    """Save MD5 hashes of source files used to build user_profiles.csv."""
    hash_path = os.path.join(data_path, '.profile_hash.json')
    hashes = {}
    for cfg_key in ('file_users', 'file_interactions'):
        fname = config.get(cfg_key) or ''
        if str(fname).strip().lower() in ('', 'none', '~', 'null'):
            continue
        fpath = os.path.join(data_path, fname)
        if os.path.exists(fpath):
            hashes[fname] = _file_md5(fpath)
    with open(hash_path, 'w') as f:
        json.dump(hashes, f, indent=2)

warnings.filterwarnings("ignore")


def _is_empty(val):
    return val is None or str(val).strip().lower() in ('none', '~', 'null', '')


SIMULATION_FUNCS = {
    "free": simulate_free,
    "static": simulate_static,
    "adaptive": simulate_adaptive,
}


def _run_parallel(sim_func, tasks, num_workers, chat_folder, logger, total):
    """Run simulation tasks in parallel using a thread pool.
    tasks: list of (identifier, params_dict)
    """
    counter = [0]
    lock = threading.Lock()
    abort_flag = threading.Event()

    def _run_one(identifier, params):
        if abort_flag.is_set():
            return
        try:
            sim_func(**params)
        except RateLimitError as e:
            if 'per day' in str(e).lower() or 'rpd' in str(e).lower():
                logger.error(f"Daily rate limit reached — stopping all remaining tasks: {e}")
                abort_flag.set()
            else:
                logger.warning(f"Error processing {identifier}: {e}")
            if 'target_item_id' in params:
                _cleanup_tmp_file(chat_folder, params['target_item_id'])
        except BaseException as e:
            msg = f"Error processing {identifier}: {e}\n{traceback.format_exc()}"
            logger.warning(msg)
            print(msg, flush=True)
            if 'target_item_id' in params:
                _cleanup_tmp_file(chat_folder, params['target_item_id'])
        with lock:
            counter[0] += 1
            if counter[0] % 10 == 0:
                logger.info(f"{counter[0]} conversations completed out of {total}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_run_one, ident, params): ident for ident, params in tasks}
        for f in as_completed(futures):
            pass  # exceptions are caught inside _run_one

    return counter[0]


def run_simulation(config_file_list=None, mode_override=None, strategies_override=None,
                   num_workers_override=None, chats_per_entry_override=None, pairs_file_override=None):
    """ entry of the running. """

    # load configurations
    config = Config(config_file_list=config_file_list)

    # CLI --mode overrides yaml value
    if mode_override:
        config['mode_refinement'] = mode_override
    if num_workers_override is not None:
        config['num_workers'] = int(num_workers_override)
    if chats_per_entry_override is not None:
        config['chats_per_entry'] = int(chats_per_entry_override)
    if pairs_file_override is not None:
        config['input_pairs_file'] = pairs_file_override
    init_seed(config['seed'])

    # logger initialization
    log_filepath = init_logger(config)
    logger = getLogger()
    filename = os.path.splitext(log_filepath)[0]

    # Validate hybrid scoring parameters
    for param in ('weight_es_score', 'weight_user_taste_short'):
        val = config[param]
        if val is not None:
            try:
                val = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"Config error: '{param}' must be a number between 0 and 1, got '{config[param]}'. Please reconfigure.")
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Config error: '{param}' must be between 0 and 1, got {val}. Please reconfigure.")

    logger.info(config)
    logger.info(config['library_name'] + ' version ' + str(config['library_version']))

    # load dataset
    timer_begins = time.time()
    dataset = Dataset(config)
    dataset.load_data()

    # create OpenAI instances
    client_chat, client_embeddings = get_openai_clients(config)
    # testing OpenAI clients
    try:
        response = client_chat.chat.completions.create(
            model=config["chat_model"],
            messages=[{"role": "user", "content": "Hello!"}],
        )
        logger.info("Connected to OpenAI.")
    except AuthenticationError:
        logger.error("Authentication failed: check your API key or endpoint.")
    except APIConnectionError:
        logger.error("Connection error: unable to reach Azure OpenAI service.")
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI chat mode: {e}")

    # create ElasticSearch instance
    username = os.getenv("ES_USER")
    pwd = os.getenv("ES_PWD")
    ES_HOST = config["es_host"]
    ES_INDEX = config["es_index"]
    
    # For local Elasticsearch, disable SSL verification completely
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    if username and pwd:
        es = Elasticsearch(
            ES_HOST, 
            basic_auth=(username, pwd), 
            verify_certs=False,
            ssl_show_warn=False
        )
    else:
        es = Elasticsearch(ES_HOST, verify_certs=False, ssl_show_warn=False)

    try:
        # Check existence of your specific index (requires only read access on that index)
        if es.indices.exists(index=ES_INDEX):
            logger.info(f"Connected to Elasticsearch (index '{ES_INDEX}' is accessible).")
        else:
            logger.warning(f"Index '{ES_INDEX}' not found — building it now ...")
            from scripts.build_es_index import build_index as _build_es_index
            sys_cfg = 'configs/system/system.yaml'
            build_cfgs = [sys_cfg] + [c for c in (config_file_list or []) if c != sys_cfg]
            _build_es_index(build_cfgs)
            logger.info(f"Index '{ES_INDEX}' built successfully.")
    except AuthenticationException:
        logger.error("Authentication failed: check username/password.")
    except AuthorizationException:
        logger.error("Not authorized: check user permissions.")
    except ConnectionError:
        logger.error("Connection error: cannot reach Elasticsearch host.")
    except TransportError as e:
        logger.error(f"Transport error: {e.info}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")

    # Auto-build or reload user_profiles.csv when source files changed
    user_profiles = {}
    data_path = os.path.join(config.get('data_path', 'data'), config['dataset'])
    user_profiles_path = os.path.join(data_path, 'user_profiles.csv')
    try:
        if _needs_profile_rebuild(config, data_path):
            # Check if at least one source file exists before attempting build
            has_source = any(
                os.path.exists(os.path.join(data_path, config.get(k) or ''))
                for k in ('file_users', 'file_interactions')
                if str(config.get(k) or '').strip().lower() not in ('', 'none', '~', 'null')
            )
            if has_source:
                logger.info("Building user_profiles.csv (source files new or changed)...")
                from utils.user_profile_builder import build_user_profiles
                build_user_profiles(config_file_list)
                _save_profile_hashes(config, data_path)
                logger.info("user_profiles.csv built and hashes saved.")
            else:
                logger.info("No user/interaction files found — skipping profile build.")

        if os.path.exists(user_profiles_path):
            profiles_df = pd.read_csv(user_profiles_path)
            user_profiles = profiles_df.set_index('userid').to_dict('index')
            logger.info(f"Loaded user profiles for {len(user_profiles)} users")
        else:
            logger.info("No user_profiles.csv found, proceeding without user profiles")
    except Exception as e:
        logger.warning(f"Error loading user profiles: {e}")

    # run simulation with different strategies
    # Set a run-level timestamp so all conversations in this run go to the same subfolder
    run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    config['run_timestamp'] = run_timestamp

    # Collect all strategies to run based on which parameters are set
    strategies_to_run = []
    if not _is_empty(config.get('input_pairs_file')):
        strategies_to_run.append('user_item_pairs')
    if not _is_empty(config.get('input_items_file')):
        strategies_to_run.append('item_list')
    if not _is_empty(config.get('input_users_file')):
        strategies_to_run.append('user_guest')
    if not _is_empty(config.get('guestuser_randomitem_records')):
        strategies_to_run.append('items')
    if not strategies_to_run:
        strategies_to_run.append('items')

    if strategies_override:
        strategies_to_run = [s for s in strategies_override if s in strategies_to_run or s in ('items', 'item_list', 'user_item_pairs', 'user_guest')]
    logger.info(f'Strategies to run: {strategies_to_run}')
    logger.info('Start producing simulated conversations...')

    mode_refinement = config['mode_refinement'].strip().lower()
    try:
        sim_func = SIMULATION_FUNCS[mode_refinement]
    except KeyError:
        raise ValueError(f"Unknown mode: {mode_refinement}")

    num_workers = int(config.get('num_workers') or 4)
    logger.info(f'Using {num_workers} parallel workers')

    for generation_strategy in strategies_to_run:
        config['generation_strategy'] = generation_strategy
        chat_folder = os.path.join(config['outputs_folder'], config['dataset'], config['mode_refinement'], generation_strategy, run_timestamp) + os.sep
        logger.info(f'--- Running strategy: {generation_strategy} ---')

        if generation_strategy == 'items':
            # Strategy 1: Random items for guest users
            num_chats = config['guestuser_randomitem_records'] or 40
            valid_items = dataset.df_items[
                dataset.df_items[dataset.itemid].notna() &
                (dataset.df_items[dataset.itemid].astype(str).str.strip() != '') &
                (dataset.df_items[dataset.itemid].astype(str).str.lower() != 'nan')
            ]
            tasks = []
            for i in range(num_chats):
                sample_row = valid_items.sample(1).iloc[0]
                sample_item_id = sample_row[dataset.itemid]
                params = {
                    "config": config, "dataset": dataset, "user_id": None,
                    "target_item_id": sample_item_id, "client_chat": client_chat,
                    "client_embeddings": client_embeddings, "es": es, "user_profiles": user_profiles
                }
                tasks.append((f"item {sample_item_id}", params))
            count = _run_parallel(sim_func, tasks, num_workers, chat_folder, logger, num_chats)
            logger.info(f'Done. Generated {count} guest user conversations.')

        elif generation_strategy == 'item_list':
            # Strategy 2: Generate N conversations per item
            input_file = config['input_items_file']
            if not input_file or not os.path.exists(input_file):
                logger.error(f"Input items file not found: {input_file}")
            else:
                items_df = pd.read_csv(input_file)
                col = 'item_id' if 'item_id' in items_df.columns else items_df.columns[0]
                item_ids = items_df[col].dropna().astype(str).tolist()
                n_per_item = int(config.get('chats_per_entry') or 1)
                total = len(item_ids) * n_per_item

                has_users = dataset.df_users is not None and not dataset.df_users.empty
                if has_users:
                    all_user_ids = dataset.df_users[dataset.userid].dropna().astype(str).tolist()
                    logger.info(f'Generating {n_per_item} conversation(s) per item with random users ({total} total)')
                else:
                    logger.info(f'Generating {n_per_item} anonymous conversation(s) per item ({total} total)')

                tasks = []
                for item_id in item_ids:
                    sampled_users = random.sample(all_user_ids, min(n_per_item, len(all_user_ids))) if has_users else [None] * n_per_item
                    for user_id in sampled_users:
                        params = {
                            "config": config, "dataset": dataset, "user_id": user_id,
                            "target_item_id": item_id, "client_chat": client_chat,
                            "client_embeddings": client_embeddings, "es": es, "user_profiles": user_profiles
                        }
                        tasks.append((f"item {item_id}", params))
                count = _run_parallel(sim_func, tasks, num_workers, chat_folder, logger, total)
                logger.info(f'Done. Generated {count} item-based conversations.')

        elif generation_strategy == 'user_item_pairs':
            # Strategy 3: Generate N conversations per user-item pair
            input_file = config['input_pairs_file']
            if not input_file or not os.path.exists(input_file):
                logger.error(f"Input pairs file not found: {input_file}")
            else:
                pairs_df = pd.read_csv(input_file)
                n_per_pair = int(config.get('chats_per_entry') or 1)
                total = len(pairs_df) * n_per_pair
                logger.info(f'Generating {n_per_pair} conversation(s) per pair for {len(pairs_df)} pairs ({total} total)')
                tasks = []
                for i, row in pairs_df.iterrows():
                    user_id = row.get('user_id') if 'user_id' in row else (row.get('userid') if 'userid' in row else None)
                    item_id = row.get('item_id') if 'item_id' in row else (row.get('product_id') if 'product_id' in row else None)
                    for _ in range(n_per_pair):
                        params = {
                            "config": config, "dataset": dataset, "user_id": user_id,
                            "target_item_id": item_id, "client_chat": client_chat,
                            "client_embeddings": client_embeddings, "es": es, "user_profiles": user_profiles
                        }
                        tasks.append((f"user {user_id} item {item_id}", params))
                count = _run_parallel(sim_func, tasks, num_workers, chat_folder, logger, total)
                logger.info(f'Done. Generated {count} user-item pair conversations.')

        elif generation_strategy == 'user_guest':
            # Strategy 4: Users + items sampled from ES
            if dataset.df_users is None or dataset.df_users.empty:
                logger.error("user_guest strategy requires file_users to be set in config.")
                continue

            input_users_file = config.get('input_users_file')
            if input_users_file and not _is_empty(input_users_file) and os.path.exists(input_users_file):
                users_df = pd.read_csv(input_users_file)
                col = 'user_id' if 'user_id' in users_df.columns else users_df.columns[0]
                user_ids = users_df[col].dropna().astype(str).tolist()
                logger.info(f"Loaded {len(user_ids)} users from {input_users_file}")
            else:
                user_ids = dataset.df_users[dataset.userid].dropna().tolist()
            n_per_user = int(config.get('chats_per_user') or config.get('chats_per_entry') or 5)
            total = len(user_ids) * n_per_user

            from utils.user_profile_builder import _build_demographic
            demo_profiles = {}
            for _, u_row in dataset.df_users.iterrows():
                uid = str(u_row[dataset.userid])
                demo_profiles[uid] = {'user_demographic': _build_demographic(u_row), 'user_likes_long': '', 'user_likes_short': ''}

            try:
                es_resp = es.search(index=ES_INDEX, body={
                    "query": {"function_score": {"query": {"match_all": {}}, "random_score": {}}},
                    "size": min(total * 3, 1000)
                })
                es_hits = es_resp['hits']['hits']
            except Exception as e:
                logger.error(f"Failed to sample items from ES: {e}")
                continue

            if not es_hits:
                logger.error("No items found in Elasticsearch index.")
                continue

            es_rows = [hit['_source'] for hit in es_hits]
            dataset.df_items = pd.DataFrame(es_rows)
            item_pool = [str(hit['_source'].get(dataset.itemid, hit['_id'])) for hit in es_hits]

            tasks = []
            for user_id in user_ids:
                sampled = random.sample(item_pool, min(n_per_user, len(item_pool)))
                for item_id in sampled:
                    params = {
                        "config": config, "dataset": dataset, "user_id": user_id,
                        "target_item_id": item_id, "client_chat": client_chat,
                        "client_embeddings": client_embeddings, "es": es, "user_profiles": demo_profiles,
                    }
                    tasks.append((f"user {user_id} item {item_id}", params))
            count = _run_parallel(sim_func, tasks, num_workers, chat_folder, logger, total)
            logger.info(f'Done. Generated {count} user-guest conversations.')
    
    logger.info(f'All simulated conversations have been saved to {chat_folder}')
    logger.info('filename pattern is: {item_id}-{number_turns}-{rec_attempts}-{succeed}-{timestamp}.txt')


