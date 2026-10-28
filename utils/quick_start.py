"""
utils.quick_start
=================
Main simulation orchestrator for OmniSim.

Entry point:  ``run_simulation(config_file_list, ...)``

Workflow
--------
1. Load and merge configuration (system defaults + dataset overrides).
2. Open the local retrieval index (built from the items CSV if missing).
3. Load dataset (items, users, interactions) and build/restore user profiles.
4. Dispatch conversation generation across strategy queues
   (user_item_pairs / item_list / user_guest / random) using a thread pool.
5. Write completed dialogues to ``chats/{dataset}/{mode}/{strategy}/{run_ts}/``.
"""

import io
import warnings
import time
import os
import json
import hashlib
import random
import traceback
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from logging import getLogger
from datetime import datetime
from openai import AuthenticationError, APIConnectionError, OpenAIError, RateLimitError

from utils import llm_calls
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


# ERROR logged when a quota error (llm_calls.is_quota_exhausted) stops the run, per stop reason
_QUOTA_LOG = {
    llm_calls.THETA_403_STOP:
        "ThetaEdgeCloud returned HTTP 403 (provider {provider}, {task}): treated as the quota being used up. "
        "This may be only the 10-minute window (100 requests per 10 minutes, back after a few minutes) or the daily "
        "quota. No further conversation is started in this run (remaining tasks and later strategies are skipped, see "
        "the run summary). Limit the request rate with scripts/run_rate_limited.py or use fewer --num-workers, then "
        "rerun the unfinished tasks. Error: {error}",
    llm_calls.OPENAI_DAILY_STOP:
        "Daily rate limit reached (RateLimitError, provider {provider}, {task}): the per-day limit is used up and resets "
        "the next day. No further conversation is started in this run (remaining tasks and later strategies are "
        "skipped, see the run summary). Rerun the unfinished tasks tomorrow or with another API key. Error: {error}",
    llm_calls.OPENAI_QUOTA_STOP:
        "insufficient_quota (RateLimitError, provider {provider}, {task}): the account's quota is used up and does not "
        "come back by itself. No further conversation is started in this run (remaining tasks and later strategies "
        "are skipped, see the run summary). Check the account's quota / billing, then rerun the unfinished tasks. "
        "Error: {error}",
}


def _run_parallel(sim_func, tasks, num_workers, chat_folder, logger, total):
    """Run simulation tasks in parallel using a thread pool.
    tasks: list of (identifier, params_dict)
    """
    counter = [0]
    lock = threading.Lock()
    abort_flag = threading.Event()             # used only without a run summary (_run_parallel called directly)
    stop_reason = [None]
    summary = llm_calls.active_run()           # run summary (None when called outside run_simulation)
    strategy = tasks[0][1]['config'].get('generation_strategy') if tasks else None
    if summary is not None:
        summary.dispatched(strategy, len(tasks))

    def _stop_if_fail(e):
        if summary is not None and summary.on_llm_error == 'fail' and llm_calls.is_llm_error(e):
            if not summary.stop.is_set():
                logger.error(f"on_llm_error: fail — a conversation stopped on an LLM / embedding error; "
                             f"no further conversation is started")
            summary.stop.set()

    def _dropped(identifier, e):
        # conversations stopped inside simulate_* are merged by llm_calls.conversation; count the rest here
        if summary is not None and not getattr(e, 'omnisim_merged', False):
            summary.dropped_outside(strategy, identifier, e)

    def _dispatch_stopped():
        """Why no new task may start (a quota error, llm_calls.is_quota_exhausted): for the whole run when there is a run summary,
        so later strategies are skipped too (as with on_llm_error: fail); else for this call only."""
        if summary is not None:
            return (summary.dispatch_stop or {}).get('reason')
        return stop_reason[0] if abort_flag.is_set() else None

    def _stop_dispatch(reason, identifier, provider):
        stop_reason[0] = reason
        abort_flag.set()
        if summary is not None:
            summary.stop_dispatch(reason, identifier, provider)

    def _run_one(identifier, params):
        reason = _dispatch_stopped()
        if reason:
            if summary is not None:
                summary.skipped(strategy, reason)
            return
        if summary is not None and summary.stop.is_set():      # on_llm_error: fail
            summary.skipped(strategy, 'on_llm_error=fail')
            return
        try:
            sim_func(**params)
        except BaseException as e:
            if isinstance(e, RateLimitError):
                logger.warning(f"Error processing {identifier}: {llm_calls.sanitize(e)}")
            else:
                msg = llm_calls.mask_secrets(f"Error processing {identifier}: {e}\n{traceback.format_exc()}")
                logger.warning(msg)
                print(msg, flush=True)
            if 'target_item_id' in params:
                _cleanup_tmp_file(chat_folder, params['target_item_id'])
            _dropped(identifier, e)
            _stop_if_fail(e)
            # a quota / rate-limit error of the provider that made the call stops the whole run
            provider = llm_calls.provider_of(e, params['config'])
            reason = llm_calls.is_quota_exhausted(e, provider)
            fail_mode_stop = summary is not None and summary.stop.is_set()   # on_llm_error: fail stops on its own
            if reason and not _dispatch_stopped() and not fail_mode_stop:
                logger.error(_QUOTA_LOG[reason].format(provider=provider, task=identifier, error=llm_calls.sanitize(e)))
                _stop_dispatch(reason, identifier, provider)
        with lock:
            counter[0] += 1
            if counter[0] % 10 == 0:
                logger.info(f"{counter[0]} conversations completed out of {total}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_run_one, ident, params): ident for ident, params in tasks}
        for f in as_completed(futures):
            pass  # exceptions are caught inside _run_one

    return counter[0]


def _open_local_backend(config, config_file_list, logger):
    """Load the local index; build it first if it is missing (never calling the embedding API),
    then check that it was built with the config's embedding provider/model."""
    from utils.retrieval.backend import LocalBackend
    from utils.retrieval.local_index import (EmbeddingsNeeded, build_local_index, check_embedding_space,
                                             index_dir_for, load_config)

    index_dir = index_dir_for(config)
    if not os.path.exists(os.path.join(index_dir, 'meta.json')):
        logger.warning(f"Local index '{index_dir}' not found — building it now ...")
        files = ' '.join(config_file_list or [])
        try:
            build_local_index(load_config(config_file_list or []), index_dir, config_files=config_file_list,
                              max_embed_items=0)
        except EmbeddingsNeeded as e:
            raise RuntimeError(
                f"Could not build the local index '{index_dir}' for dataset '{config['dataset']}': {e}.\n"
                "The items CSV has no precomputed embedding column (precomputed_embedding_col), so the vectors "
                "must be computed with the paid embedding API, and the automatic build never does that. "
                "Build the index explicitly (computed vectors are cached in the index directory; the CSV is "
                "not modified):\n"
                f"  python run.py --config \"{files}\" --build-index\n"
                "To see how many items would be embedded first:\n"
                f"  python scripts/build_local_index.py --config \"{files}\" --dry-run") from e
        except Exception as e:
            raise RuntimeError(
                f"Could not build the local index '{index_dir}' for dataset '{config['dataset']}': {e}") from e
        logger.info(f"Local index '{index_dir}' built.")
    backend = LocalBackend(index_dir)
    check_embedding_space(backend.meta, config, index_dir, logger)
    logger.info(f"Opened local index '{index_dir}' ({backend.n} items).")
    return backend


def check_retrieval_backend(config, logger):
    """v2 has only the local index. A leftover `retrieval_backend` key: absent → local; 'local' →
    accepted with a deprecation warning; 'elasticsearch' (removed in v2) or anything else → error."""
    value = config.get('retrieval_backend')
    if value is None:
        return
    if value == 'local':
        logger.warning("Config key 'retrieval_backend' is deprecated: OmniSim v2 always uses the local index. "
                       "Remove the key from your config.")
        return
    if value == 'elasticsearch':
        raise ValueError("Config error: retrieval_backend 'elasticsearch' is no longer supported. OmniSim v2 removed "
                         "Elasticsearch and uses the built-in local index; remove the key from your config, "
                         "or check out the v1.x branch (or tag v1.0) for the Elasticsearch version.")
    raise ValueError(f"Config error: 'retrieval_backend' must be 'local' (deprecated, can be removed), got {value!r}.")


def _sample_guest_items(backend, config, itemid, size):
    """user_guest: up to `size` random items → (item rows, item_pool of id strings)."""
    from utils.retrieval import sampling
    rows = sampling.sample_items(backend, size)
    return rows, [str(r.get(itemid)) for r in rows]


def run_simulation(config_file_list=None, mode_override=None, strategies_override=None,
                   num_workers_override=None, chats_per_entry_override=None, pairs_file_override=None):
    """ entry of the running.

    Returns the run summary (dict; 'exit_code' is run.py's exit code: 0, or 3 after any LLM / embedding error),
    which is also written to {outputs_folder}/{dataset}/{mode}/run_summary_{run_ts}.json.
    """
    llm_calls.deactivate_run()
    try:
        _run_simulation(config_file_list, mode_override, strategies_override,
                        num_workers_override, chats_per_entry_override, pairs_file_override)
    except llm_calls.RunAborted as e:
        getLogger().error(f"Run stopped before any conversation was dispatched: {e}")
        llm_calls.active_run().aborted(e.stage, e.error)
    except BaseException as e:
        summary = llm_calls.active_run()
        if summary is not None:
            summary.crashed(e)
            _finish_run(summary)
        raise
    return _finish_run(llm_calls.active_run())


def _finish_run(summary):
    try:
        return summary.write()
    finally:
        llm_calls.deactivate_run()


def _run_simulation(config_file_list, mode_override, strategies_override,
                    num_workers_override, chats_per_entry_override, pairs_file_override):
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
    summary = llm_calls.RunSummary(config, config_file_list)
    llm_calls.activate_run(summary)
    llm_calls.check_on_llm_error(config)

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
    check_retrieval_backend(config, logger)

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
        response = llm_calls.chat(client_chat, 'startup_check',
            model=config["chat_model"],
            messages=[{"role": "user", "content": "Hello!"}],
        )
        logger.info("Connected to OpenAI.")
    except AuthenticationError as e:
        logger.error("Authentication failed: check your API key or endpoint.")
        llm_calls.startup_failed(e)
    except APIConnectionError as e:
        logger.error("Connection error: unable to reach Azure OpenAI service.")
        llm_calls.startup_failed(e)
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {llm_calls.sanitize(e)}")
        llm_calls.startup_failed(e)
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI chat mode: {llm_calls.sanitize(e)}")
        llm_calls.startup_failed(e)

    # open the retrieval backend
    backend = _open_local_backend(config, config_file_list, logger)

    # Auto-build or reload user_profiles.csv when source files changed
    user_profiles = {}
    data_path = os.path.join(config.get('data_path', 'data'), config['dataset'])
    user_profiles_path = os.path.join(data_path, 'user_profiles.csv')
    try:
        profiles_in_memory = None
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
                profiles_df, failures = build_user_profiles(config_file_list) or (None, [])
                if failures:
                    # Do not cache profiles with failed LLM calls (the next run would reuse them silently)
                    users = sorted({str(u) for u, _, _ in failures})
                    logger.warning(f"User profiles: {len(failures)} LLM call(s) failed for users {users}; "
                                   f"user_profiles.csv and .profile_hash.json were not updated, this run uses "
                                   f"the incomplete profiles in memory")
                    profiles_in_memory = pd.read_csv(io.StringIO(profiles_df.to_csv(index=False)))  # as if read back
                    summary.profile_result(True, failures, False)
                else:
                    _save_profile_hashes(config, data_path)
                    logger.info("user_profiles.csv built and hashes saved.")
                    summary.profile_result(True, [], profiles_df is not None)
            else:
                logger.info("No user/interaction files found — skipping profile build.")

        if profiles_in_memory is not None:
            user_profiles = profiles_in_memory.set_index('userid').to_dict('index')
            logger.info(f"Using in-memory user profiles for {len(user_profiles)} users")
        elif os.path.exists(user_profiles_path):
            profiles_df = pd.read_csv(user_profiles_path)
            user_profiles = profiles_df.set_index('userid').to_dict('index')
            logger.info(f"Loaded user profiles for {len(user_profiles)} users")
        else:
            logger.info("No user_profiles.csv found, proceeding without user profiles")
    except llm_calls.LLMCallFailed as e:
        raise llm_calls.RunAborted('profiles', e) from e     # on_llm_error: fail
    except Exception as e:
        logger.warning(f"Error loading user profiles: {llm_calls.sanitize(e)}")

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

    chat_folders = {}   # strategy -> its output folder, for the closing log lines only
    for generation_strategy in strategies_to_run:
        config['generation_strategy'] = generation_strategy
        chat_folder = os.path.join(config['outputs_folder'], config['dataset'], config['mode_refinement'], generation_strategy, run_timestamp) + os.sep
        chat_folders[generation_strategy] = chat_folder
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
                    "client_embeddings": client_embeddings, "backend": backend, "user_profiles": user_profiles
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
                            "client_embeddings": client_embeddings, "backend": backend, "user_profiles": user_profiles
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
                            "client_embeddings": client_embeddings, "backend": backend, "user_profiles": user_profiles
                        }
                        tasks.append((f"user {user_id} item {item_id}", params))
                count = _run_parallel(sim_func, tasks, num_workers, chat_folder, logger, total)
                logger.info(f'Done. Generated {count} user-item pair conversations.')

        elif generation_strategy == 'user_guest':
            # Strategy 4: Users + items sampled from the local index
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
                rows, item_pool = _sample_guest_items(backend, config, dataset.itemid, min(total * 3, 1000))
            except Exception as e:
                logger.error(f"Failed to sample items from the local backend: {e}")
                continue

            if not rows:
                logger.error("No items found in the local index.")
                continue

            dataset.df_items = pd.DataFrame(rows)

            tasks = []
            for user_id in user_ids:
                sampled = random.sample(item_pool, min(n_per_user, len(item_pool)))
                for item_id in sampled:
                    params = {
                        "config": config, "dataset": dataset, "user_id": user_id,
                        "target_item_id": item_id, "client_chat": client_chat,
                        "client_embeddings": client_embeddings, "backend": backend, "user_profiles": demo_profiles,
                    }
                    tasks.append((f"user {user_id} item {item_id}", params))
            count = _run_parallel(sim_func, tasks, num_workers, chat_folder, logger, total)
            logger.info(f'Done. Generated {count} user-guest conversations.')
    
    logger.info('Simulated conversations were saved to one folder per strategy:')
    for generation_strategy, chat_folder in chat_folders.items():
        logger.info(f'  {generation_strategy}: {chat_folder}')
    logger.info('filename pattern is: {user_id}-{item_id}-{number_turns}-{rec_attempts}-{succeed}-{timestamp}.txt')


