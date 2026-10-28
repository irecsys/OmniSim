"""
Tmp vs final conversation file names. The simulator's real tmp names are always
recognised; final names never are, whatever the ids contain; compute_nlp_metrics no longer counts the
utterances of a tmp file (a crash leftover duplicating a saved conversation) twice.
"""
import os

import pytest

import utils.simulator as sim
from utils.dialogue_files import TMP_NAME_RE, is_tmp_dialogue, tmp_dialogue_name

IDS = [('u1', 'tt0105643'), (None, 'tt1'), ('hu7', 1257558004), ('u_tmp3', 'tt2'), ('u1', 'tmp'),
       ('tmp', 'attmpt'), ('x-tmp-1', 'tt3'), ('user', 'tt-tmp-20261025120059-abcdef12')]


@pytest.mark.parametrize('user_id, item_id', IDS)
@pytest.mark.parametrize('strategy', ['user_item_pairs', 'item_list', 'user_guest', 'items'])
def test_setup_output_tmp_names_are_tmp(tmp_path, strategy, user_id, item_id):
    config = {'outputs_folder': str(tmp_path), 'dataset': 'toy', 'mode_refinement': 'free',
              'generation_strategy': strategy, 'run_timestamp': '20261025120059'}
    _, tmp_file, ts = sim._setup_output(config, item_id, user_id=user_id)
    assert is_tmp_dialogue(tmp_file)
    assert os.path.basename(tmp_file) == tmp_dialogue_name(strategy, item_id, user_id, ts, tmp_file[-12:-4])


@pytest.mark.parametrize('user_id, item_id', IDS)
@pytest.mark.parametrize('turns, recs, succeed', [(3, 1, 1), (11, 6, 0), (31, 0, 0), (100, 12, 1)])
def test_final_names_are_never_tmp(tmp_path, user_id, item_id, turns, recs, succeed):
    tmp = tmp_path / 'x-tmp-20261025120059-0123abcd.txt'
    tmp.write_text('Bot: hi\n', encoding='utf-8')
    final = sim._save_conversation(str(tmp_path), user_id, item_id, turns, recs, succeed, '20261025120059', str(tmp))
    assert not is_tmp_dialogue(final), final


def test_pattern_edges():
    assert is_tmp_dialogue('/a/b/items-tt1-tmp-20261025120059-0a1b2c3d.txt')
    assert not is_tmp_dialogue('items-tt1-tmp-20261025120059-0A1B2C3D.txt')      # uuid hex is lower case
    assert not is_tmp_dialogue('items-tt1-tmp-2026102512005-0a1b2c3d.txt')       # 13-digit time stamp
    assert not is_tmp_dialogue('u_tmp3-tt1-3-1-1-20261025120059.txt')
    assert not is_tmp_dialogue('tmp-notes.txt')
    assert TMP_NAME_RE.pattern == r'-tmp-\d{14}-[0-9a-f]{8}\.txt$'


def test_compute_nlp_metrics_skips_tmp_files(tmp_path):
    import scripts.compute_nlp_metrics as cnm
    run = tmp_path / 'chats' / 'toy' / 'free' / 'user_item_pairs' / '20261025120059'
    run.mkdir(parents=True)
    text = 'Bot: What are you after?\nUser: A haunted farmhouse story.\nSystem: <END> Session ended successfully.\n'
    (run / 'u1-tt1-3-1-1-20261025120059.txt').write_text(text, encoding='utf-8')
    (run / 'user_item_pairs-tt1-u1-tmp-20261025120059-0a1b2c3d.txt').write_text(text, encoding='utf-8')  # crash leftover
    (run / 'u_tmp3-tt2-3-1-1-20261025120101.txt').write_text('Bot: Hello there.\nUser: Something calm.\n', encoding='utf-8')
    utts = cnm.load_utterances(str(tmp_path / 'chats'))
    assert sorted(utts) == sorted(['What are you after?', 'A haunted farmhouse story.', 'Hello there.', 'Something calm.'])


def test_tools_use_is_tmp_dialogue_only():
    # Every tool that skips tmp files uses is_tmp_dialogue; no substring checks ('tmp' in name) are left
    import pathlib
    import re
    pattern = re.compile(r"""["']tmp["']\s+(?:not\s+)?in\b""")
    hits = [f"{p}:{n}" for d in ('UI', 'utils', 'scripts') for p in sorted(pathlib.Path(d).rglob('*.py'))
            for n, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1) if pattern.search(line)]
    assert hits == []
