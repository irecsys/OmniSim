"""Everything that decides the dialogue is 2.0.0's: 21 files have the sha256 of 2.0.0 (tag v2.0.0), and
configs/system/system.yaml differs from 2.0.0 only in its library_version line. In utils/utils.py and
utils/retrieval/local_index.py every top-level function and class has 2.0.0's source except the listed client /
index-text ones (fixtures: 2.0.0's per-definition sha256)."""
import ast
import hashlib
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, 'tests', 'fixtures', 'interface')

# sha256 of the files of 2.0.0 (git show v2.0.0:<path>)
UNCHANGED = {
    'run.py': '7f636de20bed7f8663c89473433bd349de5861f5ff5fee293dfe05b71867d763',
    'utils/simulator.py': '04e7ba300f4fc6f95a6427285c64ac801a7e13cf5a3c6f355145db2ad0ccd8c8',
    'utils/quick_start.py': '964350f987f81b080a99255dbc780a38a06a97b755d1058e9711eb0a5bdd271c',
    'utils/user_profile_builder.py': 'acb1d4f7dcdbd557d6a25adc4922312d17c4814300b3f82af9a3e7700fb54cf3',
    'utils/dataset.py': '311e98ddb6893837c323845f09a00a9fc9e86f9ddc189d99cc4813bd7a63ca03',
    'utils/llm_calls.py': 'e485c5269cd005f216ebaeefc49dd5ece22ec92b2800dfef2d0b14b92c30bbcd',
    'utils/dialogue_files.py': 'aca6007831236f57136c97fe29303e7cfa9b0b14b6ec84127a85d05ee27f877d',
    'utils/configurator.py': '54bde9b24e46c603e595bbd4ec3ffcf6ff829f84720e37eac35a80f66b612e76',
    'utils/argument_list.py': '20731892844edfea4c94c4ad217cacd5d11d36674e9e9e0999904cc70ece06b8',
    'utils/logger.py': 'aa82ceee5797e0caf54b7ff4a8202fbf4a5d689226e59372f62c85d3fdcbc072',
    'utils/retrieval/local_search.py': 'deff95e73402e029d08f82f644521973c1066f9e224b6df91022567b785628c4',
    'utils/retrieval/filters.py': '1e70377051c502b25ed1fc9a7096e819d730454def1d70110140d4b61bf44618',
    'utils/retrieval/query_parse.py': '9035079da3e1fae6bd41d506b04d6810d951d444e250b56b40dcd812fba7543d',
    'utils/retrieval/backend.py': '784a03d8c096a92ce9b93e3137ad5de8af9f081af54e9402624d52a645c34383',
    'utils/retrieval/bm25.py': 'eacea4580fe9325bdae30dfa2d68a544d63b2c293aae88000337314a9981773f',
    'utils/retrieval/ref_lookup.py': '13b5709f8499a7d455a588ca0847392169a832ba3023162ecd17fa2e68cdb58c',
    'utils/retrieval/sampling.py': 'eb4c41efaea871be200bf282adca333f4e7ff86f5042aa4996f8bc8cac1760b7',
    'configs/prompts/default.yaml': '1286e715a6d3734ae968fd2a6391f8c6a291238a0b2e397d83d799515c4d9847',
    'configs/prompts/phrase_templates.yaml': '613849d96a550eaa1e58bcb76c4ced8aed1970fe0589d7bc9aec7f422629aabd',
    'configs/imdb/imdb.yaml': '5b254f0d46fdffb002d33a290e59b75923002edb93b8fcab75c2f64de165ac3c',
    'configs/hm/hm.yaml': '6a99159cacb8e8dcf139c48ba432733ba11adbf406ae7bb0a19361963941292b',
}
# the definitions 2.1.0 changes (and adds), per file; every other top-level definition is 2.0.0's
CHANGED = {
    'utils/utils.py': {'changed': {'SentenceTransformerEmbeddingsClient', '_build_embeddings_client',
                                   '_wrap_chat_client', '_build_chat_client'},
                       # reply handling for openai_compatible (tests/test_reply_handling.py)
                       'added': {'ThinkingTruncated', 'NoReplyAfterThinking', 'ThinkingNotOff', '_reasoning_tokens',
                                 '_thinking_signs', '_strip_thinking', '_current_site'}},
    'utils/retrieval/local_index.py': {'changed': {'embedding_texts', '_cache_keys', '_api_embed_fn'},
                                       'added': {'space_key'}},
}


def sha256(path):
    return hashlib.sha256(open(os.path.join(ROOT, path), 'rb').read()).hexdigest()


def defs(path):
    src = open(os.path.join(ROOT, path), encoding='utf-8', newline='').read().replace('\r\n', '\n')
    tree = ast.parse(src)
    return {n.name: hashlib.sha256(ast.get_source_segment(src, n).encode('utf-8')).hexdigest() for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


# configs/system/system.yaml: 2.0.0's sha256 once its one library_version line is set back to 2.0.0's
SYSTEM_YAML = 'configs/system/system.yaml'
SYSTEM_YAML_2_0_0 = '81ece9be0e6b9a4e9bbf8fc92d3af472a606d52d2d0dbc57e1fa6b26d457e688'
VERSION_LINE = (b'library_version: "2.1.0"', b'library_version: "2.0.0"')


def test_dialogue_logic_prompts_and_configs_are_2_0_0s_byte_for_byte():
    assert len(UNCHANGED) == 21
    assert {p: sha256(p) for p in UNCHANGED} == UNCHANGED


def test_system_yaml_differs_from_2_0_0_only_in_library_version():
    lines = open(os.path.join(ROOT, SYSTEM_YAML), 'rb').read().split(b'\n')
    hits = [i for i, line in enumerate(lines) if line.rstrip(b'\r') == VERSION_LINE[0]]
    assert len(hits) == 1
    lines[hits[0]] = lines[hits[0]].replace(VERSION_LINE[0], VERSION_LINE[1])
    assert hashlib.sha256(b'\n'.join(lines)).hexdigest() == SYSTEM_YAML_2_0_0


def test_changed_files_differ_only_in_the_listed_definitions():
    for path, ch in CHANGED.items():
        old = json.load(open(os.path.join(FIX, 'defs_' + os.path.basename(path) + '.json'), encoding='utf-8'))
        new = defs(path)
        assert set(new) == set(old) | ch['added'], path
        same = {k for k in old if new[k] == old[k]}
        assert set(old) - same == ch['changed'], path


def test_embeddinggemma_example_overlay_has_embedding_keys_only():
    new = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'examples', 'imdb_embeddinggemma.yaml'), encoding='utf-8'))
    assert {k for k in new if not k.startswith('embeddings_')} == {'precomputed_embedding_col', 'index_dir'}
    assert new['precomputed_embedding_col'] is None and new['embeddings_dim'] is None      # ~: the native 768
    assert new['embeddings_allow_download'] is False
