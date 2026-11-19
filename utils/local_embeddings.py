"""Local embedding models through sentence-transformers (embeddings_provider: sentence-transformers).
The vector space is identified by model id, revision and weights' sha256; files load from embeddings_local_path or the
Hugging Face cache, never downloaded unless embeddings_allow_download is true. Keys: docs/API.md."""
import hashlib
import os
import threading


class ConfigError(ValueError):
    """A configuration error of the local embedding settings."""


ST_PROVIDER = 'sentence-transformers'
DTYPES = ('float32', 'bfloat16')
INSTALL_HINT = "pip install -r requirements-local-embeddings.txt"
MIN_SENTENCE_TRANSFORMERS = (2, 2)


# True for a missing setting: None, empty, "none", "~" or "null".
def _unset(v):
    return v is None or str(v).strip().lower() in ('', 'none', '~', 'null')


# True when the embeddings provider is the local sentence-transformers one.
def is_local(config):
    from utils.utils import resolve_embeddings_provider
    return resolve_embeddings_provider(config) == ST_PROVIDER


# True for a Hugging Face id ("org/name"), not a local path.
def _looks_like_hf_id(model):
    return bool(model) and not os.path.isabs(str(model)) and not str(model).startswith('.') and '/' in str(model)


def embedding_spec(config):
    """The settings that define the local vector space (and the loading settings), validated: ConfigError for a
    float16 dtype, an HF id without a revision, a non-positive batch size."""
    spec = {
        'model': config.get('embeddings_model') or 'all-MiniLM-L6-v2',
        'revision': None if _unset(config.get('embeddings_revision')) else str(config.get('embeddings_revision')),
        'local_path': None if _unset(config.get('embeddings_local_path')) else os.path.expanduser(
            str(config.get('embeddings_local_path'))),
        'weights_sha256': None if _unset(config.get('embeddings_weights_sha256')) else str(
            config.get('embeddings_weights_sha256')).lower(),
        'allow_download': bool(config.get('embeddings_allow_download') or False),
        'dtype': str(config.get('embeddings_dtype') or 'float32').lower(),
        'device': str(config.get('embeddings_device') or 'cpu'),
        'modalities': None if _unset(config.get('embeddings_modalities')) else str(
            config.get('embeddings_modalities')).lower(),
        'num_threads': None if _unset(config.get('embeddings_num_threads')) else int(config.get('embeddings_num_threads')),
        'batch_size': int(config.get('embeddings_batch_size') or 32),
        'query_prompt': None if _unset(config.get('embeddings_query_prompt')) else str(
            config.get('embeddings_query_prompt')),
        'document_template': None if _unset(config.get('embeddings_document_template')) else str(
            config.get('embeddings_document_template')),
        'dim': None if _unset(config.get('embeddings_dim')) else int(config.get('embeddings_dim')),
    }
    if spec['dim'] is not None and spec['dim'] < 1:
        raise ConfigError(f"Config error: 'embeddings_dim' must be a positive number of components, got {spec['dim']}.")
    if spec['document_template'] is not None:
        try:
            spec['document_template'].format(title='t', text='x')
        except (KeyError, IndexError, ValueError) as e:
            raise ConfigError(f"Config error: 'embeddings_document_template' may only use {{title}} and {{text}}, "
                              f"got {spec['document_template']!r} ({e}).")
    if spec['dtype'] not in DTYPES:
        raise ConfigError(f"Config error: 'embeddings_dtype' must be one of {', '.join(DTYPES)} (float16 gives NaNs "
                          f"or silently worse vectors with some models), got {spec['dtype']!r}.")
    if spec['modalities'] not in (None, 'text'):
        raise ConfigError(f"Config error: 'embeddings_modalities' must be text or unset, got {spec['modalities']!r}.")
    if spec['batch_size'] < 1:
        raise ConfigError(f"Config error: 'embeddings_batch_size' must be at least 1, got {spec['batch_size']}.")
    if _looks_like_hf_id(spec['model']) and spec['revision'] is None:
        raise ConfigError(f"Config error: embeddings_model {spec['model']!r} is a Hugging Face id, so "
                          f"'embeddings_revision' (a commit sha) is required: the repository can change.")
    return spec


SPACE_FIELDS = ('model', 'revision', 'weights_sha256', 'query_prompt', 'document_template', 'dim', 'dtype')


def space_identity(config):
    """What makes two local vector sets comparable: the backend, the model id, its revision and
    weights' sha256, the query prompt, the document template, the truncation dim, the dtype and the normalisation.
    Not the location of the files, the device, the threads or the batch size."""
    spec = embedding_spec(config)
    return {'backend': ST_PROVIDER, **{k: spec[k] for k in SPACE_FIELDS}, 'normalization': 'l2'}


def space_id(config):
    """'st:' + sha256 of space_identity: the key of the embedding cache rows of a local model."""
    import json
    blob = json.dumps(space_identity(config), sort_keys=True, ensure_ascii=False)
    return 'st:' + hashlib.sha256(blob.encode('utf-8')).hexdigest()


def library_versions():
    """Versions of the local-embedding packages (for the index meta); None for a package that is missing."""
    out = {}
    for name in ('sentence_transformers', 'transformers', 'torch'):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def check_local_embeddings_config(config):
    """At the start of a run / a build: the local embedding settings are valid (only for the local provider)."""
    if is_local(config):
        embedding_spec(config)


# sha256 of a file, read in 1 MB chunks.
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# The SentenceTransformer class; ImportError with the install hint when missing or too old.
def _import_sentence_transformers():
    try:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(f"embeddings_provider '{ST_PROVIDER}' needs the optional local-embedding packages, which "
                          f"requirements.txt does not install: {INSTALL_HINT}") from e
    version = tuple(int(x) for x in sentence_transformers.__version__.split('.')[:2] if x.isdigit())
    if version < MIN_SENTENCE_TRANSFORMERS:
        raise ImportError(f"sentence-transformers {sentence_transformers.__version__} is too old: {INSTALL_HINT}")
    return SentenceTransformer


class LocalEmbeddingsClient:
    """Mimics client.embeddings.create(model=..., input=...).data[i].embedding for a local model. Vectors are
    L2-normalised."""
    accepts_role = True                   # callers may pass role='query' / 'document'

    def __init__(self, config):
        self.spec = embedding_spec(config)
        self._lock = threading.Lock()
        SentenceTransformer = _import_sentence_transformers()
        import torch
        s = self.spec
        if s['num_threads']:
            torch.set_num_threads(s['num_threads'])
        source = s['local_path'] or s['model']
        kwargs = {'device': s['device'], 'model_kwargs': {'torch_dtype': getattr(torch, s['dtype'])}}
        if not s['local_path']:
            kwargs['revision'] = s['revision']
            kwargs['local_files_only'] = not s['allow_download']
        if s['modalities'] == 'text':
            kwargs['config_kwargs'] = {'vision_config': None, 'audio_config': None}
        try:
            self._model = SentenceTransformer(source, **kwargs)
        except Exception as e:
            text = str(e)
            if 'Processor' in text and 'import' in text.lower():
                raise ImportError(f"loading {s['model']} failed ({text}); multimodal processors need torchvision: "
                                  f"{INSTALL_HINT}") from e
            if not s['local_path'] and not s['allow_download']:
                raise ConfigError(f"Config error: the local embedding model {s['model']} (revision {s['revision']}) "
                                  f"is not on this machine ({type(e).__name__}: {text[:200]}). Download it once with "
                                  f"scripts/fetch_embedding_model.py and set embeddings_local_path, or set "
                                  f"embeddings_allow_download: true.") from e
            raise
        self.weights_path = self._weights_file()
        if s['weights_sha256']:
            found = sha256_file(self.weights_path) if self.weights_path else None
            if found != s['weights_sha256']:
                raise ConfigError(f"Config error: the weights of {s['model']} ({self.weights_path}) have sha256 "
                                  f"{found}, not embeddings_weights_sha256 {s['weights_sha256']}.")
        self.embeddings = self

    def _weights_file(self):
        """model.safetensors of the loaded files: in embeddings_local_path, or in the Hugging Face cache for the
        revision (never downloaded here); None when there is none (e.g. a model with other weight files)."""
        s = self.spec
        if s['local_path']:
            path = os.path.join(s['local_path'], 'model.safetensors')
            return path if os.path.exists(path) else None
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(s['model'], 'model.safetensors', revision=s['revision'], local_files_only=True)
        except Exception:
            return None

    def encode(self, texts, role=None):
        """L2-normalised vectors; a query gets embeddings_query_prompt in front; with embeddings_dim the vectors
        are cut to their first dim components and normalised again (cutting a unit vector does not keep it unit)."""
        import numpy as np
        texts = list(texts)
        if role == 'query' and self.spec['query_prompt']:
            texts = [self.spec['query_prompt'] + t for t in texts]
        with self._lock:         # one model, one encoding at a time (worker threads share the client)
            vecs = self._model.encode(texts, batch_size=self.spec['batch_size'], normalize_embeddings=True,
                                      convert_to_numpy=True)
        vecs = np.asarray(vecs, dtype=np.float64)
        if self.spec['dim']:
            vecs = vecs[:, :self.spec['dim']]
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    # Same interface as client.embeddings.create of the OpenAI SDK.
    def create(self, model=None, input=None, role=None, **kwargs):
        texts = [input] if isinstance(input, str) else list(input)
        vecs = self.encode(texts, role=role).tolist()
        data = [type("d", (), {"embedding": v})() for v in vecs]
        return type("r", (), {"data": data})()
