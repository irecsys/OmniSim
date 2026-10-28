# H&M: from the raw data to the local index

The repository ships the H&M items ready to use: `data/hm/hm.zip` contains `hm.csv` (3,246 products)
with a precomputed 1536-dim `embedding_vector` column (`text-embedding-3-small`). That is the normal way
to use the dataset and needs no API key:

```bash
unzip -o data/hm/hm.zip -d data/hm/
python run.py --config configs/hm/hm.yaml --build-index
```

`configs/hm/hm.yaml` sets `precomputed_embedding_col: embedding_vector`, so building the index only reads
the vectors from the CSV. **Use these shipped vectors to reproduce published results.**

The rest of this page describes how `hm.csv` is produced from the raw scrape, for when you need to
regenerate it or adapt the preprocessing.

## 1. Get the raw data

The raw product file is not shipped with this repository. `hm.csv` was built from the H&M product data
listed under Data sources in the README. The steps below record how it was processed, for readers who have
a file with the same columns (`productId`, `productName`, `brandName`, `price`, `colorName`, `mainCatCode`,
`details`, `materials`; the file used for the shipped `hm.csv` had 9,677 rows, one per product colour).
Below it is called `/path/to/work/raw_handm.csv`.

## 2. Preprocess

```bash
python utils/hm/preprocessor.py --input /path/to/work/raw_handm.csv --output /path/to/work/data/hm/hm.csv
```

> **Always pass `--input` and `--output` explicitly, with the output outside `data/hm/`.** The defaults
> of `utils/hm/preprocessor.py` are inconsistent: the `HMPreprocessor` class writes to `data/hm/hm.csv`
> by default and would overwrite the shipped file (and its vectors).

The preprocessor maps `mainCatCode` to `gender` / `item_type` / `gender_type`, merges colour variants of the
same product name into one row, and builds `description`. Its output has every column of the shipped
`hm.csv` except `embedding_vector`, and was checked to be byte-identical to it otherwise.

## 3. Build the index (computes the embeddings)

The preprocessed CSV has no vector column, so the index builder computes the vectors from
`embedding_fields` (`gender item_type productName colorName details`) with `embeddings_provider` /
`embeddings_model` from `configs/system/system.yaml`. This calls the paid embedding API
(H&M: 3,246 texts, 33 requests, about 385k tokens).

Point the dataset at the work directory with a small override config, `/path/to/work/work.yaml`:

```yaml
data_path: /path/to/work/data
index_dir: /path/to/work/index/hm
```

Then preview and build:

```bash
python scripts/build_local_index.py --config "configs/hm/hm.yaml /path/to/work/work.yaml" --dry-run
python run.py --config "configs/hm/hm.yaml /path/to/work/work.yaml" --build-index
```

- `--dry-run` reports how many items would be embedded (and the number of requests and tokens); it sends
  nothing and writes nothing. `scripts/build_local_index.py --max-embed-items N` refuses to start if
  more than N items need embedding.
- The CSV is never modified. The computed vectors are cached in `embedding_cache.db` in the index
  directory, keyed by item id, embedding text and model; rebuilding the index (for example after changing
  `bm25_tokenizer`) only embeds new or changed items.
- The automatic build in `run.py` (when the index is missing) never calls the embedding API; it stops and
  asks you to run `--build-index`.

**Recomputed vectors are close to, but not bit-identical with, the shipped ones**: the embedding service does not return bit-identical vectors, so an index built from recomputed vectors can rank items slightly differently. **Use the shipped vectors to reproduce published results.** With the shipped vectors, this pipeline produces an index byte-identical to the one built from `hm.csv`.
