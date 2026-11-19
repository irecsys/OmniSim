"""Download one revision of a local embedding model once, then run offline.
Writes the files to --out (set embeddings_local_path to it) and prints / stores the sha256 of model.safetensors (set
embeddings_weights_sha256 to it). Example: README, "Optional: local embeddings"."""
import argparse
import hashlib
import json
import os
import sys


# sha256 of a file, read in 1 MB chunks.
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# Downloads the revision, writes WEIGHTS_SHA256, checks --expect-sha256.
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', required=True, help='Hugging Face model id')
    ap.add_argument('--revision', required=True, help='commit sha (never a branch name)')
    ap.add_argument('--out', required=True, help='target directory')
    ap.add_argument('--expect-sha256', default=None, help='fail when model.safetensors has another sha256')
    args = ap.parse_args(argv)
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision.lower()):
        sys.exit('--revision must be a full 40-character commit sha')
    from huggingface_hub import snapshot_download
    out = os.path.expanduser(args.out)
    path = snapshot_download(repo_id=args.model, revision=args.revision, local_dir=out)
    weights = os.path.join(path, 'model.safetensors')
    digest = sha256_file(weights) if os.path.exists(weights) else None
    with open(os.path.join(out, 'WEIGHTS_SHA256'), 'w', encoding='utf-8') as f:
        json.dump({'model': args.model, 'revision': args.revision, 'model.safetensors': digest}, f, indent=1)
    print(json.dumps({'local_path': out, 'model.safetensors sha256': digest}, indent=1))
    if args.expect_sha256 and digest != args.expect_sha256.lower():
        sys.exit(f'sha256 {digest} != expected {args.expect_sha256}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
