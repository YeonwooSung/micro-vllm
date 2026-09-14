#!/usr/bin/env python3
"""Download a Hugging Face snapshot into --out. Resume-safe.

  python3 python/hf_fetch.py --repo Qwen/Qwen3.6-35B-A3B --out ~/models/qwen36
  python3 python/hf_fetch.py --repo MiniMaxAI/MiniMax-H3 --out ~/models/h3 \\
      --include model_index.json --include 'FL2VA/*'
"""
import argparse
import os
import sys


def token():
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(key)
        if v:
            return v
    for path in (
        os.path.expanduser("~/.hf_token"),
        os.path.expanduser("~/.cache/huggingface/token"),
    ):
        if os.path.isfile(path):
            t = open(path, encoding="utf-8").read().strip()
            if t:
                return t
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--include", action="append", default=[],
                    help="allow pattern (repeatable; huggingface allow_patterns)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="ignore pattern (repeatable)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("missing huggingface_hub — pip install huggingface_hub", file=sys.stderr)
        return 2

    dest = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(dest, exist_ok=True)
    tok = token()
    print(f"fetch {args.repo} -> {dest}"
          f"{'  rev=' + args.revision if args.revision else ''}"
          f"{'  token=yes' if tok else '  token=no'}",
          flush=True)
    try:
        snapshot_download(
            repo_id=args.repo,
            local_dir=dest,
            revision=args.revision,
            token=tok or False,
            allow_patterns=args.include or None,
            ignore_patterns=args.exclude or None,
            resume_download=True,
        )
    except TypeError:
        # Older huggingface_hub: no resume_download kw.
        snapshot_download(
            repo_id=args.repo,
            local_dir=dest,
            revision=args.revision,
            token=tok or False,
            allow_patterns=args.include or None,
            ignore_patterns=args.exclude or None,
        )
    print(f"ok {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
