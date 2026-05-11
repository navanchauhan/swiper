#!/usr/bin/env python3
"""Precompute dense sentence-transformer embeddings for enriched movie text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"), ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, default=Path("data/vectors/movie_vectors.json"))
    parser.add_argument("--out", type=Path, default=Path("data/vectors/movie_embeddings_minilm.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    payload = load_json(args.vectors)
    records = payload["records"]
    texts = [record["embedding_text"] for record in records]

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    output = {
        "schema": "sentence-transformers-dense-v1",
        "model": args.model,
        "dimension": len(embeddings[0]) if len(embeddings) else 0,
        "count": len(records),
        "records": [
            {
                "id": record["id"],
                "embedding": [round(float(value), 6) for value in vector],
            }
            for record, vector in zip(records, embeddings)
        ],
    }
    write_json(args.out, output)
    print(f"wrote {len(records)} embeddings to {args.out}")
    print(f"model: {args.model}")
    print(f"dimension: {output['dimension']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
