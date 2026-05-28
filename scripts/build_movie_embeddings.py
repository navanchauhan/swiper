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
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    payload = load_json(args.vectors)
    records = payload["records"]
    existing_by_id: dict[str, list[float]] = {}
    existing_dimension = 0
    if args.reuse_existing and args.out.exists():
        existing = load_json(args.out)
        if existing.get("model") == args.model:
            existing_by_id = {
                record["id"]: record["embedding"]
                for record in existing.get("records", [])
                if record.get("id") and record.get("embedding")
            }
            existing_dimension = int(existing.get("dimension") or 0)

    missing_records = [record for record in records if record["id"] not in existing_by_id]
    new_embeddings = []
    if missing_records:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model)
        new_embeddings = model.encode(
            [record["embedding_text"] for record in missing_records],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    new_by_id = {
        record["id"]: [round(float(value), 6) for value in vector]
        for record, vector in zip(missing_records, new_embeddings)
    }
    embeddings_by_id = {**existing_by_id, **new_by_id}
    ordered_embeddings = [embeddings_by_id[record["id"]] for record in records]
    dimension = len(ordered_embeddings[0]) if ordered_embeddings else existing_dimension

    output = {
        "schema": "sentence-transformers-dense-v1",
        "model": args.model,
        "dimension": dimension,
        "count": len(records),
        "records": [
            {
                "id": record["id"],
                "embedding": vector,
            }
            for record, vector in zip(records, ordered_embeddings)
        ],
    }
    write_json(args.out, output)
    print(f"wrote {len(records)} embeddings to {args.out}")
    print(f"model: {args.model}")
    print(f"dimension: {output['dimension']}")
    if args.reuse_existing:
        print(f"reused existing embeddings: {len(records) - len(missing_records)}")
        print(f"encoded new embeddings: {len(missing_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
