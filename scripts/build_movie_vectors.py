#!/usr/bin/env python3
"""Build local weighted TF-IDF movie vectors for browser-side ranking."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


FIELD_WEIGHTS = {
    "overview": 3.0,
    "genres": 4.0,
    "keywords": 4.0,
    "title": 1.0,
    "tagline": 1.5,
    "directors": 2.5,
    "cast_top_5": 2.0,
    "writers": 1.5,
    "producers": 1.0,
    "production_companies": 1.0,
    "year_bucket": 0.8,
    "runtime_bucket": 0.5,
    "cinematographers": 1.2,
    "composers": 1.0,
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "when",
    "who",
    "with",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"), ensure_ascii=False)
        handle.write("\n")


def slug_token(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def words(value: str | None) -> list[str]:
    if not value:
        return []
    terms = re.findall(r"[a-z0-9]+", value.lower())
    return [term for term in terms if len(term) > 1 and term not in STOPWORDS]


def add_weighted_text(tokens: Counter[str], prefix: str, value: str | None, weight: float) -> None:
    for term in words(value):
        tokens[f"{prefix}:{term}"] += weight


def add_weighted_value(tokens: Counter[str], prefix: str, value: str | None, weight: float) -> None:
    if not value:
        return
    token = slug_token(value)
    if token:
        tokens[f"{prefix}:{token}"] += weight


def runtime_bucket(runtime: int | None) -> str | None:
    if not runtime:
        return None
    if runtime < 90:
        return "short"
    if runtime <= 125:
        return "medium"
    if runtime <= 160:
        return "long"
    return "epic"


def decade_bucket(year: int | None) -> str | None:
    if not year:
        return None
    return f"{year // 10 * 10}s"


def movie_tokens(movie: dict[str, Any]) -> Counter[str]:
    tokens: Counter[str] = Counter()
    crew = movie.get("crew") or {}

    add_weighted_text(tokens, "title", movie.get("title"), FIELD_WEIGHTS["title"])
    add_weighted_text(tokens, "overview", movie.get("overview"), FIELD_WEIGHTS["overview"])
    add_weighted_text(tokens, "tagline", movie.get("tagline"), FIELD_WEIGHTS["tagline"])

    for genre in movie.get("genres") or []:
        add_weighted_value(tokens, "genre", genre, FIELD_WEIGHTS["genres"])
    for keyword in movie.get("keywords") or []:
        add_weighted_value(tokens, "keyword", keyword, FIELD_WEIGHTS["keywords"])
        add_weighted_text(tokens, "keyword_word", keyword, 0.8)

    for director in crew.get("directors") or []:
        add_weighted_value(tokens, "director", director, FIELD_WEIGHTS["directors"])
    for writer in (crew.get("screenplay_writers") or []) + (crew.get("writers") or []) + (crew.get("story_writers") or []):
        add_weighted_value(tokens, "writer", writer, FIELD_WEIGHTS["writers"])
    for cinematographer in crew.get("cinematographers") or []:
        add_weighted_value(tokens, "cinematographer", cinematographer, FIELD_WEIGHTS["cinematographers"])
    for composer in crew.get("composers") or []:
        add_weighted_value(tokens, "composer", composer, FIELD_WEIGHTS["composers"])
    for producer in (crew.get("producers") or []) + (crew.get("executive_producers") or []):
        add_weighted_value(tokens, "producer", producer, FIELD_WEIGHTS["producers"])

    for actor in (movie.get("cast") or [])[:5]:
        add_weighted_value(tokens, "actor", actor.get("name"), FIELD_WEIGHTS["cast_top_5"])
        add_weighted_text(tokens, "character", actor.get("character"), 0.7)

    for company in movie.get("production_companies") or []:
        add_weighted_value(tokens, "company", company.get("name"), FIELD_WEIGHTS["production_companies"])

    add_weighted_value(tokens, "decade", decade_bucket(movie.get("year")), FIELD_WEIGHTS["year_bucket"])
    add_weighted_value(tokens, "runtime", runtime_bucket(movie.get("runtime_minutes")), FIELD_WEIGHTS["runtime_bucket"])
    return tokens


def normalize(vector: dict[str, float]) -> dict[str, float]:
    length = math.sqrt(sum(value * value for value in vector.values()))
    if not length:
        return vector
    return {key: round(value / length, 6) for key, value in vector.items() if value}


def cosine_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def pick_diverse_seed(records: list[dict[str, Any]], count: int = 12) -> list[str]:
    candidates = sorted(
        records,
        key=lambda record: (
            -len(record["metadata"].get("genres") or []),
            -(record["metadata"].get("ratings") or {}).get("popularity", 0),
            record["title"],
        ),
    )
    if not candidates:
        return []

    selected = [candidates[0]]
    remaining = candidates[1:]
    while remaining and len(selected) < count:
        best = max(
            remaining,
            key=lambda record: min(1.0 - cosine_sparse(record["vector"], chosen["vector"]) for chosen in selected),
        )
        selected.append(best)
        remaining.remove(best)
    return [record["id"] for record in selected]


def embedding_text(movie: dict[str, Any]) -> str:
    crew = movie.get("crew") or {}
    top_cast = [
        f"{person.get('name')} as {person.get('character')}" if person.get("character") else person.get("name")
        for person in (movie.get("cast") or [])[:5]
        if person.get("name")
    ]
    companies = [company.get("name") for company in movie.get("production_companies") or [] if company.get("name")]
    producers = (crew.get("producers") or []) + (crew.get("executive_producers") or [])
    writers = (crew.get("screenplay_writers") or []) + (crew.get("writers") or []) + (crew.get("story_writers") or [])
    lines = [
        f"Title: {movie.get('title')}",
        f"Year: {movie.get('year')}",
        f"Genres: {', '.join(movie.get('genres') or [])}",
        f"Overview: {movie.get('overview') or ''}",
        f"Tagline: {movie.get('tagline') or ''}",
        f"Keywords: {', '.join(movie.get('keywords') or [])}",
        f"Cast: {'; '.join(top_cast)}",
        f"Directors: {', '.join(crew.get('directors') or [])}",
        f"Producers: {', '.join(producers)}",
        f"Writers: {', '.join(writers)}",
        f"Cinematographer: {', '.join(crew.get('cinematographers') or [])}",
        f"Composer: {', '.join(crew.get('composers') or [])}",
        f"Production Companies: {', '.join(companies)}",
        f"Runtime: {movie.get('runtime_minutes')} minutes" if movie.get("runtime_minutes") else "",
    ]
    return "\n".join(line for line in lines if line and not line.endswith(": "))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/enriched/movies.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/vectors/movie_vectors.json"))
    args = parser.parse_args()

    movies = load_jsonl(args.input)
    raw_tokens = [movie_tokens(movie) for movie in movies]
    document_frequency: Counter[str] = Counter()
    for token_counts in raw_tokens:
        document_frequency.update(token_counts.keys())

    total = len(movies)
    idf = {
        token: math.log((1 + total) / (1 + frequency)) + 1
        for token, frequency in document_frequency.items()
    }

    records = []
    for movie, token_counts in zip(movies, raw_tokens):
        weighted = {token: count * idf[token] for token, count in token_counts.items()}
        vector = normalize(weighted)
        records.append(
            {
                "id": f"tmdb:{movie['ids']['tmdb']}",
                "title": movie.get("title"),
                "year": movie.get("year"),
                "metadata": movie,
                "tokens": {token: round(value, 4) for token, value in token_counts.items()},
                "vector": vector,
                "embedding_text": embedding_text(movie),
            }
        )

    output = {
        "schema": "weighted-tfidf-v1",
        "field_weights": FIELD_WEIGHTS,
        "count": len(records),
        "seed_order": pick_diverse_seed(records, 12),
        "records": records,
    }
    write_json(args.out, output)
    print(f"wrote {len(records)} vector records to {args.out}")
    print(f"seed_order: {', '.join(output['seed_order'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
