#!/usr/bin/env python3
"""Download smaller optimized JPEG poster/backdrop assets and update data paths."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


IMAGE_BASE = "https://image.tmdb.org/t/p"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"), ensure_ascii=False)
        handle.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "swiper-static-build/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        path.write_bytes(response.read())


def slug_dir(movie: dict[str, Any]) -> Path:
    poster = movie.get("assets", {}).get("vertical_poster") or ""
    if poster:
        return Path(poster).parent
    tmdb_id = movie["ids"]["tmdb"]
    slug = movie["ids"].get("slug") or str(tmdb_id)
    return Path("data/assets/movies") / f"{tmdb_id}-{slug}"


def update_asset_paths(movie: dict[str, Any], poster_size: str, backdrop_size: str) -> tuple[int, int]:
    assets = movie.get("assets") or {}
    directory = slug_dir(movie)
    poster_bytes = 0
    backdrop_bytes = 0

    poster_path = assets.get("tmdb_poster_path")
    if poster_path:
      poster_file = directory / f"poster-{poster_size}.jpg"
      download(f"{IMAGE_BASE}/{poster_size}{poster_path}", poster_file)
      assets["vertical_poster"] = str(poster_file)
      poster_bytes = poster_file.stat().st_size

    backdrop_path = assets.get("tmdb_backdrop_path")
    if backdrop_path:
      backdrop_file = directory / f"backdrop-{backdrop_size}.jpg"
      download(f"{IMAGE_BASE}/{backdrop_size}{backdrop_path}", backdrop_file)
      assets["horizontal_backdrop"] = str(backdrop_file)
      backdrop_bytes = backdrop_file.stat().st_size

    return poster_bytes, backdrop_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, default=Path("data/vectors/movie_vectors.json"))
    parser.add_argument("--enriched", type=Path, default=Path("data/enriched/movies.jsonl"))
    parser.add_argument("--asset-dir", type=Path, default=Path("data/assets/movies"))
    parser.add_argument("--poster-size", default="w500")
    parser.add_argument("--backdrop-size", default="w780")
    parser.add_argument("--delete-webp", action="store_true")
    args = parser.parse_args()

    payload = load_json(args.vectors)
    total_bytes = 0
    for record in payload.get("records", []):
        poster_bytes, backdrop_bytes = update_asset_paths(record["metadata"], args.poster_size, args.backdrop_size)
        total_bytes += poster_bytes + backdrop_bytes
    write_json(args.vectors, payload)

    enriched = load_jsonl(args.enriched)
    for movie in enriched:
        update_asset_paths(movie, args.poster_size, args.backdrop_size)
    write_jsonl(args.enriched, enriched)

    if args.delete_webp:
        for path in args.asset_dir.rglob("*.webp"):
            path.unlink()

    print(f"wrote optimized JPEG assets: {total_bytes} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
