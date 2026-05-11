#!/usr/bin/env python3
"""Enrich watched Trakt movies with TMDB metadata and local image assets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


TMDB_API_BASE = "https://api.themoviedb.org/3"
APPEND_TO_RESPONSE = "credits,keywords,external_ids,images,release_dates,videos"
IMAGE_SIZE_POSTER = "w780"
IMAGE_SIZE_BACKDROP = "w1280"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def tmdb_request(path: str, token: str | None, api_key: str | None) -> Any:
    params = {}
    url_path = path
    if "?" in path:
        url_path, query = path.split("?", 1)
        params.update(dict(urllib.parse.parse_qsl(query)))
    if not token and api_key:
        params["api_key"] = api_key
    url = f"{TMDB_API_BASE}{url_path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = int(exc.headers.get("Retry-After", "2"))
                time.sleep(retry_after)
                continue
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"TMDB request failed {exc.code} for {path}: {detail}") from exc
        except urllib.error.URLError:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"TMDB request failed after retries: {path}")


def download(url: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "trakt-recommendation-enricher/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                path.write_bytes(response.read())
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(int(exc.headers.get("Retry-After", "2")))
                continue
            raise
        except urllib.error.URLError:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    return False


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "movie"


def people_by_job(crew: list[dict[str, Any]], jobs: set[str]) -> list[str]:
    seen = set()
    names = []
    for person in crew:
        if person.get("job") not in jobs:
            continue
        name = person.get("name")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def keyword_names(tmdb_movie: dict[str, Any]) -> list[str]:
    keywords = tmdb_movie.get("keywords") or {}
    return [item["name"] for item in keywords.get("keywords", []) if item.get("name")]


def certification(tmdb_movie: dict[str, Any], country: str = "US") -> str | None:
    release_dates = tmdb_movie.get("release_dates") or {}
    for result in release_dates.get("results", []):
        if result.get("iso_3166_1") != country:
            continue
        for release in result.get("release_dates", []):
            cert = release.get("certification")
            if cert:
                return cert
    return None


def best_trailer(tmdb_movie: dict[str, Any]) -> dict[str, str] | None:
    videos = (tmdb_movie.get("videos") or {}).get("results", [])
    youtube = [
        video
        for video in videos
        if video.get("site") == "YouTube" and video.get("key") and video.get("type") == "Trailer"
    ]
    if not youtube:
        return None
    official = [video for video in youtube if video.get("official")]
    video = (official or youtube)[0]
    return {
        "name": video.get("name") or "",
        "url": f"https://www.youtube.com/watch?v={video['key']}",
        "key": video["key"],
    }


def normalize_movie(source: dict[str, Any], tmdb_movie: dict[str, Any], asset_paths: dict[str, str | None]) -> dict[str, Any]:
    source_movie = source["movie"]
    source_ids = source_movie.get("ids", {})
    credits = tmdb_movie.get("credits") or {}
    cast = credits.get("cast") or []
    crew = credits.get("crew") or []

    normalized = {
        "ids": {
            "trakt": source_ids.get("trakt"),
            "tmdb": source_ids.get("tmdb"),
            "imdb": source_ids.get("imdb") or tmdb_movie.get("imdb_id"),
            "plex": source_ids.get("plex"),
            "slug": source_ids.get("slug"),
        },
        "title": tmdb_movie.get("title") or source_movie.get("title"),
        "original_title": tmdb_movie.get("original_title"),
        "year": source_movie.get("year"),
        "release_date": tmdb_movie.get("release_date"),
        "status": tmdb_movie.get("status"),
        "overview": tmdb_movie.get("overview"),
        "tagline": tmdb_movie.get("tagline"),
        "runtime_minutes": tmdb_movie.get("runtime"),
        "certification_us": certification(tmdb_movie, "US"),
        "genres": [genre["name"] for genre in tmdb_movie.get("genres", []) if genre.get("name")],
        "keywords": keyword_names(tmdb_movie),
        "original_language": tmdb_movie.get("original_language"),
        "spoken_languages": [
            language.get("english_name") or language.get("name")
            for language in tmdb_movie.get("spoken_languages", [])
            if language.get("english_name") or language.get("name")
        ],
        "production_countries": [
            country.get("name")
            for country in tmdb_movie.get("production_countries", [])
            if country.get("name")
        ],
        "production_companies": [
            {
                "name": company.get("name"),
                "origin_country": company.get("origin_country") or None,
            }
            for company in tmdb_movie.get("production_companies", [])
            if company.get("name")
        ],
        "cast": [
            {
                "name": person.get("name"),
                "character": person.get("character"),
                "order": person.get("order"),
                "tmdb_person_id": person.get("id"),
            }
            for person in sorted(cast, key=lambda item: item.get("order") if item.get("order") is not None else 9999)[:20]
            if person.get("name")
        ],
        "crew": {
            "directors": people_by_job(crew, {"Director"}),
            "producers": people_by_job(crew, {"Producer"}),
            "executive_producers": people_by_job(crew, {"Executive Producer"}),
            "screenplay_writers": people_by_job(crew, {"Screenplay"}),
            "writers": people_by_job(crew, {"Writer"}),
            "story_writers": people_by_job(crew, {"Story"}),
            "cinematographers": people_by_job(crew, {"Director of Photography"}),
            "editors": people_by_job(crew, {"Editor"}),
            "composers": people_by_job(crew, {"Original Music Composer"}),
        },
        "ratings": {
            "tmdb_vote_average": tmdb_movie.get("vote_average"),
            "tmdb_vote_count": tmdb_movie.get("vote_count"),
            "popularity": tmdb_movie.get("popularity"),
        },
        "assets": asset_paths,
        "links": {
            "tmdb": f"https://www.themoviedb.org/movie/{source_ids.get('tmdb')}" if source_ids.get("tmdb") else None,
            "imdb": f"https://www.imdb.com/title/{source_ids.get('imdb')}/" if source_ids.get("imdb") else None,
            "homepage": tmdb_movie.get("homepage") or None,
            "trailer": best_trailer(tmdb_movie),
        },
    }
    return normalized


def movie_document(movie: dict[str, Any]) -> dict[str, Any]:
    crew = movie.get("crew") or {}
    cast_text = "; ".join(
        f"{person['name']} as {person['character']}" if person.get("character") else person["name"]
        for person in movie.get("cast", [])
        if person.get("name")
    )
    companies = ", ".join(company["name"] for company in movie.get("production_companies", []) if company.get("name"))
    parts = [
        f"Title: {movie.get('title')} ({movie.get('year')})",
        f"Overview: {movie.get('overview') or ''}",
        f"Tagline: {movie.get('tagline') or ''}",
        f"Genres: {', '.join(movie.get('genres') or [])}",
        f"Keywords: {', '.join(movie.get('keywords') or [])}",
        f"Director: {', '.join(crew.get('directors') or [])}",
        f"Producer: {', '.join((crew.get('producers') or []) + (crew.get('executive_producers') or []))}",
        f"Screenplay and writing: {', '.join((crew.get('screenplay_writers') or []) + (crew.get('writers') or []) + (crew.get('story_writers') or []))}",
        f"Cinematography: {', '.join(crew.get('cinematographers') or [])}",
        f"Editing: {', '.join(crew.get('editors') or [])}",
        f"Music: {', '.join(crew.get('composers') or [])}",
        f"Cast: {cast_text}",
        f"Production companies: {companies}",
        f"Countries: {', '.join(movie.get('production_countries') or [])}",
        f"Languages: {', '.join(movie.get('spoken_languages') or [])}",
    ]
    return {
        "id": f"tmdb:{movie['ids']['tmdb']}",
        "tmdb_id": movie["ids"]["tmdb"],
        "title": movie["title"],
        "year": movie["year"],
        "text": "\n".join(part for part in parts if not part.endswith(": ")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("references/trakt-export-prudentwish/watched-movies.json"))
    parser.add_argument("--out", type=Path, default=Path("data/enriched/movies.jsonl"))
    parser.add_argument("--docs-out", type=Path, default=Path("data/embeddings/movie_documents.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/tmdb/movies"))
    parser.add_argument("--asset-dir", type=Path, default=Path("data/assets/movies"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    load_dotenv(args.env)
    token = os.environ.get("TMDB_READ_ACCESS_TOKEN") or os.environ.get("TMDB_BEARER_TOKEN")
    api_key = os.environ.get("TMDB_API_KEY")
    if not token and not api_key:
        print("Missing TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY", file=sys.stderr)
        return 2

    watched = load_json(args.input)
    movies = watched[: args.limit] if args.limit else watched
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)

    enriched_count = 0
    image_count = 0
    with args.out.open("w", encoding="utf-8") as out_handle, args.docs_out.open("w", encoding="utf-8") as docs_handle:
        for index, source in enumerate(movies, start=1):
            tmdb_id = source["movie"]["ids"]["tmdb"]
            title = source["movie"]["title"]
            raw_path = args.raw_dir / f"{tmdb_id}.json"
            if raw_path.exists():
                tmdb_movie = load_json(raw_path)
            else:
                path = f"/movie/{tmdb_id}?append_to_response={APPEND_TO_RESPONSE}&include_image_language=en,null"
                tmdb_movie = tmdb_request(path, token, api_key)
                write_json(raw_path, tmdb_movie)
                time.sleep(0.05)

            poster_path = tmdb_movie.get("poster_path")
            backdrop_path = tmdb_movie.get("backdrop_path")
            slug = f"{tmdb_id}-{slugify(title)}"
            local_poster = None
            local_backdrop = None
            if not args.skip_images:
                if poster_path:
                    poster_file = args.asset_dir / slug / f"poster-{IMAGE_SIZE_POSTER}{Path(poster_path).suffix or '.jpg'}"
                    if download(f"https://image.tmdb.org/t/p/{IMAGE_SIZE_POSTER}{poster_path}", poster_file):
                        image_count += 1
                    local_poster = str(poster_file)
                if backdrop_path:
                    backdrop_file = args.asset_dir / slug / f"backdrop-{IMAGE_SIZE_BACKDROP}{Path(backdrop_path).suffix or '.jpg'}"
                    if download(f"https://image.tmdb.org/t/p/{IMAGE_SIZE_BACKDROP}{backdrop_path}", backdrop_file):
                        image_count += 1
                    local_backdrop = str(backdrop_file)

            asset_paths = {
                "vertical_poster": local_poster,
                "horizontal_backdrop": local_backdrop,
                "tmdb_poster_path": poster_path,
                "tmdb_backdrop_path": backdrop_path,
            }
            normalized = normalize_movie(source, tmdb_movie, asset_paths)
            out_handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
            docs_handle.write(json.dumps(movie_document(normalized), ensure_ascii=False, sort_keys=True) + "\n")
            enriched_count += 1
            if index % 25 == 0 or index == len(movies):
                print(f"enriched {index}/{len(movies)} movies; downloaded {image_count} new images", flush=True)

    print(f"wrote {enriched_count} movies to {args.out}")
    print(f"wrote {enriched_count} embedding documents to {args.docs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
