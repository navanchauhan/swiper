#!/usr/bin/env python3
"""Incrementally sync newly watched Trakt movies into the local corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import enrich_movies


TRAKT_API_BASE = "https://api.trakt.tv"
TRAKT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
APP_POSTER_SIZE = "w500"
APP_BACKDROP_SIZE = "w780"


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def write_dotenv(path: Path, updates: dict[str, str]) -> None:
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    lines: list[str] = []
    for line in existing:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            lines.append(line)
    for key, value in updates.items():
        if key not in seen:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def emit_github_output(values: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with Path(output_path).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 30,
) -> Any:
    data = None
    request_headers = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request_headers.setdefault("Accept", "application/json")
    request_headers.setdefault("User-Agent", TRAKT_USER_AGENT)
    req = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(int(exc.headers.get("Retry-After", "2")))
                continue
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"request failed {exc.code} for {url}: {detail}") from exc
        except urllib.error.URLError:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"request failed after retries: {url}")


def refresh_trakt_token(client_id: str, client_secret: str, refresh_token: str, redirect_uri: str) -> dict[str, Any]:
    return request_json(
        f"{TRAKT_API_BASE}/oauth/token",
        method="POST",
        body={
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "refresh_token",
        },
    )


def trakt_headers(client_id: str, access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "trakt-api-version": "2",
        "trakt-api-key": client_id,
        "User-Agent": TRAKT_USER_AGENT,
    }


def fetch_trakt(path: str, client_id: str, access_token: str) -> Any:
    return request_json(f"{TRAKT_API_BASE}{path}", headers=trakt_headers(client_id, access_token), timeout=60)


def existing_movies_from_vectors(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    records = payload.get("records") or []
    return [record["metadata"] for record in records if record.get("metadata")]


def movie_tmdb_id(movie: dict[str, Any]) -> int | None:
    ids = movie.get("ids") or {}
    value = ids.get("tmdb")
    return int(value) if value else None


def source_tmdb_id(source: dict[str, Any]) -> int | None:
    ids = ((source.get("movie") or {}).get("ids") or {})
    value = ids.get("tmdb")
    return int(value) if value else None


def source_trakt_id(source: dict[str, Any]) -> int | None:
    ids = ((source.get("movie") or {}).get("ids") or {})
    value = ids.get("trakt")
    return int(value) if value else None


def watched_sort_key(source: dict[str, Any]) -> str:
    return source.get("last_watched_at") or source.get("last_collected_at") or ""


def download_movie_assets(
    tmdb_movie: dict[str, Any],
    title: str,
    tmdb_id: int,
    asset_dir: Path,
    skip_images: bool,
) -> dict[str, str | None]:
    poster_path = tmdb_movie.get("poster_path")
    backdrop_path = tmdb_movie.get("backdrop_path")
    slug = f"{tmdb_id}-{enrich_movies.slugify(title)}"
    local_poster = None
    local_backdrop = None
    if not skip_images:
        if poster_path:
            poster_file = asset_dir / slug / f"poster-{APP_POSTER_SIZE}{Path(poster_path).suffix or '.jpg'}"
            enrich_movies.download(f"https://image.tmdb.org/t/p/{APP_POSTER_SIZE}{poster_path}", poster_file)
            local_poster = str(poster_file)
        if backdrop_path:
            backdrop_file = asset_dir / slug / f"backdrop-{APP_BACKDROP_SIZE}{Path(backdrop_path).suffix or '.jpg'}"
            enrich_movies.download(f"https://image.tmdb.org/t/p/{APP_BACKDROP_SIZE}{backdrop_path}", backdrop_file)
            local_backdrop = str(backdrop_file)
    return {
        "vertical_poster": local_poster,
        "horizontal_backdrop": local_backdrop,
        "tmdb_poster_path": poster_path,
        "tmdb_backdrop_path": backdrop_path,
    }


def enrich_new_movie(
    source: dict[str, Any],
    *,
    raw_dir: Path,
    asset_dir: Path,
    tmdb_token: str | None,
    tmdb_api_key: str | None,
    skip_images: bool,
) -> dict[str, Any]:
    tmdb_id = source_tmdb_id(source)
    if not tmdb_id:
        raise RuntimeError(f"watched item has no TMDB id: {source}")
    title = (source.get("movie") or {}).get("title") or f"tmdb:{tmdb_id}"
    raw_path = raw_dir / f"{tmdb_id}.json"
    if raw_path.exists():
        tmdb_movie = load_json(raw_path)
    else:
        path = f"/movie/{tmdb_id}?append_to_response={enrich_movies.APPEND_TO_RESPONSE}&include_image_language=en,null"
        tmdb_movie = enrich_movies.tmdb_request(path, tmdb_token, tmdb_api_key)
        write_json(raw_path, tmdb_movie)
        time.sleep(0.05)
    asset_paths = download_movie_assets(tmdb_movie, title, tmdb_id, asset_dir, skip_images)
    return enrich_movies.normalize_movie(source, tmdb_movie, asset_paths)


def ordered_movies(
    existing: list[dict[str, Any]],
    new_movies: list[dict[str, Any]],
    watched: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_tmdb: dict[int, dict[str, Any]] = {}
    for movie in existing + new_movies:
        tmdb_id = movie_tmdb_id(movie)
        if tmdb_id is not None:
            by_tmdb[tmdb_id] = movie

    ordered: list[dict[str, Any]] = []
    used: set[int] = set()
    for source in sorted(watched, key=watched_sort_key, reverse=True):
        tmdb_id = source_tmdb_id(source)
        if tmdb_id is None or tmdb_id in used or tmdb_id not in by_tmdb:
            continue
        ordered.append(by_tmdb[tmdb_id])
        used.add(tmdb_id)

    for movie in existing + new_movies:
        tmdb_id = movie_tmdb_id(movie)
        if tmdb_id is not None and tmdb_id not in used:
            ordered.append(movie)
            used.add(tmdb_id)
    return ordered


def persist_rotated_refresh_token(new_refresh_token: str, old_refresh_token: str) -> str:
    if not new_refresh_token or new_refresh_token == old_refresh_token:
        return ""
    if not os.environ.get("GITHUB_OUTPUT"):
        return ""
    temp_dir = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    token_path = temp_dir / "trakt-refresh-token"
    token_path.write_text(new_refresh_token, encoding="utf-8")
    token_path.chmod(0o600)
    return str(token_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--write-env", action="store_true")
    parser.add_argument("--vectors", type=Path, default=Path("data/vectors/movie_vectors.json"))
    parser.add_argument("--watched-out", type=Path, default=Path("references/trakt-export-prudentwish/watched-movies.json"))
    parser.add_argument(
        "--last-activities-out",
        type=Path,
        default=Path("references/trakt-export-prudentwish/user-last-activities.json"),
    )
    parser.add_argument("--enriched-out", type=Path, default=Path("data/enriched/movies.jsonl"))
    parser.add_argument("--docs-out", type=Path, default=Path("data/embeddings/movie_documents.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/tmdb/movies"))
    parser.add_argument("--asset-dir", type=Path, default=Path("data/assets/movies"))
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    load_dotenv(args.env)
    client_id = os.environ.get("TRAKT_CLIENT_ID")
    client_secret = os.environ.get("TRAKT_CLIENT_SECRET")
    old_refresh_token = os.environ.get("TRAKT_REFRESH_TOKEN")
    redirect_uri = os.environ.get("TRAKT_REDIRECT_URI") or "urn:ietf:wg:oauth:2.0:oob"
    tmdb_token = os.environ.get("TMDB_READ_ACCESS_TOKEN") or os.environ.get("TMDB_BEARER_TOKEN")
    tmdb_api_key = os.environ.get("TMDB_API_KEY")

    missing = [
        name
        for name, value in {
            "TRAKT_CLIENT_ID": client_id,
            "TRAKT_CLIENT_SECRET": client_secret,
            "TRAKT_REFRESH_TOKEN": old_refresh_token,
        }.items()
        if not value
    ]
    if missing:
        print(f"Missing required environment values: {', '.join(missing)}", file=sys.stderr)
        return 2
    if not tmdb_token and not tmdb_api_key:
        print("Missing TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY", file=sys.stderr)
        return 2

    token_data = refresh_trakt_token(client_id, client_secret, old_refresh_token, redirect_uri)
    access_token = token_data["access_token"]
    new_refresh_token = token_data.get("refresh_token") or old_refresh_token
    created_at = str(int(token_data.get("created_at") or time.time()))
    expires_in = str(int(token_data.get("expires_in") or 0))
    expires_at = str(int(created_at) + int(expires_in)) if expires_in != "0" else ""

    if args.write_env:
        updates = {
            "TRAKT_ACCESS_TOKEN": access_token,
            "TRAKT_REFRESH_TOKEN": new_refresh_token,
            "TRAKT_TOKEN_TYPE": str(token_data.get("token_type") or "bearer"),
            "TRAKT_TOKEN_SCOPE": str(token_data.get("scope") or ""),
            "TRAKT_TOKEN_CREATED_AT": created_at,
            "TRAKT_TOKEN_EXPIRES_IN": expires_in,
        }
        if expires_at:
            updates["TRAKT_TOKEN_EXPIRES_AT"] = expires_at
        write_dotenv(args.env, updates)

    rotated_path = persist_rotated_refresh_token(new_refresh_token, old_refresh_token)
    existing = existing_movies_from_vectors(args.vectors)
    existing_tmdb_ids = {movie_tmdb_id(movie) for movie in existing if movie_tmdb_id(movie) is not None}
    existing_trakt_ids = {
        int(((movie.get("ids") or {}).get("trakt")))
        for movie in existing
        if (movie.get("ids") or {}).get("trakt")
    }

    watched = fetch_trakt("/sync/watched/movies?extended=full", client_id, access_token)
    watched = sorted(watched, key=watched_sort_key, reverse=True)
    last_activities = fetch_trakt("/sync/last_activities", client_id, access_token)
    write_json(args.watched_out, watched)
    write_json(args.last_activities_out, last_activities)

    new_sources = [
        source
        for source in watched
        if (source_tmdb_id(source) not in existing_tmdb_ids)
        and (source_trakt_id(source) not in existing_trakt_ids)
    ]

    if not new_sources:
        emit_github_output(
            {
                "changed": "false",
                "new_count": "0",
                "refresh_token_rotated": "true" if rotated_path else "false",
                "refresh_token_file": rotated_path,
            }
        )
        print(f"watched movies from Trakt: {len(watched)}")
        print("no new movies to enrich")
        if rotated_path:
            print("Trakt returned a rotated refresh token; persist it outside the repo before the next run.")
        return 0

    new_movies: list[dict[str, Any]] = []
    for index, source in enumerate(new_sources, start=1):
        source_movie = source.get("movie") or {}
        title = source_movie.get("title") or f"tmdb:{source_tmdb_id(source)}"
        year = source_movie.get("year")
        print(f"enriching {index}/{len(new_sources)}: {title} ({year})", flush=True)
        new_movies.append(
            enrich_new_movie(
                source,
                raw_dir=args.raw_dir,
                asset_dir=args.asset_dir,
                tmdb_token=tmdb_token,
                tmdb_api_key=tmdb_api_key,
                skip_images=args.skip_images,
            )
        )

    movies = ordered_movies(existing, new_movies, watched)
    docs = [enrich_movies.movie_document(movie) for movie in movies]
    write_jsonl(args.enriched_out, movies)
    write_jsonl(args.docs_out, docs)
    emit_github_output(
        {
            "changed": "true",
            "new_count": str(len(new_movies)),
            "refresh_token_rotated": "true" if rotated_path else "false",
            "refresh_token_file": rotated_path,
        }
    )

    print(f"watched movies from Trakt: {len(watched)}")
    print(f"existing movie records: {len(existing)}")
    print(f"new movies enriched: {len(new_movies)}")
    for movie in new_movies:
        print(f"- {movie.get('title')} ({movie.get('year')})")
    print(f"wrote {len(movies)} movies to {args.enriched_out}")
    print(f"wrote {len(docs)} embedding documents to {args.docs_out}")
    if rotated_path:
        print("Trakt returned a rotated refresh token; persist it outside the repo before the next run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
