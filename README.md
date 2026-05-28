# Trakt Recommendation

The goal of this project is to take my raw Trakt JSON data, enrich with more information, and then vectorize it, so that I can provide a card like swipe interface on my website where people can swipe left or right until they find the perfect movie. Except, the base is only the movies I have watched.

## Scheduled Trakt sync

The `Sync Trakt movies` GitHub Action runs every Monday at 10:00 UTC and can also be run manually. It refreshes the Trakt OAuth token, checks the authenticated watched-movies feed, enriches only movies that are not already in `data/vectors/movie_vectors.json`, then rebuilds the tracked vector files only when new movies are found.

Repository secrets required:

- `TRAKT_CLIENT_ID`
- `TRAKT_CLIENT_SECRET`
- `TRAKT_REFRESH_TOKEN`
- `TMDB_READ_ACCESS_TOKEN` or `TMDB_API_KEY`

Optional:

- `GH_PAT_FOR_SECRETS`: a GitHub token that can update repository secrets. If Trakt rotates the refresh token, the workflow uses this to update `TRAKT_REFRESH_TOKEN`; otherwise it emits a warning so the secret can be updated manually.
