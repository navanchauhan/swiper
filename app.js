const VECTOR_URL = "data/vectors/movie_vectors.json";
const EMBEDDING_URL = "data/vectors/movie_embeddings_minilm.json";
const STORAGE_KEY = "movie-swipe-vector-state-v1";
const RESULTS_KEY = "movie-swipe-results-v1";
const LOCAL_SERVER_URL = "http://127.0.0.1:8787/";
const SEED_COUNT = 12;

const state = {
  records: [],
  byId: new Map(),
  seedOrder: [],
  sessionSeedOrder: [],
  embeddingModel: null,
  currentId: null,
  nextQueue: [],
  liked: [],
  disliked: [],
  superLiked: [],
  skipped: [],
  shown: [],
  history: [],
  results: [],
  activeDrag: null,
  isAnimating: false,
};

const deck = document.querySelector("#deck");
const deckMeta = document.querySelector("#deckMeta");
const emptyState = document.querySelector("#emptyState");
const rejectButton = document.querySelector("#rejectButton");
const likeButton = document.querySelector("#likeButton");
const superLikeButton = document.querySelector("#superLikeButton");
const skipButton = document.querySelector("#skipButton");
const profileButton = document.querySelector("#profileButton");
const undoButton = document.querySelector("#undoButton");
const resetButton = document.querySelector("#resetButton");
const recommendationPanel = document.querySelector("#recommendationPanel");
const recommendationList = document.querySelector("#recommendationList");
const dialog = document.querySelector("#profileDialog");
const closeProfileButton = document.querySelector("#closeProfileButton");

const profileBackdrop = document.querySelector("#profileBackdrop");
const profilePoster = document.querySelector("#profilePoster");
const profileTitle = document.querySelector("#profileTitle");
const profileSubline = document.querySelector("#profileSubline");
const profileOverview = document.querySelector("#profileOverview");
const profileDetails = document.querySelector("#profileDetails");

function loadSavedState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    for (const key of ["liked", "disliked", "superLiked", "skipped", "shown", "history"]) {
      state[key] = Array.isArray(saved[key]) ? saved[key] : [];
    }
    state.results = Array.isArray(saved.results) ? saved.results : loadSavedResults();
    state.sessionSeedOrder = Array.isArray(saved.sessionSeedOrder) ? saved.sessionSeedOrder.filter((id) => state.byId.has(id)) : [];
    state.currentId = saved.currentId || null;
    state.nextQueue = Array.isArray(saved.nextQueue) ? saved.nextQueue.filter((id) => state.byId.has(id)) : [];
  } catch {
    resetStateOnly();
  }
}

function saveState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      currentId: state.currentId,
      nextQueue: state.nextQueue,
      liked: state.liked,
      disliked: state.disliked,
      superLiked: state.superLiked,
      skipped: state.skipped,
      shown: state.shown,
      results: state.results,
      sessionSeedOrder: state.sessionSeedOrder,
      history: state.history.slice(-120),
    }),
  );
  localStorage.setItem(RESULTS_KEY, JSON.stringify(state.results));
}

function loadSavedResults() {
  try {
    const results = JSON.parse(localStorage.getItem(RESULTS_KEY) || "[]");
    return Array.isArray(results) ? results : [];
  } catch {
    return [];
  }
}

function resetStateOnly() {
  state.currentId = null;
  state.nextQueue = [];
  state.liked = [];
  state.disliked = [];
  state.superLiked = [];
  state.skipped = [];
  state.shown = [];
  state.history = [];
  state.results = [];
  state.sessionSeedOrder = [];
}

function shuffle(items) {
  const shuffled = [...items];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
}

function ensureSessionSeedOrder() {
  if (!state.sessionSeedOrder.length) {
    const seedSet = new Set(state.seedOrder);
    const allIds = state.records.map((record) => record.id);
    const extraSeeds = shuffle(allIds.filter((id) => !seedSet.has(id))).slice(0, 8);
    state.sessionSeedOrder = shuffle([...state.seedOrder, ...extraSeeds]);
  }
}

function list(items, fallback = "Unknown") {
  const clean = (items || []).filter(Boolean);
  return clean.length ? clean.join(", ") : fallback;
}

function crewList(movie, keys) {
  const crew = movie.crew || {};
  return keys.flatMap((key) => crew[key] || []);
}

function topCast(movie, count = 5) {
  return (movie.cast || [])
    .slice(0, count)
    .map((person) => (person.character ? `${person.name} as ${person.character}` : person.name));
}

function assetUrl(path) {
  return path || "";
}

function currentRecord() {
  return state.byId.get(state.currentId) || null;
}

function currentMovie() {
  return currentRecord()?.metadata || null;
}

function uniqueIds(ids) {
  return [...new Set(ids.filter((id) => state.byId.has(id)))];
}

function feedbackCount() {
  return state.liked.length + state.disliked.length + state.superLiked.length;
}

function decidedIds() {
  return new Set([...state.liked, ...state.disliked, ...state.superLiked, ...state.skipped, ...state.shown]);
}

function sparseAdd(target, vector, scale) {
  for (const [token, value] of Object.entries(vector || {})) {
    target[token] = (target[token] || 0) + value * scale;
  }
}

function sparseNormalize(vector) {
  const length = Math.sqrt(Object.values(vector).reduce((sum, value) => sum + value * value, 0));
  if (!length) return vector;
  for (const token of Object.keys(vector)) {
    vector[token] /= length;
  }
  return vector;
}

function denseAdd(target, vector, scale) {
  if (!vector) return;
  for (let index = 0; index < vector.length; index += 1) {
    target[index] = (target[index] || 0) + vector[index] * scale;
  }
}

function denseNormalize(vector) {
  const length = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (!length) return vector;
  return vector.map((value) => value / length);
}

function cosine(left, right) {
  if (!left || !right) return 0;
  const [small, large] = Object.keys(left).length < Object.keys(right).length ? [left, right] : [right, left];
  return Object.entries(small).reduce((sum, [token, value]) => sum + value * (large[token] || 0), 0);
}

function cosineDense(left, right) {
  if (!left || !right || !left.length || !right.length) return 0;
  let score = 0;
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    score += left[index] * right[index];
  }
  return score;
}

function averageVector(ids) {
  const vector = {};
  const validIds = uniqueIds(ids);
  if (!validIds.length) return vector;
  for (const id of validIds) sparseAdd(vector, state.byId.get(id).vector, 1 / validIds.length);
  return vector;
}

function averageEmbedding(ids) {
  const validIds = uniqueIds(ids);
  const first = validIds.map((id) => state.byId.get(id)?.embedding).find(Boolean);
  if (!first) return [];
  const vector = new Array(first.length).fill(0);
  for (const id of validIds) denseAdd(vector, state.byId.get(id)?.embedding, 1 / validIds.length);
  return vector;
}

function tasteVector() {
  const taste = {};
  sparseAdd(taste, averageVector(state.liked), 1.0);
  sparseAdd(taste, averageVector(state.disliked), -0.7);
  sparseAdd(taste, averageVector(state.superLiked), 1.5);
  return sparseNormalize(taste);
}

function tasteEmbedding() {
  const positives = averageEmbedding(state.liked);
  const negatives = averageEmbedding(state.disliked);
  const strong = averageEmbedding(state.superLiked);
  const dimension = positives.length || negatives.length || strong.length;
  if (!dimension) return [];
  const taste = new Array(dimension).fill(0);
  denseAdd(taste, positives, 1.0);
  denseAdd(taste, negatives, -0.7);
  denseAdd(taste, strong, 1.5);
  return denseNormalize(taste);
}

function maxSimilarityToShown(record) {
  const recent = state.shown.slice(-16).map((id) => state.byId.get(id)).filter(Boolean);
  if (!recent.length) return 0;
  return Math.max(...recent.map((shown) => cosine(record.vector, shown.vector)));
}

function unseenRecords(includeCurrent = false, extraBlocked = new Set()) {
  const blocked = decidedIds();
  if (!includeCurrent && state.currentId) blocked.add(state.currentId);
  for (const id of state.nextQueue || []) blocked.add(id);
  for (const id of extraBlocked) blocked.add(id);
  return state.records.filter((record) => !blocked.has(record.id));
}

function candidateScore(record, taste) {
  const semanticTaste = taste.embedding;
  const tfidfTaste = taste.tfidf;
  const semantic = semanticTaste.length ? cosineDense(record.embedding, semanticTaste) : 0;
  const tfidf = Object.keys(tfidfTaste).length ? cosine(record.vector, tfidfTaste) : 0;
  const relevance = semanticTaste.length ? 0.55 * semantic + 0.35 * tfidf : tfidf;
  const diversityBonus = 0.12 * (1 - maxSimilarityToShown(record));
  const uncertaintyBonus = feedbackCount() >= 4 ? 0.08 * (1 - Math.abs(relevance)) : 0;
  const popularity = Math.min((record.metadata.ratings?.popularity || 0) / 100, 0.05);
  return relevance + diversityBonus + uncertaintyBonus + popularity;
}

function rankedCandidates(limit = 20, extraBlocked = new Set()) {
  const taste = {
    tfidf: tasteVector(),
    embedding: tasteEmbedding(),
  };
  return unseenRecords(false, extraBlocked)
    .map((record) => ({
      record,
      score: candidateScore(record, taste),
      relevance: taste.embedding.length ? cosineDense(record.embedding, taste.embedding) : cosine(record.vector, taste.tfidf),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

function nextSeedId(extraBlocked = new Set()) {
  ensureSessionSeedOrder();
  const blocked = decidedIds();
  if (state.currentId) blocked.add(state.currentId);
  for (const id of state.nextQueue || []) blocked.add(id);
  for (const id of extraBlocked) blocked.add(id);
  return state.sessionSeedOrder.find((id) => !blocked.has(id)) || null;
}

function chooseNextId(excludeIds = new Set()) {
  if (feedbackCount() < SEED_COUNT) {
    const seed = nextSeedId(excludeIds);
    if (seed) return seed;
  }
  const ranked = rankedCandidates(20, excludeIds);
  if (!ranked.length) return null;

  const step = state.shown.length + feedbackCount();
  if (step % 10 === 0 && ranked[6]) return ranked[6].record.id;
  if (step % 5 === 0 && ranked[3]) return ranked[3].record.id;
  return ranked[0].record.id;
}

function peekNextRecords(count = 2) {
  ensureNextQueue(count);
  return state.nextQueue.slice(0, count).map((id) => state.byId.get(id)).filter(Boolean);
}

function ensureNextQueue(count = 2) {
  state.nextQueue = (state.nextQueue || []).filter((id) => state.byId.has(id) && id !== state.currentId && !decidedIds().has(id));
  while (state.nextQueue.length < count) {
    const nextId = chooseNextId(new Set(state.nextQueue));
    if (!nextId || state.nextQueue.includes(nextId) || nextId === state.currentId) break;
    state.nextQueue.push(nextId);
  }
}

function tokenLabel(token) {
  const [kind, rawValue] = token.split(":");
  const value = (rawValue || token).replaceAll("_", " ");
  const labels = {
    actor: value,
    character: value,
    cinematographer: value,
    company: value,
    composer: value,
    decade: `${value} movies`,
    director: value,
    genre: `${value} movies`,
    keyword: value,
    keyword_word: value,
    producer: value,
    runtime: `${value} runtime`,
    writer: value,
  };
  return labels[kind] || value;
}

function explanationFor(record, max = 3) {
  const positives = uniqueIds([...state.superLiked, ...state.liked]);
  if (!positives.length) return ["broad seed pick"];
  const positiveTokens = {};
  for (const id of positives) {
    const likedRecord = state.byId.get(id);
    if (!likedRecord) continue;
    for (const [token, value] of Object.entries(likedRecord.tokens || {})) {
      positiveTokens[token] = (positiveTokens[token] || 0) + value;
    }
  }
  return Object.entries(record.tokens || {})
    .filter(([token]) => positiveTokens[token])
    .map(([token, value]) => ({ token, score: value * positiveTokens[token] }))
    .sort((a, b) => b.score - a.score)
    .slice(0, max)
    .map((item) => tokenLabel(item.token));
}

function updateMeta() {
  const unseen = unseenRecords(true).length;
  deckMeta.innerHTML = `${unseen} remaining &nbsp;·&nbsp; <span style="color:var(--green)">♥ ${state.liked.length}</span> &nbsp;·&nbsp; <span style="color:var(--gold)">★ ${state.superLiked.length}</span> &nbsp;·&nbsp; <span style="color:var(--red)">✕ ${state.disliked.length}</span>`;
  emptyState.hidden = Boolean(state.currentId);
  recommendationPanel.hidden = feedbackCount() < 10;
}

function makeCard(record, offset) {
  const movie = record.metadata;
  const card = document.createElement("article");
  card.className = "movie-card";
  card.dataset.tmdbId = movie.ids.tmdb;
  card.style.zIndex = String(10 - offset);
  const reason = explanationFor(record, 2).join(" · ");
  card.innerHTML = `
    <img src="${assetUrl(movie.assets.vertical_poster)}" alt="${movie.title} poster" draggable="false" />
    <div class="decision-badge like">Like</div>
    <div class="decision-badge nope">Nope</div>
    <div class="decision-badge super">Strong</div>
    <div class="card-scrim">
      <h2 class="card-title">${movie.title}</h2>
      <div class="card-meta">
        <span>${movie.year || "Unknown year"}</span>
        <span>${movie.runtime_minutes ? `${movie.runtime_minutes} min` : ""}</span>
        <span>${list((movie.genres || []).slice(0, 2), "")}</span>
      </div>
      <p class="card-overview">${movie.overview || ""}</p>
      <p class="card-reason">${reason ? `Why this: ${reason}` : ""}</p>
    </div>
  `;
  if (offset === 0) wireDrag(card);
  card.addEventListener("click", (event) => {
    if (state.activeDrag?.moved) return;
    if (event.target.closest("button")) return;
    openProfile(movie, record);
  });
  return card;
}

function renderRecommendations() {
  const ranked = rankedCandidates(5);
  const showScore = feedbackCount() >= SEED_COUNT;
  recommendationList.innerHTML = ranked
    .map(({ record, relevance }, index) => {
      const movie = record.metadata;
      const reasons = explanationFor(record, 2);
      const matchPct = showScore ? Math.round(Math.max(0, Math.min(1, relevance)) * 100) : null;
      const scoreHtml = matchPct !== null
        ? `<small class="rec-score">${matchPct}% match</small>`
        : `<small>${reasons.join(" · ")}</small>`;
      return `
        <button class="recommendation-item" type="button" data-id="${record.id}">
          <span>${index + 1}</span>
          <img src="${assetUrl(movie.assets.vertical_poster)}" alt="" />
          <strong>${movie.title}</strong>
          ${scoreHtml}
        </button>
      `;
    })
    .join("");
  recommendationList.querySelectorAll(".recommendation-item").forEach((button) => {
    button.addEventListener("click", () => {
      const record = state.byId.get(button.dataset.id);
      if (record) openProfile(record.metadata, record);
    });
  });
}

function renderDeck() {
  deck.innerHTML = "";
  const top = currentRecord();
  if (!top) {
    updateMeta();
    renderRecommendations();
    deck.classList.remove("is-swiping");
    state.isAnimating = false;
    return;
  }
  ensureNextQueue(2);
  const cards = [top, ...peekNextRecords(2)];
  cards.reverse().forEach((record, reverseOffset, allCards) => {
    const offset = allCards.length - reverseOffset - 1;
    deck.appendChild(makeCard(record, offset));
  });
  updateMeta();
  renderRecommendations();
  requestAnimationFrame(() => {
    deck.classList.remove("is-swiping");
    state.isAnimating = false;
  });
}

function recordDecision(decision) {
  const record = currentRecord();
  if (!record) return;
  const previous = snapshotState();
  state.history.push(previous);
  if (!state.shown.includes(record.id)) state.shown.push(record.id);
  if (decision === "like") state.liked.push(record.id);
  if (decision === "reject") state.disliked.push(record.id);
  if (decision === "super") state.superLiked.push(record.id);
  if (decision === "skip") state.skipped.push(record.id);
  state.results.push({
    id: record.id,
    tmdbId: record.metadata.ids?.tmdb,
    title: record.title,
    year: record.year,
    decision,
    decidedAt: new Date().toISOString(),
  });
  ensureNextQueue(2);
  state.currentId = state.nextQueue.shift() || chooseNextId();
  ensureNextQueue(2);
  saveState();
  renderDeck();
}

function snapshotState() {
  return {
    currentId: state.currentId,
    nextQueue: [...state.nextQueue],
    liked: [...state.liked],
    disliked: [...state.disliked],
    superLiked: [...state.superLiked],
    skipped: [...state.skipped],
    shown: [...state.shown],
    results: [...state.results],
  };
}

function restoreSnapshot(snapshot) {
  Object.assign(state, {
    currentId: snapshot.currentId,
    nextQueue: [...(snapshot.nextQueue || [])],
    liked: [...snapshot.liked],
    disliked: [...snapshot.disliked],
    superLiked: [...snapshot.superLiked],
    skipped: [...snapshot.skipped],
    shown: [...snapshot.shown],
    results: [...snapshot.results],
  });
}

function animateSwipe(card, decision) {
  if (state.isAnimating) return;
  state.isAnimating = true;
  ensureNextQueue(2);
  const direction = decision === "reject" ? -1 : decision === "like" ? 1 : 0;
  const y = decision === "super" ? -120 : decision === "skip" ? 120 : -24;
  const x = direction * 120;
  card.style.transition = "transform 260ms ease, opacity 260ms ease";
  card.style.transform = `translate(${x}vw, ${y}px) rotate(${direction * 22}deg)`;
  card.style.opacity = "0";
  window.setTimeout(() => recordDecision(decision), 180);
}

function updateDragVisual(card, dx, dy) {
  const rotation = dx / 18;
  const likeOpacity = Math.min(Math.max(dx, 0) / 120, 1);
  const nopeOpacity = Math.min(Math.max(-dx, 0) / 120, 1);
  const superOpacity = Math.min(Math.max(-dy, 0) / 120, 1);
  card.style.transform = `translate(${dx}px, ${dy}px) rotate(${rotation}deg)`;
  card.querySelector(".decision-badge.like").style.opacity = likeOpacity;
  card.querySelector(".decision-badge.nope").style.opacity = nopeOpacity;
  card.querySelector(".decision-badge.super").style.opacity = superOpacity;
}

function wireDrag(card) {
  card.addEventListener("pointerdown", (event) => {
    if (state.isAnimating) return;
    card.setPointerCapture(event.pointerId);
    state.activeDrag = {
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
    };
    card.style.transition = "none";
  });

  card.addEventListener("pointermove", (event) => {
    if (!state.activeDrag) return;
    const dx = event.clientX - state.activeDrag.startX;
    const dy = event.clientY - state.activeDrag.startY;
    state.activeDrag.moved = state.activeDrag.moved || Math.abs(dx) > 6 || Math.abs(dy) > 6;
    updateDragVisual(card, dx, dy);
  });

  card.addEventListener("pointerup", (event) => {
    if (!state.activeDrag) return;
    const dx = event.clientX - state.activeDrag.startX;
    const dy = event.clientY - state.activeDrag.startY;
    const decision = dx > 110 ? "like" : dx < -110 ? "reject" : dy < -120 ? "super" : dy > 120 ? "skip" : null;
    state.activeDrag = null;
    if (decision) {
      animateSwipe(card, decision);
    } else {
      card.style.transition = "transform 220ms ease";
      card.style.transform = "";
      card.querySelector(".decision-badge.like").style.opacity = 0;
      card.querySelector(".decision-badge.nope").style.opacity = 0;
      card.querySelector(".decision-badge.super").style.opacity = 0;
    }
  });
}

function detailBlock(title, value) {
  if (!value) return "";
  return `
    <section class="detail-block">
      <h3>${title}</h3>
      <p>${value}</p>
    </section>
  `;
}

function openProfile(movie = currentMovie(), record = currentRecord()) {
  if (!movie) return;
  const writers = crewList(movie, ["screenplay_writers", "writers", "story_writers"]);
  const producers = crewList(movie, ["producers", "executive_producers"]);
  const reasons = record ? explanationFor(record, 5) : [];
  profileBackdrop.src = assetUrl(movie.assets.horizontal_backdrop);
  profilePoster.src = assetUrl(movie.assets.vertical_poster);
  profilePoster.alt = `${movie.title} poster`;
  profileTitle.textContent = `${movie.title}`;
  profileSubline.textContent = [
    movie.year,
    movie.certification_us,
    movie.runtime_minutes ? `${movie.runtime_minutes} min` : null,
    list(movie.genres, ""),
  ]
    .filter(Boolean)
    .join(" · ");
  profileOverview.textContent = movie.overview || "No overview available.";
  profileDetails.innerHTML = [
    detailBlock("Recommended Because", reasons.length ? reasons.join(", ") : "Cold-start diversity pick"),
    detailBlock("Director", list(movie.crew?.directors)),
    detailBlock("Producer", list(producers)),
    detailBlock("Screenplay / Writing", list(writers)),
    detailBlock("Cinematography", list(movie.crew?.cinematographers)),
    detailBlock("Composer", list(movie.crew?.composers)),
    detailBlock("Cast", list(topCast(movie, 12))),
    detailBlock("Keywords", list((movie.keywords || []).slice(0, 18))),
    detailBlock("Production", list((movie.production_companies || []).map((company) => company.name))),
    detailBlock("Languages", list(movie.spoken_languages)),
  ].join("");
  dialog.showModal();
}

function undo() {
  const last = state.history.pop();
  if (!last) return;
  restoreSnapshot(last);
  saveState();
  renderDeck();
}

function reset() {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(RESULTS_KEY);
  resetStateOnly();
  ensureSessionSeedOrder();
  state.currentId = chooseNextId();
  ensureNextQueue(2);
  saveState();
  renderDeck();
}

rejectButton.addEventListener("click", () => {
  if (state.isAnimating) return;
  const card = deck.querySelector(".movie-card:last-child");
  if (card) animateSwipe(card, "reject");
});

likeButton.addEventListener("click", () => {
  if (state.isAnimating) return;
  const card = deck.querySelector(".movie-card:last-child");
  if (card) animateSwipe(card, "like");
});

superLikeButton.addEventListener("click", () => {
  if (state.isAnimating) return;
  const card = deck.querySelector(".movie-card:last-child");
  if (card) animateSwipe(card, "super");
});

skipButton.addEventListener("click", () => {
  if (state.isAnimating) return;
  const card = deck.querySelector(".movie-card:last-child");
  if (card) animateSwipe(card, "skip");
});

profileButton.addEventListener("click", () => openProfile());
undoButton.addEventListener("click", undo);
resetButton.addEventListener("click", reset);
closeProfileButton.addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});

window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    rejectButton.click();
  }
  if (event.key === "ArrowRight") {
    event.preventDefault();
    likeButton.click();
  }
  if (event.key === "ArrowUp") {
    event.preventDefault();
    superLikeButton.click();
  }
  if (event.key === "ArrowDown") {
    event.preventDefault();
    skipButton.click();
  }
  if (event.key === "Enter") {
    event.preventDefault();
    openProfile();
  }
});

async function init() {
  if (window.location.protocol === "file:") {
    deckMeta.textContent = "Opening local server...";
    window.location.href = LOCAL_SERVER_URL;
    return;
  }

  try {
    const response = await fetch(VECTOR_URL);
    if (!response.ok) throw new Error(`Failed to load ${VECTOR_URL}`);
    const payload = await response.json();
    const embeddingResponse = await fetch(EMBEDDING_URL);
    if (!embeddingResponse.ok) throw new Error(`Failed to load ${EMBEDDING_URL}`);
    const embeddingPayload = await embeddingResponse.json();
    const embeddings = new Map((embeddingPayload.records || []).map((record) => [record.id, record.embedding]));
    state.records = payload.records || [];
    state.records.forEach((record) => {
      record.embedding = embeddings.get(record.id) || [];
    });
    state.byId = new Map(state.records.map((record) => [record.id, record]));
    state.seedOrder = payload.seed_order || [];
    state.embeddingModel = embeddingPayload.model || null;
    loadSavedState();
    ensureSessionSeedOrder();
    if (!state.currentId || !state.byId.has(state.currentId)) state.currentId = chooseNextId();
    ensureNextQueue(2);
    renderDeck();
  } catch (error) {
    deckMeta.textContent = error.message;
  }
}

init();
