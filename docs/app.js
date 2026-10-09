const POSTER = "https://image.tmdb.org/t/p/w342";
const MIN_RATINGS = 5;
const GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
  "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
  "Thriller", "War", "Western"];

const $ = (id) => document.getElementById(id);
const store = {
  load() { try { return JSON.parse(localStorage.getItem("ratings")) || {}; } catch { return {}; } },
  save(r) { try { localStorage.setItem("ratings", JSON.stringify(r)); } catch {} },
};
let ratings = store.load(); // movieId -> rating
let offset = 0;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v != null) node.setAttribute(k, v);
  }
  node.append(...children.filter((c) => c != null));
  return node;
}

function poster(movie) {
  return el("img", { class: "poster", src: POSTER + movie.poster, alt: `${movie.title} poster`, loading: "lazy" });
}

function starRow(movie, card, onRate) {
  const row = el("div", { class: "stars", role: "group", "aria-label": `Rate ${movie.title}` });
  const paint = () => [...row.children].forEach((b, i) => b.classList.toggle("on", i < (ratings[movie.id] || 0)));
  for (let s = 1; s <= 5; s++) {
    row.append(el("button", {
      "aria-label": `${s} star${s > 1 ? "s" : ""}`,
      onclick: () => {
        if (ratings[movie.id] === s) delete ratings[movie.id]; else ratings[movie.id] = s;
        store.save(ratings);
        card.classList.toggle("rated", movie.id in ratings);
        paint();
        updateBar();
        if (onRate) onRate();
      },
    }, "★"));
  }
  paint();
  return row;
}

function rateCard(movie) {
  const card = el("article", { class: "card" + (movie.id in ratings ? " rated" : "") });
  card.append(poster(movie), el("div", { class: "meta" },
    el("div", { class: "title" }, movie.title),
    el("div", { class: "sub" }, [movie.year, movie.genres.slice(0, 2).join(", ")].filter(Boolean).join(" · ")),
    starRow(movie, card)));
  return card;
}

// "Because you rated X and Y", using the titles that pushed this movie up the most.
function becauseLine(titles) {
  const parts = [];
  titles.forEach((t, k) => {
    if (k) parts.push(k === titles.length - 1 ? " and " : ", ");
    parts.push(el("em", {}, t));
  });
  return el("div", { class: "sub because" }, "Because you rated ", ...parts);
}

let recsTimer;
function recCard(movie, i) {
  const card = el("li", { class: "card" });
  // Rating a recommended movie removes it and re-ranks after a short pause (lets the user adjust the stars).
  const stars = starRow(movie, card, () => { clearTimeout(recsTimer); recsTimer = setTimeout(() => loadRecs().catch(fail), 900); });
  stars.hidden = true;
  const seen = el("button", { class: "link", onclick: () => { stars.hidden = false; seen.hidden = true; } }, "Seen it? Rate it");
  card.append(el("span", { class: "rank", "aria-hidden": "true" }, i + 1), poster(movie), el("div", { class: "meta" },
    el("div", { class: "title" }, movie.title),
    el("div", { class: "sub" }, [movie.year, movie.genres.slice(0, 2).join(", ")].filter(Boolean).join(" · ")),
    movie.because?.length ? becauseLine(movie.because) : null,
    movie.predicted != null ? el("div", { class: "sub" }, "You'd rate it ", el("span", { class: "predicted" }, `★ ${movie.predicted.toFixed(1)}`)) : null,
    seen, stars,
    el("button", { class: "link", onclick: () => showSimilar(movie) }, "More like this →")));
  return card;
}

async function loadRateGrid(reset) {
  if (reset) { offset = 0; $("rate-grid").replaceChildren(); }
  const genre = $("genre-filter").value;
  const movies = await Rec.popular(40, genre, offset);
  offset += movies.length;
  $("rate-grid").append(...movies.map(rateCard));
  $("more").hidden = movies.length < 40;
}

let searchTimer;
function onSearch() {
  clearTimeout(searchTimer);
  const q = $("search").value.trim();
  searchTimer = setTimeout(async () => {
    try {
      if (q.length < 2) return await loadRateGrid(true);
      const movies = await Rec.search(q, 24);
      if ($("search").value.trim() !== q) return; // the user kept typing
      $("rate-grid").replaceChildren(...movies.map(rateCard));
      $("more").hidden = true;
    } catch (err) {
      console.error(err);
    }
  }, 250);
}

function updateBar() {
  const n = Object.keys(ratings).length;
  $("rated-count").textContent = n;
  $("bar").hidden = n === 0;
  $("go").disabled = n < MIN_RATINGS;
  $("bar-text").textContent = n < MIN_RATINGS
    ? `Rated ${n} of ${MIN_RATINGS}: keep going`
    : `Rated ${n} movies`;
}

let recsRequest = 0;
let ranking = []; // full ranked list for the current ratings; filters are applied on top of it
let recsLimit = 40;

async function loadRecs() {
  const request = ++recsRequest; // a slower, older request must not overwrite a newer one
  const n = Object.keys(ratings).length;
  if (n < MIN_RATINGS) {
    $("recs-status").textContent = `Rate ${MIN_RATINGS - n} more movie${MIN_RATINGS - n === 1 ? "" : "s"} to unlock recommendations.`;
    $("rec-grid").replaceChildren();
    $("recs-more").hidden = true;
    ranking = [];
    return;
  }
  $("recs-status").textContent = "Finding movies for you…";
  const all = await Rec.recommend(ratings);
  if (request !== recsRequest) return;
  ranking = all;
  recsLimit = 40;
  renderRecs();
}

function yearInput(id) {
  const v = parseInt($(id).value, 10);
  return Number.isFinite(v) ? v : null;
}

// Applies the genre, year range and hide-rated filters to the ranked list, then shows the top of what is left.
function renderRecs() {
  const n = Object.keys(ratings).length;
  if (n < MIN_RATINGS) return;
  const genre = $("rec-genre").value, from = yearInput("year-min"), to = yearInput("year-max");
  const hideRated = $("hide-rated").checked;
  const yearBounded = from != null || to != null;
  const matches = ranking.filter((m) => (!hideRated || !(m.id in ratings))
    && (!genre || m.genres.includes(genre))
    && (!yearBounded || (m.year != null && (from == null || m.year >= from) && (to == null || m.year <= to))));
  if (!matches.length) {
    $("recs-status").textContent = ranking.length
      ? "No movies match these filters. Try widening them."
      : "Rate a few movies you liked (4 or 5 stars) so the model knows your taste.";
    $("rec-grid").replaceChildren();
    $("recs-more").hidden = true;
    return;
  }
  const shown = matches.slice(0, recsLimit);
  $("recs-status").textContent = `Ranked for you, best match first, based on your ${n} ratings. ${matches.length} movie${matches.length === 1 ? "" : "s"} match your filters.`;
  $("rec-grid").replaceChildren(...shown.map(recCard));
  $("recs-more").hidden = shown.length >= matches.length;
}

// --- Letterboxd import (all parsing happens in the browser) ---

// RFC 4180 CSV parser: quoted fields may contain commas, doubled quotes and line breaks.
function parseCSV(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}

// Titles compare case, accent and punctuation insensitively; "Matrix, The" matches "The Matrix".
function normTitle(s) {
  return s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase()
    .replace(/^(.*), (the|a|an)$/, "$2 $1")
    .replace(/[^a-z0-9]/g, "");
}

async function importLetterboxd(file) {
  const status = $("import-status");
  status.textContent = "Reading the file…";
  const rows = parseCSV((await file.text()).replace(/^﻿/, "")).filter((r) => r.some((c) => c.trim()));
  const [header = [], ...body] = rows;
  const [nameCol, yearCol, ratingCol] = ["Name", "Year", "Rating"].map((h) => header.indexOf(h));
  if (nameCol < 0 || ratingCol < 0) throw new Error("This does not look like a Letterboxd ratings.csv file.");

  const catalog = await Rec.catalog();
  const byKey = new Map(), byTitle = new Map();
  for (const m of catalog) {
    const t = normTitle(m.title), key = `${t}|${m.year}`;
    if (!byKey.has(key) || byKey.get(key).n < m.n) byKey.set(key, m); // keep the most-rated copy on a collision
    byTitle.set(t, [...(byTitle.get(t) || []), m]);
  }

  let added = 0, updated = 0, same = 0, unrated = 0;
  const unmatched = [];
  for (const row of body) {
    const stars = parseFloat(row[ratingCol]); // Letterboxd uses 0.5 to 5 in half steps
    if (!(stars >= 0.5)) { unrated++; continue; } // logged or watched without a rating
    const title = row[nameCol] || "", year = (row[yearCol] || "").trim(), t = normTitle(title);
    let movie = byKey.get(`${t}|${year}`);
    if (!movie && !year && (byTitle.get(t) || []).length === 1) movie = byTitle.get(t)[0]; // no year: accept a unique title
    if (!movie) { unmatched.push(year ? `${title} (${year})` : title); continue; }
    const value = Math.min(5, Math.max(1, Math.round(stars))); // 0.5 to 5 becomes 1 to 5 whole stars
    if (!(movie.id in ratings)) added++;
    else if (ratings[movie.id] !== value) updated++;
    else same++;
    ratings[movie.id] = value; // the imported rating replaces any rating already set here
  }
  store.save(ratings);
  updateBar();
  loadRateGrid(true).catch(console.error);

  const matched = added + updated + same;
  const sample = unmatched.slice(0, 8).join("; ");
  status.textContent = `Matched ${matched} of ${matched + unmatched.length} rated films (${added} new, ${updated} updated, ${same} unchanged). `
    + `Not found: ${unmatched.length}${unmatched.length ? ` (${sample}${unmatched.length > 8 ? "; …" : ""})` : ""}. `
    + `${unrated} entries without a rating were skipped.`;
}

async function showSimilar(movie) {
  $("similar-title").textContent = `Because you're looking at ${movie.title}`;
  $("similar-grid").replaceChildren();
  $("similar").showModal();
  try {
    const movies = await Rec.similar(movie.id, 12);
    $("similar-grid").replaceChildren(...movies.map(rateCard));
  } catch (err) {
    console.error(err);
    $("similar-grid").replaceChildren(el("p", { class: "lede" }, "Couldn't load similar movies. Try again."));
  }
}

function show(view) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  $("view-rate").hidden = view !== "rate";
  $("view-recs").hidden = view !== "recs";
  $("view-methods").hidden = view !== "methods";
  $("bar").classList.toggle("off", view !== "rate");
  if (view === "recs") loadRecs().catch(fail);
  history.replaceState(null, "", view === "rate" ? location.pathname : `#${view}`);
  window.scrollTo(0, 0);
}

function fail(err) {
  console.error(err);
  $("recs-status").textContent = "Something went wrong loading recommendations. Try refreshing the page.";
}

for (const select of [$("genre-filter"), $("rec-genre")]) {
  select.append(...GENRES.map((g) => el("option", { value: g }, g)));
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => show(t.dataset.view)));
$("go").addEventListener("click", () => show("recs"));
$("more").addEventListener("click", () => loadRateGrid(false).catch(console.error));
$("search").addEventListener("input", onSearch);
$("genre-filter").addEventListener("change", () => { $("search").value = ""; loadRateGrid(true).catch(console.error); });
for (const id of ["rec-genre", "year-min", "year-max", "hide-rated"]) {
  $(id).addEventListener("input", () => { recsLimit = 40; renderRecs(); });
  $(id).addEventListener("change", () => { recsLimit = 40; renderRecs(); });
}
$("recs-more").addEventListener("click", () => { recsLimit += 40; renderRecs(); });
$("lb-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  importLetterboxd(file).catch((err) => {
    console.error(err);
    $("import-status").textContent = err.message || "Could not read that file.";
  });
});
$("reset").addEventListener("click", () => { ratings = {}; store.save(ratings); updateBar(); show("rate"); loadRateGrid(true).catch(console.error); });
$("close-similar").addEventListener("click", () => $("similar").close());

updateBar();
if (["recs", "methods"].includes(location.hash.slice(1))) show(location.hash.slice(1));
loadRateGrid(true).catch((err) => {
  console.error(err);
  $("rate-grid").replaceChildren(el("p", { class: "lede" },
    "Couldn't load the movie list. Try refreshing the page."));
});
