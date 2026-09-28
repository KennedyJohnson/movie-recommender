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

let recsTimer;
function recCard(movie, i) {
  const card = el("li", { class: "card" });
  // Rating a recommended movie removes it and re-ranks after a short pause (lets the user adjust the stars).
  const stars = starRow(movie, card, () => { clearTimeout(recsTimer); recsTimer = setTimeout(loadRecs, 900); });
  stars.hidden = true;
  const seen = el("button", { class: "link", onclick: () => { stars.hidden = false; seen.hidden = true; } }, "Seen it? Rate it");
  card.append(el("span", { class: "rank", "aria-hidden": "true" }, i + 1), poster(movie), el("div", { class: "meta" },
    el("div", { class: "title" }, movie.title),
    el("div", { class: "sub" }, [movie.year, movie.genres.slice(0, 2).join(", ")].filter(Boolean).join(" · ")),
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
    if (q.length < 2) return loadRateGrid(true);
    const movies = await Rec.search(q, 24);
    $("rate-grid").replaceChildren(...movies.map(rateCard));
    $("more").hidden = true;
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

async function loadRecs() {
  const n = Object.keys(ratings).length;
  if (n < MIN_RATINGS) {
    $("recs-status").textContent = `Rate ${MIN_RATINGS - n} more movie${MIN_RATINGS - n === 1 ? "" : "s"} to unlock recommendations.`;
    $("rec-grid").replaceChildren();
    return;
  }
  $("recs-status").textContent = "Finding movies for you…";
  const recs = await Rec.recommend(ratings, 40, $("rec-genre").value || null);
  if (!recs.length) {
    $("recs-status").textContent = "Rate a few movies you liked (4 or 5 stars) so the model knows your taste.";
    $("rec-grid").replaceChildren();
    return;
  }
  $("recs-status").textContent = `Ranked for you, best match first, based on your ${n} ratings.`;
  $("rec-grid").replaceChildren(...recs.map(recCard));
}

async function showSimilar(movie) {
  $("similar-title").textContent = `Because you're looking at ${movie.title}`;
  $("similar-grid").replaceChildren();
  $("similar").showModal();
  const movies = await Rec.similar(movie.id, 12);
  $("similar-grid").replaceChildren(...movies.map(rateCard));
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
$("more").addEventListener("click", () => loadRateGrid(false));
$("search").addEventListener("input", onSearch);
$("genre-filter").addEventListener("change", () => { $("search").value = ""; loadRateGrid(true); });
$("rec-genre").addEventListener("change", () => loadRecs().catch(fail));
$("reset").addEventListener("click", () => { ratings = {}; store.save(ratings); updateBar(); show("rate"); loadRateGrid(true); });
$("close-similar").addEventListener("click", () => $("similar").close());

updateBar();
if (["recs", "methods"].includes(location.hash.slice(1))) show(location.hash.slice(1));
loadRateGrid(true).catch((err) => {
  console.error(err);
  $("rate-grid").replaceChildren(el("p", { class: "lede" },
    "Couldn't load the movie list. Try refreshing the page."));
});
