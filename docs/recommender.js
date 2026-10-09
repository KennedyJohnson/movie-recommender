// In-browser port of api/recommender.py (EASE only). Data comes from scripts/export_static.py.
const Rec = (() => {
  const shards = new Map();
  let movies, index, meta;

  async function init() {
    if (movies) return;
    const [m, mt] = await Promise.all(["data/movies.json", "data/meta.json"].map((u) => fetch(u).then((r) => {
      if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`);
      return r.json();
    })));
    [movies, meta] = [m, mt]; // only set once both loaded, so a failed load is retried
    index = new Map(movies.map((m, i) => [m.id, i]));
  }

  function shard(s) {
    if (!shards.has(s)) {
      shards.set(s, fetch(`data/ease/${s}.bin`).then((r) => {
        if (!r.ok) throw new Error(`shard ${s}: HTTP ${r.status}`);
        return r.arrayBuffer();
      }).then((buf) => {
        const [rows, nnz] = new Int32Array(buf, 0, 2);
        const indptr = new Int32Array(buf, 8, rows + 1);
        const indices = new Int32Array(buf, 8 + 4 * (rows + 1), nnz);
        const data = new Float32Array(buf, 8 + 4 * (rows + 1 + nnz), nnz);
        return { indptr, indices, data };
      }).catch((err) => {
        shards.delete(s); // don't keep a failed download; the next request retries it
        throw err;
      }));
    }
    return shards.get(s);
  }

  const NEUTRAL = 3; // star rating that neither pulls similar movies up nor pushes them down

  // EASE score = weighted sum of the given movies' weight rows.
  // Also tracks the two given movies that pushed each target up the most (positive pushes only),
  // for "Because you rated X and Y" labels. scores.because = [firstIds, secondIds], -1 when unset.
  async function scoreRows(rows, weights) {
    const n = movies.length;
    const scores = new Float64Array(n);
    const v1 = new Float64Array(n), v2 = new Float64Array(n);
    const id1 = new Int32Array(n).fill(-1), id2 = new Int32Array(n).fill(-1);
    await Promise.all(rows.map(async (i, j) => {
      const w = weights ? weights[j] : 1;
      const sh = await shard(Math.floor(i / meta.shard));
      const r = i % meta.shard;
      for (let k = sh.indptr[r]; k < sh.indptr[r + 1]; k++) {
        const t = sh.indices[k], push = w * sh.data[k];
        scores[t] += push;
        if (push > 0) {
          if (push > v1[t]) { v2[t] = v1[t]; id2[t] = id1[t]; v1[t] = push; id1[t] = i; }
          else if (push > v2[t]) { v2[t] = push; id2[t] = i; }
        }
      }
    }));
    scores.because = [id1, id2];
    return scores;
  }

  function top(scores, n, keep) {
    return [...scores.keys()].filter((i) => Number.isFinite(scores[i]) && keep(i))
      .sort((a, b) => scores[b] - scores[a]).slice(0, n);
  }

  return {
    async popular(n, genre, offset) {
      await init();
      return movies.filter((m) => !genre || m.genres.includes(genre)).slice(offset, offset + n);
    },
    async search(q, n) {
      await init();
      q = q.toLowerCase().trim();
      return movies.filter((m) => m.title.toLowerCase().includes(q))
        .sort((a, b) => (!a.title.toLowerCase().startsWith(q)) - (!b.title.toLowerCase().startsWith(q)) || b.n - a.n)
        .slice(0, n);
    },
    // Returns the full ranking (best first) so the caller can filter client side.
    // Already-rated movies stay in the list with a score; the caller decides whether to hide them.
    // Each item gets `because`: up to two rated titles that pushed it up the most.
    async recommend(rated, n = Infinity) {
      await init();
      const known = Object.entries(rated).map(([id, r]) => [index.get(+id), r]).filter(([i]) => i !== undefined);
      if (!known.length) return this.popular(n, null, 0);
      if (!known.some(([, r]) => r >= meta.min_rating)) return [];
      // weight each rated movie by (stars - 3)^3: 4/2 stars nudge similar movies (+/-1), 5/1 stars move them hard (+/-8)
      const used = known.filter(([, r]) => r !== NEUTRAL);
      const scores = await scoreRows(used.map(([i]) => i), used.map(([, r]) => (r - NEUTRAL) ** 3));
      const [first, second] = scores.because;
      return top(scores, n, () => true).map((i) => {
        const because = [first[i], second[i]].filter((b) => b >= 0).map((b) => movies[b].title);
        return { ...movies[i], because };
      });
    },
    async similar(id, n) {
      await init();
      const i = index.get(id);
      const sims = await scoreRows([i]);
      sims[i] = -Infinity;
      return top(sims, n, () => true).map((j) => ({ ...movies[j], similarity: Math.round(sims[j] * 1000) / 1000 }));
    },
    async updated() { await init(); return meta.updated; },
    async catalog() { await init(); return movies; },
  };
})();
