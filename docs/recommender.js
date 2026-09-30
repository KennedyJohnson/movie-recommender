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
  // Also tracks which given movie pushed each target up the most, for "Because you liked" labels.
  async function scoreRows(rows, weights) {
    const scores = new Float64Array(movies.length);
    const bestPush = new Float64Array(movies.length);
    const because = new Int32Array(movies.length).fill(-1);
    await Promise.all(rows.map(async (i, j) => {
      const w = weights ? weights[j] : 1;
      const sh = await shard(Math.floor(i / meta.shard));
      const r = i % meta.shard;
      for (let k = sh.indptr[r]; k < sh.indptr[r + 1]; k++) {
        const t = sh.indices[k], push = w * sh.data[k];
        scores[t] += push;
        if (push > bestPush[t]) { bestPush[t] = push; because[t] = i; }
      }
    }));
    scores.because = because;
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
    async recommend(rated, n, genre) {
      await init();
      const known = Object.entries(rated).map(([id, r]) => [index.get(+id), r]).filter(([i]) => i !== undefined);
      if (!known.length) return this.popular(n, genre, 0);
      if (!known.some(([, r]) => r >= meta.min_rating)) return [];
      // weight each rated movie by (stars - 3)^3: 4/2 stars nudge similar movies (+/-1), 5/1 stars move them hard (+/-8)
      const used = known.filter(([, r]) => r !== NEUTRAL);
      const scores = await scoreRows(used.map(([i]) => i), used.map(([, r]) => (r - NEUTRAL) ** 3));
      for (const [i] of known) scores[i] = -Infinity;
      return top(scores, n, (i) => !genre || movies[i].genres.includes(genre)).map((i) => {
        const b = scores.because[i];
        return b >= 0 ? { ...movies[i], because: movies[b].title } : movies[i];
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
  };
})();
