// In-browser port of api/recommender.py (EASE only). Data comes from scripts/export_static.py.
const Rec = (() => {
  const shards = new Map();
  let movies, index, meta;

  async function init() {
    if (movies) return;
    [movies, meta] = await Promise.all(["data/movies.json", "data/meta.json"].map((u) => fetch(u).then((r) => r.json())));
    index = new Map(movies.map((m, i) => [m.id, i]));
  }

  function shard(s) {
    if (!shards.has(s)) {
      shards.set(s, fetch(`data/ease/${s}.bin`).then((r) => r.arrayBuffer()).then((buf) => {
        const [rows, nnz] = new Int32Array(buf, 0, 2);
        const indptr = new Int32Array(buf, 8, rows + 1);
        const indices = new Int32Array(buf, 8 + 4 * (rows + 1), nnz);
        const data = new Float32Array(buf, 8 + 4 * (rows + 1 + nnz), nnz);
        return { indptr, indices, data };
      }));
    }
    return shards.get(s);
  }

  // EASE score = sum of the given movies' weight rows
  async function scoreRows(rows) {
    const scores = new Float64Array(movies.length);
    await Promise.all(rows.map(async (i) => {
      const sh = await shard(Math.floor(i / meta.shard));
      const r = i % meta.shard;
      for (let k = sh.indptr[r]; k < sh.indptr[r + 1]; k++) scores[sh.indices[k]] += sh.data[k];
    }));
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
      const liked = known.filter(([, r]) => r >= meta.min_rating).map(([i]) => i);
      if (!liked.length) return [];
      const scores = await scoreRows(liked);
      for (const [i] of known) scores[i] = -Infinity;
      return top(scores, n, (i) => !genre || movies[i].genres.includes(genre)).map((i) => movies[i]);
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
