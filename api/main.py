import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from recommender import Recommender

app = FastAPI(title="Movie Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
rec = Recommender()


class Rating(BaseModel):
    movie_id: int
    rating: float = Field(ge=0.5, le=5.0)


class RecommendRequest(BaseModel):
    ratings: list[Rating] = Field(max_length=500)
    n: int = Field(20, ge=1, le=100)
    genre: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": rec.kind, "movies": len(rec.movies)}


@app.get("/movies/popular")
def popular(n: int = Query(40, ge=1, le=100), offset: int = Query(0, ge=0), genre: str | None = None):
    return rec.popular(n, genre, offset)


@app.get("/movies/search")
def search(q: str = Query(min_length=2, max_length=100), n: int = Query(10, ge=1, le=50)):
    return rec.search(q, n)


@app.get("/movies/{movie_id}/similar")
def similar(movie_id: int, n: int = Query(12, ge=1, le=50)):
    if movie_id not in rec.index:
        raise HTTPException(404, "Unknown movie")
    return rec.similar(movie_id, n)


@app.post("/recommend")
def recommend(req: RecommendRequest):
    return rec.recommend({r.movie_id: r.rating for r in req.ratings}, req.n, req.genre)
