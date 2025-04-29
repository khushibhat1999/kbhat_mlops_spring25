from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading model pipeline... 🎵")
    app.state.model_pipeline = joblib.load("../songs_pipeline.joblib")  # updated here 🔥
    print("model loaded successfully! 🚀")
    yield
    print("server shutting down... Bye! 👋")

app = FastAPI(
    title="Song Genre Classifier",
    description="Predict the genre of a song based on its features 🎶",
    version="0.1",
    lifespan=lifespan
)

class SongRequestBody(BaseModel):
    artist: str
    song: str
    duration_ms: int
    explicit: bool
    year: int
    popularity: int
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float

@app.get('/')
def main():
    return {'message': 'Welcome to the Song Genre Classifier API'}

@app.post('/predict')
def predict(data: SongRequestBody):
    model_pipeline = app.state.model_pipeline
    X = [{
        "artist": data.artist,
        "song": data.song,
        "duration_ms": data.duration_ms,
        "explicit": data.explicit,
        "year": data.year,
        "popularity": data.popularity,
        "danceability": data.danceability,
        "energy": data.energy,
        "key": data.key,
        "loudness": data.loudness,
        "mode": data.mode,
        "speechiness": data.speechiness,
        "acousticness": data.acousticness,
        "instrumentalness": data.instrumentalness,
        "liveness": data.liveness,
        "valence": data.valence,
        "tempo": data.tempo,
    }]
    prediction = model_pipeline.predict(X)
    return {'Predicted Genre': prediction.tolist()}
