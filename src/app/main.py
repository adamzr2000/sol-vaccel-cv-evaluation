from pathlib import Path
from uuid import uuid4

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.api import routes
from app.config.secrets import load_secrets

SECRETS = load_secrets()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init logic
    app.state.secrets = SECRETS
    app.state.models_dir = Path("models")
    app.state.videos_dir = Path("app/static/videos")
    app.state.videos_upload_dir = Path("app/static/videos/uploads")
    app.state.inference_sessions = {}
    app.state.websockets = {}
    yield


def create_app():
    # Create FastAPI app
    app = FastAPI(lifespan=lifespan)

    # Generate an ID for each session
    @app.middleware("http")
    async def assign_session_id(request: Request, call_next):
        if "id" not in request.session:
            request.session["id"] = str(uuid4())
        response = await call_next(request)
        return response

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yolo-demo.nbfc.io", "http://localhost", "http://127.0.0.1", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add Session middleware for per-session cookies
    app.add_middleware(
        SessionMiddleware,
        secret_key=SECRETS["SESSION_SECRET_KEY"],
        session_cookie="session",
        https_only=False,
    )

    # Mount static files and templates
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    templates = Jinja2Templates(directory="app/templates")
    routes.templates = templates
    app.include_router(routes.router)

    return app


app = create_app()