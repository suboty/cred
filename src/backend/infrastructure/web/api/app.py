import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import RedirectResponse # noqa

from container import init_container
from infrastructure.config import app_settings
from infrastructure.web.api import api_router
from infrastructure.web.api.exceptions import FormFieldValidationException


container = init_container()


@asynccontextmanager
async def lifespan(app: FastAPI): # noqa
    yield

app = FastAPI(
    title=app_settings.PROJECT_NAME,
    version=app_settings.PROJECT_VERSION,
    openapi_url=f"/api/openapi.json",
    lifespan=lifespan,
)
app.include_router(api_router, prefix='/api')

app.add_middleware(
    CORSMiddleware, # noqa
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(FormFieldValidationException)
async def validation_exception_handler(_: Request, exc: FormFieldValidationException):
    return JSONResponse(status_code=exc.status_code, content=exc.fields)


@app.exception_handler(HTTPException)
async def http_validation_exception_handler(_: Request, exc: HTTPException):
    content = {"detail": exc.detail}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request): # noqa
    return RedirectResponse(url="/docs")
