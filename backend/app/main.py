import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.exceptions import ExternalServiceError, InvalidModelOutputError, TruthEngineError
from app.models import ErrorResponse, VerifyRequest, VerificationResponse
from app.neo4j_repository import Neo4jGraphRepository
from app.pipeline import VerificationPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("truth")

app = FastAPI(title="Truth Verification Engine", version="0.1.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


def get_pipeline() -> VerificationPipeline:
    return VerificationPipeline()


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        {
            "event": "request_completed",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
        }
    )
    return response


@app.exception_handler(TruthEngineError)
async def truth_engine_exception_handler(_: Request, exc: TruthEngineError) -> JSONResponse:
    return JSONResponse(status_code=502, content=ErrorResponse(detail=str(exc)).model_dump())


@app.get("/health")
async def health() -> dict[str, str]:
    try:
        Neo4jGraphRepository(settings).verify_connection()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc
    return {"status": "ok", "neo4j": "connected"}


@app.post("/verify", response_model=VerificationResponse, responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
async def verify(request: VerifyRequest) -> VerificationResponse:
    logger.info({"event": "verify_requested", "input_type": request.type.value, "input_length": len(request.input)})
    try:
        return await get_pipeline().verify(request.input, request.type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ExternalServiceError, InvalidModelOutputError):
        raise
