# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.config import get_settings
from app.services.database import create_tables
import uvicorn
import logging

settings = get_settings()

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-agent procurement system with non-deterministic A2A communication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses"""
    
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


# Include routes
app.include_router(router, prefix="/api/v1", tags=["procurement"])


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    
    logging.info(f"Starting {settings.APP_NAME}...")
    logging.info(f"Debug mode: {settings.DEBUG}")
    logging.info(f"Model: {settings.MODEL_NAME}")
    
    # Create database tables
    try:
        create_tables()
        logging.info("Database tables initialized")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logging.info("Shutting down...")


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    
    return {
        "message": "Procurement Multi-Agent System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/info")
async def info():
    """System information endpoint"""
    
    return {
        "app_name": settings.APP_NAME,
        "version": "1.0.0",
        "debug": settings.DEBUG,
        "model": settings.MODEL_NAME,
        "max_iterations": settings.MAX_ITERATIONS,
        "endpoints": {
            "query": "/api/v1/query",
            "health": "/api/v1/health",
            "sessions": "/api/v1/sessions",
            "session_trace": "/api/v1/session/{session_id}",
            "debug": "/api/v1/debug/session/{session_id}"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )