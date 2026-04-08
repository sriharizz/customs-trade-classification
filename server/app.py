try:
    from ..models import CustomsAction, CustomsObservation
except ImportError:
    from models import CustomsAction, CustomsObservation  # type: ignore

from openenv.core.env_server import create_fastapi_app
from .environment import CustomsEnvironment

app = create_fastapi_app(
    CustomsEnvironment,
    CustomsAction,
    CustomsObservation,
)

# Friendly root endpoint for visitors to the Hugging Face Space
@app.get("/")
def read_root():
    return {
        "project": "Customs Trade Classification Environment",
        "description": "Meta PyTorch OpenEnv Hackathon 2026",
        "status": "Running",
        "endpoints": ["POST /reset", "POST /step", "GET /health"],
        "docs": "https://github.com/sriharizz/customs-trade-classification"
    }

# Explicit health check endpoint (used by Dockerfile HEALTHCHECK and HF Spaces)
@app.get("/health")
def health():
    return {"status": "ok"}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
