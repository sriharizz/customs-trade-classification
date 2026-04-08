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

from fastapi.responses import HTMLResponse

# Friendly root endpoint with a beautiful UI for the Hackathon Judges!
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Customs Trade Classification Environment</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0f172a;
                --surface: rgba(30, 41, 59, 0.7);
                --border: rgba(255, 255, 255, 0.1);
                --text: #f8fafc;
                --text-muted: #94a3b8;
                --accent: #38bdf8;
                --accent-glow: rgba(56, 189, 248, 0.5);
                --green: #22c55e;
            }
            body {
                margin: 0;
                padding: 0;
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, var(--bg) 0%, #020617 100%);
                color: var(--text);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }
            .grid-bg {
                position: absolute;
                inset: 0;
                background-image: 
                    linear-gradient(to right, rgba(255,255,255,0.02) 1px, transparent 1px),
                    linear-gradient(to bottom, rgba(255,255,255,0.02) 1px, transparent 1px);
                background-size: 50px 50px;
                z-index: 0;
            }
            .glow {
                position: absolute;
                width: 600px;
                height: 600px;
                background: radial-gradient(circle, var(--accent-glow) 0%, transparent 60%);
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 0;
                opacity: 0.4;
                pointer-events: none;
            }
            .container {
                position: relative;
                z-index: 10;
                background: var(--surface);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 3rem;
                max-width: 600px;
                width: 90%;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                text-align: center;
                animation: fade-in 1s ease-out;
            }
            @keyframes fade-in {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            h1 {
                font-size: 2.5rem;
                font-weight: 800;
                margin: 0 0 1rem 0;
                background: linear-gradient(to right, #fff, var(--accent));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.02em;
            }
            .badge {
                display: inline-block;
                background: rgba(34, 197, 94, 0.1);
                color: var(--green);
                border: 1px solid rgba(34, 197, 94, 0.2);
                padding: 0.5rem 1rem;
                border-radius: 999px;
                font-size: 0.875rem;
                font-weight: 600;
                margin-bottom: 2rem;
                box-shadow: 0 0 10px rgba(34, 197, 94, 0.2);
            }
            p {
                color: var(--text-muted);
                line-height: 1.6;
                font-size: 1.125rem;
                margin-bottom: 2.5rem;
            }
            .api-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
                gap: 1rem;
                margin-bottom: 2.5rem;
            }
            .api-card {
                background: rgba(0, 0, 0, 0.2);
                border: 1px solid var(--border);
                padding: 1rem;
                border-radius: 12px;
                font-family: monospace;
                font-size: 0.9rem;
                color: var(--accent);
                transition: all 0.2s ease;
            }
            .api-card:hover {
                background: rgba(56, 189, 248, 0.1);
                border-color: var(--accent);
                transform: translateY(-2px);
            }
            .btn {
                display: inline-block;
                text-decoration: none;
                color: #0f172a;
                background: var(--text);
                padding: 0.875rem 2rem;
                border-radius: 12px;
                font-weight: 600;
                transition: all 0.2s ease;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 15px -3px rgba(255, 255, 255, 0.1);
            }
            .footer {
                margin-top: 2rem;
                font-size: 0.875rem;
                color: var(--text-muted);
            }
        </style>
    </head>
    <body>
        <div class="grid-bg"></div>
        <div class="glow"></div>
        <div class="container">
            <div class="badge">● Server Online & Ready</div>
            <h1>Customs Trade <br/> Classification</h1>
            <p>An autonomous compliance pipeline environment evaluating reasoning boundaries across geopolitical risk and complex HTS tabular data.</p>
            
            <div class="api-grid">
                <div class="api-card">POST /reset</div>
                <div class="api-card">POST /step</div>
                <div class="api-card">GET /health</div>
            </div>

            <a href="https://github.com/sriharizz/customs-trade-classification" target="_blank" class="btn">
                View Documentation
            </a>
            
            <div class="footer">
                Built for the Meta PyTorch OpenEnv Hackathon 2026
            </div>
        </div>
    </body>
    </html>
    """

# Explicit health check endpoint (used by Dockerfile HEALTHCHECK and HF Spaces)
@app.get("/health")
def health():
    return {"status": "ok"}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
