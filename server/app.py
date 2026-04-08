"""OpenEnv server entry point for local and container execution."""

from __future__ import annotations

import uvicorn

from api import app

def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Run the OpenEnv API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
