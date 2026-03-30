"""OpenEnv server entry point for local and container execution."""

from __future__ import annotations

import uvicorn

from app import app as root_app

app = root_app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the OpenEnv API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
