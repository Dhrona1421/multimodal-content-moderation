"""Dedicated API entrypoint for the moderation environment."""

from __future__ import annotations

import os

import uvicorn

from app import create_app

app = create_app(api_only=True)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main(port=int(os.getenv("PORT", "7860")))
