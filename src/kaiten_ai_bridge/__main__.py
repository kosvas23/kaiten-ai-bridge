"""Console entry point."""

from __future__ import annotations

import uvicorn

from .app import create_app
from .config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.bind_host,
        port=settings.bind_port,
        access_log=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
