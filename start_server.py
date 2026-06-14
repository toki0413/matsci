#!/usr/bin/env python3
"""Quick-start script for Huginn server.

Usage:
    python start_server.py          # Start server on default port 8000
    python start_server.py --port 8080  # Start on custom port
"""

import argparse
import sys
from pathlib import Path

# Ensure the agent package is importable
sys.path.insert(0, str(Path(__file__).parent / "agent"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Huginn Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    import uvicorn
    from huginn.server import app

    print(f"\n{'='*50}")
    print(f"  Huginn Server v0.1.0")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  WebSocket: ws://{args.host}:{args.port}/ws/agent")
    print(f"{'='*50}\n")

    uvicorn.run(
        "huginn.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
