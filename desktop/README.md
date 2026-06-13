# MatSci-Agent Desktop

Tauri v2 + React 18 + TypeScript desktop frontend for MatSci-Agent.

## Prerequisites

- [Node.js](https://nodejs.org/) (LTS recommended)
- [Rust](https://rustup.rs/) toolchain
- Windows only: `dlltool.exe` must be on `PATH` for the GNU toolchain build.
  Add MinGW to PATH, for example:
  ```powershell
  $env:PATH += ";C:\mingw64\mingw64\bin"
  ```
- A running MatSci-Agent backend:
  ```bash
  cd agent
  matsci serve
  # or
  python matsci_agent/server.py
  ```

## Setup

```bash
cd desktop
cp .env.example .env
npm install
```

## Development

```bash
npm run tauri dev
```

This starts the Vite dev server and the Tauri window. The app connects to:

- WebSocket: `ws://localhost:8000/ws/agent` (chat)
- HTTP: `http://localhost:8000` (tools, skills, health)

Override the WebSocket URL with `VITE_WS_URL` in `.env`.

## Build

```bash
npm run tauri build
```

The installer/bundle will be placed in `src-tauri/target/release/bundle/`.

## Features

- **Chat**: Real-time streaming conversation with the agent via WebSocket.
- **Tools**: Browse and call registered tools with custom JSON arguments.
- **Skills**: Browse and execute declarative skills (e.g. `standard_dft`,
  `uncertainty_propagation`, `bayesian_calibration`).
- **Memory**: Session overview placeholder; long-term memory backend API is
  not exposed yet.

## Project Structure

```
desktop/
├── src/
│   ├── App.tsx      # Main UI (chat/tools/memory/skills tabs)
│   └── main.tsx     # React entry point
├── src-tauri/
│   ├── src/main.rs  # Tauri runtime + backend health check command
│   └── Cargo.toml   # Rust dependencies
├── package.json
├── vite.config.ts
└── .env.example
```

## Troubleshooting

- `dlltool.exe not found` on Windows: ensure MinGW `bin` directory is on PATH.
- `Backend offline` in the status bar: start `matsci serve` first.
- `Cargo` build errors: run `cargo clean` in `src-tauri` and try again.
