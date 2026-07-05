// Ambient types for Vite's import.meta.env so build-time variables such as
// VITE_WS_URL are statically known instead of being reached via `as any`.

/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_WS_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
