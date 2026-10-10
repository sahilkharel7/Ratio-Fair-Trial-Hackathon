// Fonts and other assets are emitted as files, never inlined as data: URLs, so the strict
// Content-Security-Policy (default-src 'self', set by scripts/serve_workspace.py and vercel.json)
// loads every bundled Lato file instead of blocking the small ones.
export default {
  build: { assetsInlineLimit: 0 },
};
