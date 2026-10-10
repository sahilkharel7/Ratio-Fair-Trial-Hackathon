import { defineConfig } from "vite";

export default defineConfig({
  build: {
    // Fonts are always emitted as files: the Content-Security-Policy sent by
    // scripts/serve_workspace.py and vercel.json does not allow data: fonts.
    assetsInlineLimit: (file) => (/\.woff2?$/.test(file) ? false : undefined),
  },
});
