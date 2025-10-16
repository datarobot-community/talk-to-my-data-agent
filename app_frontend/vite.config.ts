import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { nodePolyfills } from 'vite-plugin-node-polyfills';
import path from 'path';
import { VITE_DEFAULT_PORT, VITE_STATIC_DEFAULT_PORT } from './src/constants/dev';

let base: string = '';
// 1. if NOTEBOOK_ID is set, use /notebook-sessions/${NOTEBOOK_ID}/ports/5173/ for dev server
// 2. if NOTEBOOK_ID and NODE_ENV === 'development' are set, use /notebook-sessions/${NOTEBOOK_ID}/ports/8080/ for static codespace server
if (process.env.NOTEBOOK_ID && process.env.NODE_ENV === 'development') {
  const notebookId = process.env.NOTEBOOK_ID;
  const defaultPort = process.env.STATIC_CODESPACE ? VITE_STATIC_DEFAULT_PORT : VITE_DEFAULT_PORT;
  base = `/notebook-sessions/${notebookId}/ports/${defaultPort}/`;
}
// Proxy base for development server (local + codespaces)
const devProxyBase: string = base === '' ? '/' : base;

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    nodePolyfills({
      exclude: [],
      // for plotly.js
      protocolImports: true,
    }),
    {
      name: 'strip-base',
      apply: 'serve',
      configureServer({ middlewares }) {
        middlewares.use((req, _res, next) => {
          if (base !== '' && !req.url?.startsWith(base)) {
            req.url = base.slice(0, -1) + req.url;
          }
          next();
        });
      },
    },
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '~': path.resolve(__dirname, './src'),
    },
  },
  base: base,
  build: {
    outDir: '../app_backend/static/',
    emptyOutDir: true,
    manifest: true,
    rollupOptions: {
      external: ['_dr_env.js'],
    },
  },
  // this config is for interactive development (local + codespaces)
  server: {
    host: true,
    allowedHosts: ['localhost', '127.0.0.1', '.datarobot.com'],
    proxy: {
      [`${devProxyBase}api/`]: {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: path => path.replace(new RegExp(`^${devProxyBase}`), ''),
      },
      [`${devProxyBase}_dr_env.js`]: {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: path => path.replace(new RegExp(`^${devProxyBase}`), ''),
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './tests/setupTests.ts',
    typecheck: {
      tsconfig: './tsconfig.test.json',
    },
  },
});
