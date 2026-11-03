import { defineConfig } from 'vite';

export default defineConfig({
  build: { target: 'es2022' },  // top-level await in main.js
  server: { proxy: { '/api': 'http://127.0.0.1:8790' } },  // platform/server/app.py
});
