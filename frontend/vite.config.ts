import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Proxy /api/* to the FastAPI backend during development.
// This avoids CORS issues — the browser talks to :5173 only,
// and Vite transparently forwards /api/* to :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
