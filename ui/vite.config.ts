/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE_PATH || '/demo/',
  server: {
    port: 5173,
    proxy: {
      '/health': {
        target: process.env.VITE_API_BASE_URL || 'http://127.0.0.1:3001',
        changeOrigin: true,
      },
      '/predict': {
        target: process.env.VITE_API_BASE_URL || 'http://127.0.0.1:3001',
        changeOrigin: true,
      },
      '/batch': {
        target: process.env.VITE_API_BASE_URL || 'http://127.0.0.1:3001',
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test-setup.ts',
  },
});
