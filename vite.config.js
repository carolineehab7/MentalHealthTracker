import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Every fetch('/api/...') in React → http://localhost:5000/...
      '/api': {
        target:       'http://localhost:5000',
        changeOrigin: true,
        rewrite:      (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
