import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'spa-fallback',
      configureServer(server) {
        // Serve index.html for any path that doesn't match a file (SPA routing).
        // Run first so we rewrite before Vite tries to serve the path.
        const fallback = (req, res, next) => {
          const raw = req.url || ''
          const path = raw.split('?')[0]
          const isGet = req.method === 'GET' || req.method === 'HEAD'
          const isAsset =
            path.startsWith('/api') ||
            path.startsWith('/@') ||
            path.startsWith('/node_modules') ||
            path.includes('/src/') ||
            path.includes('.')
          if (isGet && path !== '/' && !isAsset) {
            req.url = '/'
          }
          next()
        }
        // Prepend so fallback runs before Vite (avoids 404 for /predict, /about, etc.)
        const stack = server.middlewares.stack
        stack.unshift({ route: '/', handle: fallback })
      },
    },
  ],
  server: {
    port: 3000,
    strictPort: false,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  preview: {
    port: 3000,
  },
})
