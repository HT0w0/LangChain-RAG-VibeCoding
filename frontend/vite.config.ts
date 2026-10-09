// **从 `vitest/config` 引入 `defineConfig`**,而不是从 `vite`。
// 从 `vite` 引入的话 `test` 这个键不在它的类型里,`tsc` 会报
// 「对象字面量只能指定已知属性」——而 `vite` 本身跑起来完全正常,
// 所以这个错只在类型检查时出现。
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // 开发期走代理而不是配 CORS:浏览器眼里前后端同源,于是 `fetch` 不需要
    // 任何跨域协商,SSE 也不会踩到「预检请求把流打断」这类问题。
    // 后端的 CORS 中间件仍然留着——生产部署时前后端常常不同源。
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // **`ws: false` 是必须的,而且不是默认值。** 打开的话 Vite 会对
        // `/api` 下的请求尝试升级成 WebSocket,而 SSE 走的是普通 HTTP——
        // 症状是流式响应偶尔卡住不动。
        ws: false,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
