import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './index.css'

const container = document.getElementById('root')
if (container === null) {
  // 不静默失败。`index.html` 少了 `#root` 时,createRoot(null) 会抛一个
  // 相当难读的类型错误,而真正的原因在一行之外。
  throw new Error('index.html 里没有 #root')
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
