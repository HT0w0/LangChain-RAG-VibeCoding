import { ConfigProvider, App as AntApp } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/AppLayout'
import { RequireAuth } from './components/RequireAuth'
import { ChatPage } from './pages/Chat'
import { KnowledgeBasesPage } from './pages/KnowledgeBases'
import { LoginPage } from './pages/Login'
import { MetricsPage } from './pages/Metrics'
import { ProfilePage } from './pages/Profile'
import { RegisterPage } from './pages/Register'
import { UsersPage } from './pages/Users'
import { AuthProvider } from './store/auth'

export default function App() {
  return (
    // `zhCN` 是必须的:antd 默认是英文,表格的「暂无数据」「每页 20 条」
    // 这些文案会中英混杂。
    <ConfigProvider locale={zhCN}>
      {/* antd 的 `message`/`notification` 需要这层 Provider 才能拿到
          ConfigProvider 的上下文,否则它们用的是默认主题。 */}
      <AntApp>
        <BrowserRouter>
          <AuthProvider>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />

              {/* 普通用户及以上 */}
              <Route element={<RequireAuth />}>
                <Route element={<AppLayout />}>
                  <Route path="/chat" element={<ChatPage />} />
                  <Route path="/profile" element={<ProfilePage />} />
                </Route>
              </Route>

              {/* 仅管理员。**这是体验,不是防线**——后端每一个管理接口
                  都挂了 `require_admin`。 */}
              <Route element={<RequireAuth adminOnly />}>
                <Route element={<AppLayout />}>
                  <Route path="/kb" element={<KnowledgeBasesPage />} />
                  <Route path="/users" element={<UsersPage />} />
                  <Route path="/metrics" element={<MetricsPage />} />
                </Route>
              </Route>

              <Route path="*" element={<Navigate to="/chat" replace />} />
            </Routes>
          </AuthProvider>
        </BrowserRouter>
      </AntApp>
    </ConfigProvider>
  )
}
