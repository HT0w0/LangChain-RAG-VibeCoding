/**
 * 应用外壳:顶部导航 + 内容区。
 *
 * **菜单项按角色生成,而不是「全渲染再隐藏」。** 隐藏只是视觉上的,
 * 普通用户的浏览器里仍然存在那个入口的代码;更重要的是,菜单是用户
 * 理解「我能做什么」的唯一途径,按角色生成才能让这件事诚实。
 * 真正的防线在后端(`require_admin`),这里只是体验。
 *
 * 布局上有一个容易忽略的点:**内容区是固定高度 + 自己滚**。做成整页滚动
 * 的话,会话页的输入框会被长答案顶到屏幕外面去,用户得先滚到底才能打字。
 */

import {
  BarChartOutlined,
  BookOutlined,
  MessageOutlined,
  TeamOutlined,
  UserOutlined,
} from '@ant-design/icons'
import { Layout, Menu, Tag } from 'antd'
import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../store/auth'

const { Header, Content } = Layout

export function AppLayout() {
  const { user, isAdmin, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()

  const items = [
    { key: '/chat', icon: <MessageOutlined />, label: <Link to="/chat">问答</Link> },
    { key: '/profile', icon: <UserOutlined />, label: <Link to="/profile">个人设置</Link> },
    ...(isAdmin
      ? [
          { key: '/kb', icon: <BookOutlined />, label: <Link to="/kb">知识库</Link> },
          { key: '/users', icon: <TeamOutlined />, label: <Link to="/users">用户</Link> },
          { key: '/metrics', icon: <BarChartOutlined />, label: <Link to="/metrics">指标</Link> },
        ]
      : []),
  ]

  // 用**前缀匹配**高亮:详情页的路径(`/kb/3`)在菜单里没有对应项,
  // 精确匹配的话进详情后整个菜单都不高亮,看起来像「没在任何一节里」。
  const selected = items
    .map((item) => item.key)
    .filter((key) => location.pathname === key || location.pathname.startsWith(`${key}/`))
    .slice(-1)

  return (
    <Layout style={{ minHeight: '100vh', background: 'var(--app-bg)' }}>
      <Header className="app-header">
        <span className="app-brand">电商知识库问答</span>
        <Menu theme="dark" mode="horizontal" selectedKeys={selected} items={items} />
        <div className="app-user">
          <span>{user?.username}</span>
          {isAdmin && <Tag color="gold">管理员</Tag>}
          <a
            className="app-logout"
            onClick={() => {
              logout()
              navigate('/login', { replace: true })
            }}
          >
            退出
          </a>
        </div>
      </Header>
      <Content className="app-content">
        {/* 宽度约束放在这里而不是各页自己写:漏一页就会在宽屏上横跨
            1440px,而那一页看起来"只是字有点长",不会有人去查。 */}
        <div className="page">
          <Outlet />
        </div>
      </Content>
    </Layout>
  )
}
