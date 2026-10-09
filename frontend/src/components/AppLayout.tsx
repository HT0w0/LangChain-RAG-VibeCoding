/**
 * 应用外壳:顶部导航 + 内容区。
 *
 * **菜单项按角色生成,而不是「全渲染再隐藏」。** 隐藏只是视觉上的,
 * 普通用户的浏览器里仍然存在那个入口的代码;更重要的是,菜单是用户
 * 理解「我能做什么」的唯一途径,按角色生成才能让这件事诚实。
 * 真正的防线在后端(`require_admin`),这里只是体验。
 */

import { Layout, Menu, Space, Tag, Typography } from 'antd'
import {
  BarChartOutlined,
  BookOutlined,
  MessageOutlined,
  TeamOutlined,
  UserOutlined,
} from '@ant-design/icons'
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
  // 精确匹配的话进详情后整个菜单都不高亮,看起来像"没在任何一节里"。
  const selected = items
    .map((item) => item.key)
    .filter((key) => location.pathname === key || location.pathname.startsWith(`${key}/`))
    .slice(-1)

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
        <Typography.Text strong style={{ color: '#fff', whiteSpace: 'nowrap' }}>
          电商知识库问答
        </Typography.Text>
        <Menu
          theme="dark"
          mode="horizontal"
          selectedKeys={selected}
          items={items}
          style={{ flex: 1, minWidth: 0 }}
        />
        <Space>
          <Typography.Text style={{ color: 'rgba(255,255,255,0.85)' }}>
            {user?.username}
          </Typography.Text>
          {isAdmin && <Tag color="gold">管理员</Tag>}
          <a
            style={{ color: 'rgba(255,255,255,0.85)' }}
            onClick={() => {
              logout()
              navigate('/login', { replace: true })
            }}
          >
            退出
          </a>
        </Space>
      </Header>
      <Content style={{ padding: 24 }}>
        <Outlet />
      </Content>
    </Layout>
  )
}
