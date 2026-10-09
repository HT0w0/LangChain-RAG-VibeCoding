/**
 * 路由守卫。
 *
 * **它只是体验,不是防线。** 真正的权限在后端:每个受保护接口都有
 * `get_current_user` / `require_admin` 依赖。前端守卫的作用是别让用户
 * 点了半天才发现没权限——绕过它顶多能看到一个空壳页面,拿不到任何数据。
 *
 * 两处细节都不能省:
 *
 * * **loading 期间什么都不渲染。** 直接渲染路由的话,刷新页面时会先
 *   闪一下登录页再跳回来——因为「有没有 token」要等 `/auth/me` 回来才
 *   知道。那一闪看起来像"被登出了"。
 * * **记下来源路径。** 从 `/kb/3` 被踢到登录页之后,登录完该回到
 *   `/kb/3`,而不是一律丢回首页。
 */

import { Spin } from 'antd'
import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from '../store/auth'

export function RequireAuth({ adminOnly = false }: { adminOnly?: boolean }) {
  const { user, loading, isAdmin } = useAuth()
  const location = useLocation()

  if (loading) {
    return (
      <div style={{ display: 'grid', placeItems: 'center', minHeight: '60vh' }}>
        <Spin size="large" />
      </div>
    )
  }

  if (user === null) {
    // `replace` 是必须的:不替换的话用户在登录页按「后退」会回到那个
    // 受保护页面,又被弹回来,历史里堆满来回跳转的记录。
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  if (adminOnly && !isAdmin) {
    // 已经登录但权限不够:**回首页,而不是跳登录页**。跳登录页的话
    // 用户会以为自己掉线了,重新登录一次结果还是一样。
    return <Navigate to="/chat" replace />
  }

  return <Outlet />
}
