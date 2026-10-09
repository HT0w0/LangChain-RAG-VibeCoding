/**
 * 登录状态。
 *
 * **每次加载都要拿 token 去换一次用户信息**,不能只凭 localStorage 里有
 * 一把钥匙就认为登录着。token 可能已经过期、账号可能已被管理员停用——
 * 这两种情况下界面会先渲染出完整的应用外壳,然后每个操作都失败,
 * 而用户完全不知道自己其实没登录。
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { getToken, setToken, setUnauthorizedHandler } from '../api/client'
import { authApi } from '../api/endpoints'
import type { UserOut } from '../api/types'

interface AuthState {
  user: UserOut | null
  /** 首次校验 token 期间为 true。**这期间不能渲染路由**,否则会先闪一下登录页。 */
  loading: boolean
  isAdmin: boolean
  login: (username: string, password: string) => Promise<void>
  register: (username: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserOut | null>(null)
  const [loading, setLoading] = useState(true)

  const logout = useCallback(() => {
    setToken(null)
    setUser(null)
  }, [])

  // 任意请求撞上 401 时清掉身份。**这里只清状态、不跳转**——跳转交给
  // 路由守卫,否则「从哪个页面被踢出去、回来时该回哪儿」会散在两处。
  useEffect(() => {
    setUnauthorizedHandler(() => setUser(null))
    return () => setUnauthorizedHandler(null)
  }, [])

  useEffect(() => {
    let cancelled = false

    async function restore() {
      if (getToken() === null) {
        setLoading(false)
        return
      }
      try {
        const me = await authApi.me()
        if (!cancelled) setUser(me)
      } catch {
        // token 失效(过期/被停用/签名换了)。清掉它,免得每次刷新
        // 都白跑一次请求。
        if (!cancelled) setToken(null)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void restore()
    return () => {
      cancelled = true
    }
  }, [])

  const login = useCallback(async (username: string, password: string) => {
    const result = await authApi.login(username, password)
    setToken(result.access_token)
    setUser(result.user)
  }, [])

  const register = useCallback(async (username: string, password: string) => {
    const result = await authApi.register(username, password)
    // 注册接口直接返回 token,不用再登录一次。
    setToken(result.access_token)
    setUser(result.user)
  }, [])

  const value = useMemo<AuthState>(
    () => ({ user, loading, isAdmin: user?.role === 'admin', login, register, logout }),
    [user, loading, login, register, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext)
  if (context === null) {
    // 抛而不是返回一个默认值:忘了套 Provider 的话,默认值会让「登录后
    // 界面不变」这种症状到处冒,而真正的错因离现场很远。
    throw new Error('useAuth 必须在 <AuthProvider> 内使用')
  }
  return context
}
