import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

/**
 * 整棵树的挂载冒烟。
 *
 * **这是唯一一条能回答「这个应用到底能不能打开」的用例。** 别的用例都在
 * 测模块,而模块全对、应用打不开是完全可能的:import 路径、Provider 嵌套、
 * 路由配置、以及**组件库版本**——任何一处对不上,浏览器里白屏,而这
 * 二十几条模块用例一条都不会红。
 *
 * 版本那一项不是假想的风险:`antd` 装成的是 v6(脚手架拉的是 latest),
 * 而这套代码是按 v5 的习惯写的。v6 删改了一批 API(静态 `message`、
 * `Empty.PRESENTED_IMAGE_SIMPLE`、`destroyOnClose`),漏掉一处就是运行时
 * 报错。下面的 `console.error` 断言把它兜住——React 对渲染期的类型问题、
 * 未知 DOM 属性、重复 key 都会往这里写。
 */

const USER = {
  id: 1,
  username: 'admin',
  role: 'admin',
  is_active: true,
  created_at: '2026-10-09T12:00:00+00:00',
  last_login_at: null,
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

/** 按 URL 片段分派。没登记的路径一律 404,而不是静默给个空对象。 */
function stubApi(overrides: Record<string, () => Response> = {}) {
  const routes: Record<string, () => Response> = {
    '/auth/login': () =>
      jsonResponse({
        access_token: 'test-token',
        token_type: 'bearer',
        expires_in: 720,
        user: USER,
      }),
    '/auth/me': () => jsonResponse(USER),
    '/kb/options': () => jsonResponse({ items: [{ id: 1, name: '商品库' }] }),
    '/kb': () =>
      jsonResponse({
        total: 1,
        items: [
          {
            id: 1,
            name: '商品库',
            description: null,
            created_by: 1,
            doc_count: 2,
            chunk_count: 8,
            created_at: '2026-10-09T12:00:00+00:00',
            updated_at: '2026-10-09T12:00:00+00:00',
          },
        ],
      }),
    '/sessions': () => jsonResponse({ total: 0, items: [] }),
    '/admin/users': () => jsonResponse({ total: 1, items: [USER] }),
    '/admin/metrics': () =>
      jsonResponse({
        window_minutes: 60,
        requests: 0,
        outcomes: {},
        error_rate: 0,
        total_ms: { p50: 0, p95: 0, max: 0, avg: 0 },
        first_token_ms: null,
        stages: {},
        cache_hit_rate_retrieval: 0,
        cache_hit_rate_answer: 0,
        grounded_rate: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        cached_prompt_tokens: 0,
        cache_read_rate: null,
      }),
    ...overrides,
  }

  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString()
    for (const [fragment, make] of Object.entries(routes)) {
      if (url.includes(fragment)) return make()
    }
    return jsonResponse({ detail: `测试没登记这条路径:${url}` }, 404)
  })

  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

let consoleErrors: unknown[] = []

beforeEach(() => {
  localStorage.clear()
  consoleErrors = []
  // 抓 console.error 而不是让它打出来:React 的渲染期告警全走这条路,
  // 而它们在测试输出里只是一段滚动过去的文字,没人会看。
  vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    consoleErrors.push(args[0])
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('应用挂载', () => {
  it('未登录时落在登录页,并且渲染期间不报错', async () => {
    stubApi()
    window.history.pushState({}, '', '/login')

    render(<App />)

    expect(await screen.findByText('电商知识库问答')).toBeInTheDocument()
    expect(screen.getByLabelText('用户名')).toBeInTheDocument()
    expect(consoleErrors).toEqual([])
  })

  it('**登录成功后进到问答页** —— 这条覆盖 Provider、路由与守卫的整套装配', async () => {
    stubApi()
    window.history.pushState({}, '', '/login')

    render(<App />)
    await userEvent.type(await screen.findByLabelText('用户名'), 'admin')
    await userEvent.type(screen.getByLabelText('密码'), '123456')
    await userEvent.click(screen.getByRole('button', { name: '登 录' }))

    // 进到问答页的标志:会话侧栏 + 知识库范围选择器
    expect(await screen.findByText('会话')).toBeInTheDocument()
    expect(await screen.findByText('商品库')).toBeInTheDocument()
    expect(consoleErrors).toEqual([])
  })

  it('登录失败时把**服务端的话**显示出来,而不是自己编一句', async () => {
    stubApi({
      '/auth/login': () => jsonResponse({ detail: '账号已被停用' }, 403),
    })
    window.history.pushState({}, '', '/login')

    render(<App />)
    await userEvent.type(await screen.findByLabelText('用户名'), 'someone')
    await userEvent.type(screen.getByLabelText('密码'), 'whatever')
    await userEvent.click(screen.getByRole('button', { name: '登 录' }))

    expect(await screen.findByText('账号已被停用')).toBeInTheDocument()
  })

  it('管理员能看到知识库/用户/指标三个入口', async () => {
    localStorage.setItem('rag.token', 'test-token')
    stubApi()
    window.history.pushState({}, '', '/chat')

    render(<App />)

    await waitFor(() => expect(screen.getByRole('link', { name: '知识库' })).toBeInTheDocument())
    expect(screen.getByRole('link', { name: '用户' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '指标' })).toBeInTheDocument()
  })

  it('普通用户看不到管理入口', async () => {
    localStorage.setItem('rag.token', 'test-token')
    stubApi({ '/auth/me': () => jsonResponse({ ...USER, role: 'user' }) })
    window.history.pushState({}, '', '/chat')

    render(<App />)

    await waitFor(() => expect(screen.getByRole('link', { name: '问答' })).toBeInTheDocument())
    expect(screen.queryByRole('link', { name: '知识库' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: '指标' })).not.toBeInTheDocument()
  })

  // 一页一条。**每页都渲染一遍并且在渲染期不许有任何 console.error**——
  // 这既是「每个页面都打得开」的证明,也是组件库版本变更的探测器:
  // antd 的废弃告警正是走这条通道,而它们不会让任何别的用例变红。
  // 标记文本必须是**这一页独有**的。写「知识库」会同时匹配到导航项和
  // 卡片标题,于是断言变成「找到多个」——那是测试写得不严,不是页面错了。
  it.each([
    ['/chat', '会话'],
    ['/profile', '账号信息'],
    ['/kb', '商品库'],
    ['/users', '用户管理'],
    ['/metrics', '请求数'],
  ])('%s 能正常渲染', async (route, marker) => {
    localStorage.setItem('rag.token', 'test-token')
    stubApi()
    window.history.pushState({}, '', route)

    render(<App />)

    expect(await screen.findByText(marker)).toBeInTheDocument()
    await waitFor(() => expect(consoleErrors).toEqual([]))
  })

  it('**普通用户直接敲管理页地址会被挡回问答页**', async () => {
    localStorage.setItem('rag.token', 'test-token')
    stubApi({ '/auth/me': () => jsonResponse({ ...USER, role: 'user' }) })
    window.history.pushState({}, '', '/metrics')

    render(<App />)

    // 被重定向到 /chat,而不是停在指标页、也不是被踢到登录页。
    await waitFor(() => expect(window.location.pathname).toBe('/chat'))
    expect(screen.queryByText('请求数')).not.toBeInTheDocument()
  })
})
