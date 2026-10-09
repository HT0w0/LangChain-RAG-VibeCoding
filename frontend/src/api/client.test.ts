import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getToken, notifyUnauthorized, request, setToken, setUnauthorizedHandler } from './client'

/**
 * HTTP 出口的那几条规矩。
 *
 * 这里测的不是「请求发得出去」——那是端到端的事。测的是**出错时的处置**,
 * 而它们全都是「不报错、只是行为不对」的那一类:
 *
 *   * 401 之后身份没清干净,用户会停在一个**每个操作都失败、界面却显示
 *     已登录**的页面上,完全不知道该重新登录;
 *   * `fetch` 只在网络层失败时抛,不包一层的话调用方拿到的是一个
 *     `TypeError`,和「参数写错了」长得一样;
 *   * 给 `FormData` 手写 `Content-Type` 会**丢掉 boundary**,
 *     服务端一个字段都解析不出来。
 */

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function fetchMockReturning(response: Response) {
  // **参数要显式写出来。** 写成 `vi.fn(async () => response)` 的话,
  // mock 的调用签名被推成 `[]`,于是 `mock.mock.calls[0][1]` 在类型上
  // 不存在——测试全绿而 `tsc` 红。这个坑是 `npx tsc -b` 抓到的。
  const mock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response)
  vi.stubGlobal('fetch', mock)
  return mock
}

beforeEach(() => {
  localStorage.clear()
})

afterEach(() => {
  vi.unstubAllGlobals()
  setUnauthorizedHandler(null)
})

describe('request', () => {
  it('带上 localStorage 里那把 token', async () => {
    setToken('test-token')
    const mock = fetchMockReturning(jsonResponse({ ok: true }))

    await request('/anything')

    const init = mock.mock.calls[0]?.[1]
    // 先兜到空对象再取属性。写成 `(init?.headers as X).Authorization` 的话,
    // `init` 是 undefined 时会抛 TypeError 而不是干净地断言失败——
    // 报出来的是「测试炸了」,而不是「Authorization 没带上」。
    const headers = (init?.headers ?? {}) as Record<string, string>
    expect(headers.Authorization).toBe('Bearer test-token')
  })

  it('**401 时先清 token,再通知上层**', async () => {
    // 顺序不能反。先跳转再清 token 的话,auth store 检查登录状态时手里
    // 还攥着那把废钥匙,于是它认为自己仍然登录着、把用户又弹回原来的页面
    // ——表现为「点了重新登录,还是被踢回来」。
    //
    // 所以这条断言的是**回调里看到的 token 已经是空的**,
    // 而不是「回调被调用了」——后者两种顺序都通过。
    setToken('stale-token')
    let tokenSeenByHandler: string | null | undefined
    setUnauthorizedHandler(() => {
      tokenSeenByHandler = getToken()
    })
    fetchMockReturning(jsonResponse({ detail: '登录状态已失效,请重新登录' }, 401))

    await expect(request('/anything')).rejects.toBeInstanceOf(ApiError)

    expect(tokenSeenByHandler).toBeNull()
    expect(getToken()).toBeNull()
  })

  it('401 抛出来的消息来自后端,不是自己编的', async () => {
    fetchMockReturning(jsonResponse({ detail: '登录状态已失效,请重新登录' }, 401))

    await expect(request('/anything')).rejects.toThrow('登录状态已失效,请重新登录')
  })

  it('**网络层失败包成 ApiError,而不是让 TypeError 漏出去**', async () => {
    // `fetch` 只在**网络层**失败时抛(后端没起、断网)。HTTP 的 4xx/5xx
    // 都走正常返回。不包一层的话,调用方拿到的是一个 `TypeError`,
    // 而它在界面上和「参数写错了」没有任何区别。
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    )

    const error = await request('/anything').catch((e: unknown) => e)

    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).message).toContain('连不上服务器')
  })

  it('**传 FormData 时不手写 Content-Type**', async () => {
    // multipart 的边界串是浏览器生成的。手写一个 `multipart/form-data`
    // 会缺掉 boundary,服务端解析不出任何字段——报的是 422「缺少 files」,
    // 看起来像字段名写错了。
    setToken('t')
    const mock = fetchMockReturning(jsonResponse({}))
    const formData = new FormData()
    formData.append('files', new Blob(['x']), 'a.csv')

    await request('/upload', { method: 'POST', formData })

    const init = mock.mock.calls[0]?.[1]
    expect(init?.headers).not.toHaveProperty('Content-Type')
    expect(init?.body).toBe(formData)
  })
})

describe('notifyUnauthorized', () => {
  it('没注册回调时也不炸', () => {
    // auth store 挂载前、或已经卸载之后,这个函数仍然可能被调用
    // (比如一个迟到的 401)。`onUnauthorized?.()` 少个问号就是
    // 「偶发的 TypeError」,而它只在特定时序下出现。
    setToken('t')

    expect(() => notifyUnauthorized()).not.toThrow()
    expect(getToken()).toBeNull()
  })
})
