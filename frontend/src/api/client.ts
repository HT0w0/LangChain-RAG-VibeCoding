/**
 * 唯一的 HTTP 出口。
 *
 * 两件事必须**只在这里做一次**,散到各个页面里做的话迟早会漏:
 *
 * 1. **带 token。** 漏一处的症状是那个页面随机报 401,而登录明明是好的。
 * 2. **401 时清掉身份并跳登录。** 这是「token 过期」的唯一正确处置——
 *    留在页面上继续发请求的话,用户看到的是每个操作都失败、但界面看起来
 *    还是登录状态,完全不知道该重新登录。
 */

import { describeError } from './errors'

/** 开发期由 Vite 代理到 127.0.0.1:8000,生产期通常同源部署。 */
export const API_BASE = '/api'

const TOKEN_KEY = 'rag.token'

/**
 * token 存在 `localStorage`,**不是内存**。
 *
 * 内存里的 token 一刷新页面就没了,用户每按一次 F5 都要重新登录。
 * 代价是 XSS 能偷到它——本项目已经在渲染 markdown 时关掉了原始 HTML,
 * 而彻底的做法是 httpOnly cookie + CSRF token,那是部署形态的问题,
 * 单机演示下不划算(见 README 的「已知优化点」)。
 */
export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string | null): void {
  if (token === null) localStorage.removeItem(TOKEN_KEY)
  else localStorage.setItem(TOKEN_KEY, token)
}

/** 401 的回调。由 auth store 注册,避免这里 import 它造成循环依赖。 */
let onUnauthorized: (() => void) | null = null

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  onUnauthorized = handler
}

/**
 * 401 的统一处置:**清掉 token,再通知上层跳登录。**
 *
 * 顺序不能反。先跳转再清 token 的话,auth store 检查登录状态时手里
 * 还攥着那把废钥匙,于是它会认为自己仍然登录着、把用户又弹回去。
 *
 * 聊天那条 SSE 也要调它——那里不走 `request()`,漏掉的话表现是
 * 「token 过期后,问答页静默失败而别的页面会跳登录」。
 */
export function notifyUnauthorized(): void {
  setToken(null)
  onUnauthorized?.()
}

export class ApiError extends Error {
  // 见 `chat.ts::ChatError`:`erasableSyntaxOnly` 不允许构造函数参数属性。
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function readBody(response: Response): Promise<unknown> {
  // 204 / 空响应体没有内容可解析。`response.json()` 在这里会抛,
  // 而调用方只关心「成功了」——把它当成 null 而不是异常。
  const raw = await response.text()
  if (raw === '') return null
  try {
    return JSON.parse(raw)
  } catch {
    // 解析失败时**把原文交出去**,由 describeError 决定怎么处置:
    // 它不会把这段原文回显给用户(那可能是一整页 HTML)。
    return raw
  }
}

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE'
  body?: unknown
  signal?: AbortSignal
  /** multipart 上传时传 FormData,跳过 JSON 序列化与 Content-Type。 */
  formData?: FormData
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, signal, formData } = options

  const headers: Record<string, string> = {}
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`
  // **不要给 FormData 手动设 Content-Type。** 边界串是浏览器生成的,
  // 手写一个 `multipart/form-data` 会缺掉 boundary,服务端解析不出任何字段。
  if (formData === undefined && body !== undefined) {
    headers['Content-Type'] = 'application/json'
  }

  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: formData ?? (body === undefined ? undefined : JSON.stringify(body)),
      signal,
    })
  } catch (error) {
    // `fetch` 只在**网络层**失败时抛(后端没起、跨域被拦、断网)。
    // HTTP 层面的 4xx/5xx 都走正常返回,所以这里必须自己包一层,
    // 否则调用方拿到的是一个 TypeError,和「参数写错」长得一样。
    if ((error as Error)?.name === 'AbortError') throw error
    throw new ApiError(0, '连不上服务器,请确认后端已启动')
  }

  const payload = await readBody(response)

  if (!response.ok) {
    if (response.status === 401) notifyUnauthorized()
    throw new ApiError(response.status, describeError(response.status, payload))
  }

  return payload as T
}
