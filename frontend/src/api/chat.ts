/**
 * 问答接口的 SSE 消费端。
 *
 * **必须用 `fetch` 而不是 `EventSource`**:后者不能带 `Authorization` 头,
 * 只能发 GET,而问答是 POST 带 JSON 体。代价是分帧要自己做(见 `sse.ts`)。
 */

import { ApiError, API_BASE, getToken, notifyUnauthorized } from './client'
import { describeError } from './errors'
import { readSse } from './sse'
import type { ChatRequest, DoneEvent, ErrorEvent, SourcesEvent } from './types'

export interface ChatHandlers {
  onSources?: (event: SourcesEvent) => void
  onToken?: (text: string) => void
  onDone?: (event: DoneEvent) => void
  /** 生成中途失败。注意这只是**两种失败之一**,另一种见下面 `response.ok` 那段。 */
  onError?: (event: ErrorEvent) => void
}

export class ChatError extends Error {
  // **字段显式声明,不用构造函数参数属性。** TS 6 的脚手架开了
  // `erasableSyntaxOnly`,而参数属性(`constructor(readonly x: T)`)是
  // 有运行时行为的语法糖——它和「类型只在编译期存在」的前提冲突,
  // 于是 `tsc` 直接报 TS1294。
  readonly status: number
  readonly code: string

  constructor(message: string, status = 0, code = '') {
    super(message)
    this.name = 'ChatError'
    this.status = status
    this.code = code
  }
}

/**
 * 发起一轮问答,把事件逐个交给回调。
 *
 * **两条失败路径,都要处理:**
 *
 * 1. **响应头发出之前失败** —— 后端会返回一个正常的 JSON 错误,带正确的
 *    状态码(`403`/`422`/`502`)。这时响应体根本不是事件流,拿 `readSse`
 *    去读会得到零个事件、静默结束,界面表现为「点了发送,什么都没发生」。
 * 2. **流已经开始之后失败** —— 只能走 `error` 事件,因为响应头已经提交,
 *    状态码改不了了(这正是后端那段「检索先做完再开流」的原因)。
 *
 * 取消(用户点「停止」)会让 `fetch`/`read()` 抛 `AbortError`,那是正常
 * 路径,**不要当成错误弹提示**。
 */
export async function streamChat(
  payload: ChatRequest,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`

  let response: Response
  try {
    response = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal,
    })
  } catch (error) {
    if ((error as Error)?.name === 'AbortError') return
    throw new ChatError('连不上服务器,请确认后端已启动')
  }

  if (!response.ok) {
    // 走到这里说明**流还没开始**。响应体是 JSON,不是事件流——
    // 交给 `describeError` 翻译,和普通接口走同一条路径。
    if (response.status === 401) notifyUnauthorized()
    const raw = await response.text()
    let parsed: unknown = raw
    try {
      parsed = raw === '' ? null : JSON.parse(raw)
    } catch {
      /* 保持原文,describeError 会兜底 */
    }
    throw new ApiError(response.status, describeError(response.status, parsed))
  }

  if (response.body === null) {
    throw new ChatError('服务端没有返回响应体')
  }

  for await (const frame of readSse(response.body)) {
    // 解析失败只影响这一帧。整条流因为一帧坏掉而中断的话,用户会看到
    // 「答案突然停在半句」,而真正的原因是一个不该出现的帧。
    let data: unknown
    try {
      data = JSON.parse(frame.data)
    } catch {
      continue
    }

    switch (frame.event) {
      case 'sources':
        handlers.onSources?.(data as SourcesEvent)
        break
      case 'token':
        handlers.onToken?.((data as { text: string }).text)
        break
      case 'done':
        handlers.onDone?.(data as DoneEvent)
        break
      case 'error': {
        const event = data as ErrorEvent
        handlers.onError?.(event)
        break
      }
      default:
        // 未知事件名**忽略而不是报错**:服务端将来加一类事件(比如
        // 「正在检索」的进度),旧前端不该因此整轮崩掉。
        break
    }
  }
}
