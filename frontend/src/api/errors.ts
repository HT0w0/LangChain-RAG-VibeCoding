/**
 * 把后端错误体翻译成人能读的一句话。
 *
 * FastAPI 的错误体有好几种形状,而**直接把 `detail` 丢进提示框**是最省事
 * 也最糟的写法:422 那一种渲染出来是 `[object Object]`,用户完全无法据此
 * 行动,而它看起来像「系统出错了」而不是「你填错了」。
 */

/** 状态码兜底文案。**每一条都要让人知道下一步能做什么。** */
const FALLBACK: Record<number, string> = {
  401: '登录状态已失效,请重新登录',
  403: '没有权限执行这个操作',
  404: '请求的资源不存在',
  429: '操作太频繁了,请稍后再试',
}

function fallbackFor(status: number): string {
  return FALLBACK[status] ?? '服务端出错了,请稍后重试'
}

function describeFieldError(item: unknown): string | null {
  if (typeof item === 'string') return item
  if (typeof item !== 'object' || item === null) return null

  const record = item as Record<string, unknown>
  const message = record.msg ?? record.message
  if (typeof message !== 'string' || message === '') return null

  // `loc` 是 `["body", "username"]` 这样的路径。`body` 对用户是噪音,
  // 而**只取最后一段**也够用:表单里没有同名字段。
  const loc = record.loc
  const field =
    Array.isArray(loc) && loc.length > 1 ? String(loc[loc.length - 1]) : null

  return field ? `${field}: ${message}` : message
}

/**
 * @param payload 已经解析过的响应体;**解析失败时把原始文本传进来**,
 *   它会被当成「不可解析」处理,不会原样显示给用户。
 */
export function describeError(status: number, payload: unknown): string {
  // **原始字符串一律不回声。** 走到这里说明响应体不是后端那套结构化错误
  // ——最常见的是反向代理挂掉时返回的一整页 HTML。把它显示给用户等于
  // 把一屏 `<html>` 糊在提示框里,而用户需要知道的只是「服务端出错了」。
  if (typeof payload === 'string') return fallbackFor(status)
  if (typeof payload !== 'object' || payload === null) return fallbackFor(status)

  const detail = (payload as Record<string, unknown>).detail

  if (typeof detail === 'string' && detail !== '') return detail

  if (Array.isArray(detail)) {
    const parts = detail.map(describeFieldError).filter((s): s is string => s !== null)
    // 空数组退到兜底,**不能返回空字符串**:空字符串会被提示框渲染成
    // 一个没有内容的框,看起来像「系统卡了一下」。
    return parts.length > 0 ? parts.join(';') : fieldFallback(status)
  }

  // **后端业务异常的形状是 `{code, message, details?}`(见后端 `main.py` 的
  // `AppError` 处理器),它没有 `detail`。**
  //
  // 只认 `detail` 的话,后端写的每一句业务文案都会被丢掉,用户永远只看到
  // 按状态码给的兜底 —— 而这个缺口一直没被发现,是因为**兜底那句看起来
  // 也说得通**。它有信息量、不像报错、也没人会觉得少了什么。
  //
  // 直到限流:兜底那句「操作太频繁了,请稍后再试」**说不出等多久**,
  // 而业务异常里正带着这个数。这才让它显形。
  const message = (payload as Record<string, unknown>).message
  if (typeof message === 'string' && message !== '') return message

  return fallbackFor(status)
}

function fieldFallback(status: number): string {
  // 422 有专门的文案:它几乎总是「填错了」,让人去改表单比让人重试有用。
  return status === 422 ? '提交的内容不合法' : fallbackFor(status)
}
