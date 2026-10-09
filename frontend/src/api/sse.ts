/**
 * SSE 分帧。
 *
 * **为什么不用 `EventSource`**:它不能带自定义请求头,而这套接口全都要
 * `Authorization: Bearer`;它还只能发 GET,而问答是 POST 带 JSON 体。
 * 所以走 `fetch` + `ReadableStream`,自己把字节流切成事件。
 *
 * 分包边界和帧边界**毫无关系**:一帧被劈成两半是常态而不是异常,而
 * 分帧写错的症状是「答案偶尔少半句」或「流永远不结束」——不报错,
 * 只是不对。这个文件里每一处看起来多余的宽容都对应一种真实输入。
 */

export interface RawSseEvent {
  event: string
  data: string
}

//: 帧的分隔:一个空行。三种换行都认。
const FRAME_END = /\r\n\r\n|\n\n|\r\r/

/** 行内换行:`\r\n` 必须排在 `\r` 前面,否则 `\r\n` 会被拆成两行。 */
const LINE_END = /\r\n|\n|\r/

function parseFrame(raw: string): RawSseEvent | null {
  let name: string | null = null
  const dataLines: string[] = []

  for (const line of raw.split(LINE_END)) {
    // 空行不算字段;`:` 开头的是注释(服务端的保活心跳),按协议必须忽略。
    // 把它当成一帧的话,`JSON.parse` 会炸,而症状是「偶尔弹一次解析错误」。
    if (line === '' || line.startsWith(':')) continue

    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? '' : line.slice(colon + 1)

    // **只吃掉一个前导空格。** 这是 SSE 协议规定的,而多吃一个会让
    // `data:  {"a": 1}` 里的空格消失——JSON 仍然合法,内容仍然错。
    if (value.startsWith(' ')) value = value.slice(1)

    if (field === 'event') name = value
    else if (field === 'data') dataLines.push(value)
  }

  // 只有 `event:` 没有 `data:` 的帧不产生事件。返回一个 data 为空串的
  // 事件会往上冒一层「解析了的 JSON 是空串」的错误,离根因很远。
  if (dataLines.length === 0) return null

  return { event: name ?? 'message', data: dataLines.join('\n') }
}

export class SseDecoder {
  private buffer = ''

  /** 送入一段文本,返回**这次能凑齐的所有**完整帧。 */
  push(chunk: string): RawSseEvent[] {
    this.buffer += chunk
    const out: RawSseEvent[] = []

    for (;;) {
      const found = FRAME_END.exec(this.buffer)
      if (!found) break

      const raw = this.buffer.slice(0, found.index)
      this.buffer = this.buffer.slice(found.index + found[0].length)

      const parsed = parseFrame(raw)
      if (parsed) out.push(parsed)
    }

    return out
  }

  /**
   * 连接结束时吐出缓冲区里剩下的那一帧。
   *
   * **不能省。** 服务端干净地关掉连接时,最后一帧后面可能没有空行,
   * 而那通常是 `done` 事件——前端判断「这轮结束了」的唯一依据。
   * 丢掉它的症状是「回答完了却一直转圈」。
   */
  flush(): RawSseEvent[] {
    const raw = this.buffer
    this.buffer = ''
    const parsed = raw ? parseFrame(raw) : null
    return parsed ? [parsed] : []
  }
}

/**
 * 把一个 `ReadableStream` 读成事件流。
 *
 * `TextDecoder` 必须以 `{stream: true}` 增量解码:一个中文字符占三个字节,
 * 跨在两个网络块之间是常事。不用流式解码的话,那半个字符会变成 U+FFFD
 * ——**替换字符不会报错**,只会让答案里偶尔冒出一个问号或方块。
 */
export async function* readSse(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<RawSseEvent> {
  const reader = body.getReader()
  const textDecoder = new TextDecoder('utf-8')
  const decoder = new SseDecoder()

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      for (const event of decoder.push(textDecoder.decode(value, { stream: true }))) {
        yield event
      }
    }
    // 把解码器内部残留的半个字符冲出来,再冲分帧缓冲区。
    for (const event of decoder.push(textDecoder.decode())) yield event
    for (const event of decoder.flush()) yield event
  } finally {
    // 取消请求(用户点了「停止」)时 `read()` 会抛,而**不释放锁会让这条
    // 连接一直挂着**——重复几次就把浏览器的并发连接额度耗光,
    // 表现为「点了几次停止之后,页面再也发不出请求」。
    reader.releaseLock()
  }
}
