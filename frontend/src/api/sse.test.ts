import { describe, expect, it } from 'vitest'
import { SseDecoder, readSse } from './sse'

/**
 * SSE 分帧。
 *
 * **这里测的是「静默出错」的那一类。** 分帧写错的症状不是白屏或报错,
 * 而是:答案少半句、引用对不上号、或者流永远不结束。而它平时看起来
 * 完全正常——直到某个请求的答案恰好长到跨过了网络分包的边界。
 *
 * 所以下面每一条都对应一个**真实会发生的**输入形状,而不是凑覆盖率。
 */

const decoder = () => new SseDecoder()

describe('SseDecoder', () => {
  it('解析一帧完整的报文', () => {
    const events = decoder().push('event: token\ndata: {"text":"你好"}\n\n')

    expect(events).toEqual([{ event: 'token', data: '{"text":"你好"}' }])
  })

  it('**一帧被拆到两个 chunk 里也要能拼回来**', () => {
    // 这是整条链路上最容易出错的一格:网络分包的位置和 SSE 的帧边界
    // 毫无关系,一帧被劈成两半是常态而不是异常。
    const d = decoder()

    expect(d.push('event: tok')).toEqual([])
    expect(d.push('en\ndata: {"te')).toEqual([])
    expect(d.push('xt":"你好"}\n\n')).toEqual([
      { event: 'token', data: '{"text":"你好"}' },
    ])
  })

  it('一个 chunk 里有多帧时全部返回', () => {
    const events = decoder().push(
      'event: token\ndata: {"text":"a"}\n\nevent: token\ndata: {"text":"b"}\n\n',
    )

    expect(events.map((e) => e.data)).toEqual(['{"text":"a"}', '{"text":"b"}'])
  })

  it('接受 CRLF 换行', () => {
    // sse-starlette 实际发出来的就是 `\r\n`。只认 `\n` 的话,`data:` 后面
    // 会残留一个 `\r`,而 JSON.parse 能容忍它——所以这个错**不会报**,
    // 直到某个字段是字符串末尾带 \r 的时候才显形。
    const events = decoder().push('event: token\r\ndata: {"text":"a"}\r\n\r\n')

    expect(events).toEqual([{ event: 'token', data: '{"text":"a"}' }])
  })

  it('忽略心跳注释行', () => {
    // 服务端会定期发 `: ping` 之类保持连接。把它当成一帧的话,
    // JSON.parse 会炸,而症状是「偶尔弹一次解析错误」。
    const events = decoder().push(': ping\n\nevent: token\ndata: {"text":"a"}\n\n')

    expect(events).toEqual([{ event: 'token', data: '{"text":"a"}' }])
  })

  it('没有 event 名时默认叫 message', () => {
    const events = decoder().push('data: {"text":"a"}\n\n')

    expect(events[0].event).toBe('message')
  })

  it('多行 data 用换行拼起来', () => {
    // SSE 协议规定如此。后端目前总是单行 `data:`,但这是协议的一部分,
    // 而且**断行的那一侧看起来和正常报文一模一样**。
    const events = decoder().push('event: token\ndata: {"a":1,\ndata: "b":2}\n\n')

    expect(events[0].data).toBe('{"a":1,\n"b":2}')
  })

  it('flush 吐出结尾没有空行的那一帧', () => {
    // 连接被干净地关掉时,最后一帧后面可能没有空行。丢掉它就是丢掉
    // `done` 事件——而 `done` 是前端判断「这轮结束了」的唯一依据,
    // 症状是回答完了却一直转圈。
    const d = decoder()
    d.push('event: done\ndata: {"answer":"好了"}')

    expect(d.flush()).toEqual([{ event: 'done', data: '{"answer":"好了"}' }])
  })

  it('flush 之后不会重复吐出已经发过的帧', () => {
    const d = decoder()
    d.push('event: token\ndata: {"text":"a"}\n\n')

    expect(d.flush()).toEqual([])
  })

  it('空行不产生事件', () => {
    expect(decoder().push('\n\n\n')).toEqual([])
  })

  it('冒号后的一个空格被去掉,第二个保留', () => {
    // `data:  x` 的值是 ` x`。SSE 规范只吃掉一个前导空格,而多吃一个
    // 会让 JSON 里字符串开头的空格消失——仍然合法,仍然错。
    const events = decoder().push('event: token\ndata:  {"text":"  缩进"}\n\n')

    expect(events[0].data).toBe(' {"text":"  缩进"}')
  })
})

describe('readSse', () => {
  async function collect(stream: ReadableStream<Uint8Array>) {
    const out = []
    for await (const event of readSse(stream)) out.push(event)
    return out
  }

  function streamOf(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
    return new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk)
        controller.close()
      },
    })
  }

  it('**跨字节块的中文字符不会被解成替换字符**', async () => {
    // 一个中文字符占三个字节,而网络块边界落在这三个字节中间是常事。
    // 不用增量解码的话那半个字符会变成 U+FFFD——**替换字符不报错**,
    // 只是让答案里偶尔冒出一个方块或问号,而且只在特定分包下出现。
    const payload = new TextEncoder().encode('event: token\ndata: {"text":"你好"}\n\n')
    const cut = payload.indexOf(0xe4) + 1 // 「你」= E4 BD A0,切在第一个字节后

    const events = await collect(streamOf([payload.slice(0, cut), payload.slice(cut)]))

    expect(events).toEqual([{ event: 'token', data: '{"text":"你好"}' }])
  })

  it('流干净结束时最后一帧不丢', async () => {
    // 服务端关连接时最后一帧后面常常没有空行。丢了 `done` 的症状是
    // 「回答完了却一直转圈」。
    const payload = new TextEncoder().encode('event: done\ndata: {"answer":"好了"}')

    const events = await collect(streamOf([payload]))

    expect(events).toEqual([{ event: 'done', data: '{"answer":"好了"}' }])
  })
})
