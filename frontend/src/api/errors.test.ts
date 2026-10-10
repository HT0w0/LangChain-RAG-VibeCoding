import { describe, expect, it } from 'vitest'
import { describeError } from './errors'

/**
 * 把后端错误体翻译成人能读的一句话。
 *
 * **做错了不会崩,只会让用户看到垃圾。** FastAPI 的错误体有好几种形状,
 * 而且**同一种形状在不同状态码下含义不同**:
 *
 *   401 `{"detail": "用户名或密码错误"}`
 *   403 `{"detail": "需要管理员权限"}`
 *   422 `{"detail": [{"loc": ["body","question"], "msg": "..."}]}`
 *
 * 直接把 `detail` 丢进 toast 是最省事的写法,而 422 那条会渲染成
 * `[object Object]`——一个用户完全无法据此行动的提示,而它看起来
 * 像是「系统出错了」,不是「你填错了」。
 */

describe('describeError', () => {
  it('detail 是字符串时直接用', () => {
    expect(describeError(401, { detail: '用户名或密码错误' })).toBe('用户名或密码错误')
  })

  it('**422 的字段错误拼成一句人话**', () => {
    const body = {
      detail: [
        { loc: ['body', 'username'], msg: '用户名需为 3-32 位' },
        { loc: ['body', 'password'], msg: '密码至少 6 位' },
      ],
    }

    expect(describeError(422, body)).toBe(
      'username: 用户名需为 3-32 位;password: 密码至少 6 位',
    )
  })

  it('422 的 loc 去掉 body 前缀,只留字段名', () => {
    // `loc` 是 `["body","username"]` 这样的路径,直接拼出来是
    // 「body.username: ...」——对用户来说 `body` 是噪音。
    //
    // **字段名要留着。** 后端的消息目前都是自解释的中文句子,带上字段名
    // 显得重复;但那是后端的措辞习惯,不是契约——换一条消息写成「不能为空」,
    // 没有字段名就完全不知道说的是哪个框。
    const body = { detail: [{ loc: ['body', 'name'], msg: '不能为空' }] }

    expect(describeError(422, body)).toBe('name: 不能为空')
  })

  it('loc 只有 body 时不硬拼出个空字段名', () => {
    const body = { detail: [{ loc: ['body'], msg: '请求体格式不对' }] }

    expect(describeError(422, body)).toBe('请求体格式不对')
  })

  it('message 字段的变体也认', () => {
    // pydantic v2 的字段是 `msg`,但别的错误来源用的是 `message`。
    // 只认一个的话,另一种形状会退化成兜底文案。
    const body = { detail: [{ loc: ['body', 'x'], message: '长度超限' }] }

    expect(describeError(422, body)).toBe('x: 长度超限')
  })

  it('**没有 detail 时按状态码给一句有信息量的话**', () => {
    // 兜底文案不能是「请求失败」——那等于没说。用户至少该知道
    // 是权限问题、还是服务端问题、还是自己的输入有问题。
    expect(describeError(403, {})).toBe('没有权限执行这个操作')
    expect(describeError(401, {})).toBe('登录状态已失效,请重新登录')
    expect(describeError(500, {})).toBe('服务端出错了,请稍后重试')
  })

  it('**认后端业务异常的形状**(`{code, message, details?}`)', () => {
    // 后端 `main.py` 的 AppError 处理器返回的是 `{code, message, details?}` ——
    // 它**没有 `detail`**。只认 `detail` 的话,后端精心写的每一句业务文案
    // 都会被丢掉,用户看到的是按状态码给的兜底。
    //
    // 这不是措辞问题:兜底那句「操作太频繁了,请稍后再试」说不出
    // **等多久**,而业务异常里带着这句话(见限流)。
    const body = {
      code: 'rate_limited',
      message: '问得太快了,请等 23 秒再试。(限制:30 次/分钟)',
      details: { retry_after_seconds: 23.4 },
    }

    expect(describeError(429, body)).toBe('问得太快了,请等 23 秒再试。(限制:30 次/分钟)')
  })

  it('message 是空串时退到兜底,不给一个空提示框', () => {
    expect(describeError(429, { code: 'rate_limited', message: '' })).toBe(
      '操作太频繁了,请稍后再试',
    )
  })

  it('响应体不是 JSON(比如网关返回的 HTML 错误页)时不崩', () => {
    // 反向代理挂掉时返回的是一页 HTML,`response.json()` 会抛。
    // 让它冒出去的话,用户看到的是「Unexpected token < in JSON」。
    expect(describeError(502, '<html>Bad Gateway</html>')).toBe(
      '服务端出错了,请稍后重试',
    )
  })

  it('detail 是空数组时退到兜底,而不是给个空字符串', () => {
    // 空字符串会被 toast 渲染成一个**没有内容的提示框**——
    // 看起来像「系统卡了一下」,而实际是后端给了个空错误体。
    expect(describeError(422, { detail: [] })).toBe('提交的内容不合法')
  })

  it('detail 是别的类型(数组里套的不是对象)时不崩', () => {
    expect(describeError(422, { detail: ['随便一个字符串'] })).toBe('随便一个字符串')
  })
})
