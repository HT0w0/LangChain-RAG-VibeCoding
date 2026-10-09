import { describe, expect, it } from 'vitest'
import { splitCitations } from './citations'

/**
 * 把答案正文切成「普通文字」和「引用标记」两种片段。
 *
 * **这是引用溯源的核心,也是最容易静默出错的地方。** 切错了不会有任何
 * 报错:页面上一切正常,`[1]` 长得还是 `[1]`,只是点了跳不到对应的来源
 * 卡片,或者跳到了错的那张。用户看不出来——**除非他真的去核对原文**,
 * 而那正是这个产品存在的意义。
 *
 * 判据只有一条:编号必须落在**这次真实装进上下文的资料条数**之内。
 * 上界由服务端算好随 `sources` 一起给,前端不自己数。
 */

const text = (t: string) => ({ kind: 'text', text: t })
const cite = (index: number) => ({ kind: 'citation', index })

describe('splitCitations', () => {
  it('没有标记时原样返回一段文字', () => {
    expect(splitCitations('这是一句普通的话。', 5)).toEqual([
      text('这是一句普通的话。'),
    ])
  })

  it('切出夹在文字中间的引用', () => {
    expect(splitCitations('价格为 4299 元 [1],库存 12 件。', 5)).toEqual([
      text('价格为 4299 元 '),
      cite(1),
      text(',库存 12 件。'),
    ])
  })

  it('连着写多个引用时每个都切出来', () => {
    expect(splitCitations('参见 [1][2]。', 5)).toEqual([
      text('参见 '),
      cite(1),
      cite(2),
      text('。'),
    ])
  })

  it('**超出资料条数的编号不当作引用**', () => {
    // 模型会臆造编号。当成引用的话,点过去会指向一张不存在的卡片——
    // 而它在页面上看起来和真引用一模一样。上界由服务端给,前端不猜。
    expect(splitCitations('成立于 [2024] 年。', 3)).toEqual([
      text('成立于 [2024] 年。'),
    ])
  })

  it('**编号 0 不当作引用**', () => {
    // 引用编号从 1 起。把 [0] 当成引用会去取 `sources[-1]`,拿到最后一条
    // ——一个**看起来完全合理**的错误来源卡片。
    expect(splitCitations('见 [0]。', 3)).toEqual([text('见 [0]。')])
  })

  it('**markdown 链接不被误认成引用**', () => {
    // `[1](https://example.com)` 的方括号里恰好是数字。前处理要是没绕开它,
    // 链接会被拆成「一个引用 + 一段残缺的 (https://...) 文字」。
    //
    // 期望是**一整段文字**:除了那处链接,整句里没有别的东西该被切。
    expect(splitCitations('见 [1](https://example.com)。', 5)).toEqual([
      text('见 [1](https://example.com)。'),
    ])
  })

  it('markdown 链接与真引用同时出现时各自安好', () => {
    expect(splitCitations('见 [1](https://a.com) 和 [2]。', 5)).toEqual([
      text('见 [1](https://a.com) 和 '),
      cite(2),
      text('。'),
    ])
  })

  it('一句里同时出现合法与越界的编号', () => {
    expect(splitCitations('[1] 与 [9]', 3)).toEqual([
      cite(1),
      text(' 与 [9]'),
    ])
  })

  it('上界为 0(没有任何资料)时什么都不切', () => {
    expect(splitCitations('资料中没有找到相关内容 [1]。', 0)).toEqual([
      text('资料中没有找到相关内容 [1]。'),
    ])
  })

  it('不会产出空文字片段', () => {
    // 空片段在渲染层没影响,但它会让「相邻片段合并」这类后续处理
    // 变得难写,而且断言里到处是 `{kind:'text',text:''}` 很吵。
    expect(splitCitations('[1]', 5)).toEqual([cite(1)])
  })

  it('空字符串返回空数组', () => {
    expect(splitCitations('', 5)).toEqual([])
  })
})
