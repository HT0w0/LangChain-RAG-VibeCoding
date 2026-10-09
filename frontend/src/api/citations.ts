/**
 * 把答案正文切成「普通文字」和「引用标记」两种片段。
 *
 * **引用溯源的核心,也是最容易静默出错的地方。** 切错了不会有任何报错:
 * 页面上一切正常,`[1]` 长得还是 `[1]`,只是点了跳不到对应的来源卡片,
 * 或者跳到了错的那一张。用户看不出来——除非他真的去核对原文,
 * 而那正是这个产品存在的意义。
 */

export type AnswerSegment =
  | { kind: 'text'; text: string }
  | { kind: 'citation'; index: number }

//: `(?!\()` 是关键:markdown 链接 `[1](https://…)` 的方括号里恰好也是数字。
//: 不排掉它的话,一个链接会被拆成「一个引用 + 一段残缺的 `(https://…)` 文字」。
//: 后端的 `extract_citations` 用的是同一条规则,两边必须一致——
//: 不一致的表现是「正文里的引用数和服务端算出来的对不上」。
const CITATION_RE = /\[(\d+)\](?!\()/g

/**
 * @param maxIndex 这次**真实装进上下文**的资料条数,由服务端的 `sources` 给。
 *   越界的编号一律不当作引用:模型会臆造编号(还有 `[2024]` 这种年份),
 *   把它们当成引用会指向一张不存在的卡片——而它在页面上和真引用一模一样。
 */
export function splitCitations(text: string, maxIndex: number): AnswerSegment[] {
  const out: AnswerSegment[] = []
  let cursor = 0

  for (const match of text.matchAll(CITATION_RE)) {
    const index = Number(match[1])
    // 越界的直接跳过,**并且不动 cursor**——那段文字要原样留在后面的片段里,
    // 不能因为它「不是引用」就把它从正文里删掉。模型写错了,用户得看见。
    if (index < 1 || index > maxIndex) continue

    const start = match.index
    if (start > cursor) out.push({ kind: 'text', text: text.slice(cursor, start) })
    out.push({ kind: 'citation', index })
    cursor = start + match[0].length
  }

  // 不产出空片段:空片段在渲染上没有影响,但它让下游的「合并相邻文字」
  // 变得难写,也让断言里到处是 `{kind:'text',text:''}` 这种噪音。
  if (cursor < text.length) out.push({ kind: 'text', text: text.slice(cursor) })

  return out
}
