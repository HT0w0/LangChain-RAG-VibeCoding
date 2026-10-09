/**
 * 渲染助手回答:markdown + 可点击的引用标记。
 *
 * **引用标记是在 markdown 的语法树上替换的,不是拿正则去改 markdown 源码。**
 * 后者看起来简单得多,但它会把代码块里的 `arr[1]` 也变成一个链接——
 * 而"这个项目的回答里不会有代码"是个**假设**,不是一个保证。改源码的
 * 做法在假设成立时完全正确,所以它不会在开发时暴露问题。
 */

import { useMemo } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { splitCitations } from '../api/citations'
import type { SourceRef } from '../api/types'

//: 自定义标签名。用 `cite-ref` 而不是 `span` + 类名,是为了让组件映射
//: 不必判断「这个 span 是引用还是别的什么」——页面里别的 span 不是我的,
//: 不该被我的组件接管。
const CITE_TAG = 'cite-ref'

interface HastNode {
  type: string
  value?: string
  tagName?: string
  properties?: Record<string, unknown>
  children?: HastNode[]
}

function textToNodes(text: string, maxIndex: number): HastNode[] {
  return splitCitations(text, maxIndex).map((segment) =>
    segment.kind === 'text'
      ? { type: 'text', value: segment.text }
      : {
          type: 'element',
          tagName: CITE_TAG,
          properties: { dataIndex: segment.index },
          children: [{ type: 'text', value: String(segment.index) }],
        },
  )
}

function rewrite(node: HastNode, maxIndex: number): void {
  if (!node.children) return

  const next: HastNode[] = []
  for (const child of node.children) {
    if (child.type === 'text' && typeof child.value === 'string') {
      next.push(...textToNodes(child.value, maxIndex))
      continue
    }
    // **代码块与行内代码里的一切都是字面量。** 不走进它们,`arr[1]`
    // 就还是 `arr[1]`。
    if (child.tagName !== 'code' && child.tagName !== 'pre') rewrite(child, maxIndex)
    next.push(child)
  }
  node.children = next
}

export interface AnswerTextProps {
  content: string
  /** 这次真实装进上下文的资料。**上界只能由服务端给**,前端不自己数。 */
  sources: SourceRef[]
  /** 点引用标记时回调,通常用来滚动并展开对应的来源卡片。 */
  onCiteClick?: (index: number) => void
}

export function AnswerText({ content, sources, onCiteClick }: AnswerTextProps) {
  const maxIndex = sources.length

  // `useMemo` 有两层作用:
  //   * 上界作为选项传进插件,而不是用闭包捕获——否则 maxIndex 一变
  //     就有个新的插件函数,而 `react-markdown` 会因此重解析整棵树。
  //   * 组件映射带着 `onCiteClick` 进闭包,不必为了「让对象稳定」而
  //     把回调拆成全局事件或者 context——那些做法为了让一个常量不变,
  //     把父子关系换成了隐式的。
  const plugins = useMemo(() => [[citationPlugin, maxIndex] as const], [maxIndex])
  const components = useMemo(
    () =>
      ({
        [CITE_TAG]: ({ node }: { node?: HastNode }) => {
          const index = Number(node?.properties?.dataIndex ?? 0)
          return (
            <button
              type="button"
              className="cite-chip"
              onClick={() => onCiteClick?.(index)}
              aria-label={`查看第 ${index} 条来源`}
            >
              {index}
            </button>
          )
        },
      }) as Components,
    [onCiteClick],
  )

  return (
    <div className="answer-text">
      <Markdown
        remarkPlugins={[remarkGfm]}
        // `react-markdown` 默认**不渲染原始 HTML**,这一点不要放开:
        // 回答里可能夹着被检索到的资料原文,而资料是用户上传的文件。
        rehypePlugins={plugins as never}
        components={components}
      >
        {content}
      </Markdown>
    </div>
  )
}

//: `(maxIndex) => Transformer`。写成工厂而不是常量插件,是因为上界
//: 来自服务端、每次回答都可能不同。
function citationPlugin(maxIndex: number) {
  return (tree: HastNode) => {
    rewrite(tree, maxIndex)
  }
}
