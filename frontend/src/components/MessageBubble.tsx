/**
 * 一条消息。
 *
 * **折叠状态放在消息自己身上**,而不是提到页面里去。理由不是省事:
 * 一屏上挂着好几轮问答,每轮各有自己的来源,提到页面里就要维护一个
 * 「哪个消息展开了」的集合、一个「高亮哪一张」的记录,还要处理消息被
 * 删掉之后的清理——而那些状态的生命周期本来就跟着消息走。
 *
 * 点正文里的 `[1]` 也是在这里处理的:展开 + 滚动到对应卡片,两件事必须
 * 一起发生。放到父组件去编排的话,两个 setState 之间隔着一层渲染,
 * 滚动会找不到那个还没渲染出来的元素。
 */

import { useState } from 'react'
import type { SourceRef } from '../api/types'
import { AnswerText } from './AnswerText'
import { SourceDisclosure } from './SourceDisclosure'

export interface MessageBubbleProps {
  role: 'user' | 'assistant'
  content: string
  sources?: SourceRef[]
  /**
   * 这一轮还在流式输出。
   *
   * **为真时不渲染来源**,哪怕 `sources` 已经到了——服务端确实先发
   * `sources` 再流 token,但那是**传输**顺序,不该决定**显示**顺序:
   * 答案还没写完就先铺开一屏来源卡片,会把用户真正在等的东西挤下去。
   */
  streaming?: boolean
  /** 锚点前缀,按消息区分,否则多轮之间会撞 id。见 `SourceCards`。 */
  idPrefix: string
}

export function MessageBubble({
  role,
  content,
  sources = [],
  streaming = false,
  idPrefix,
}: MessageBubbleProps) {
  const [open, setOpen] = useState(false)
  const [highlighted, setHighlighted] = useState<number | null>(null)

  function revealSource(index: number) {
    setOpen(true)
    setHighlighted(index)
    // **等这一帧渲染完再找那个元素。** 展开之前它根本不在 DOM 里,
    // 直接找会拿到 null,而 `?.` 会把这一步静默跳过——用户点了引用,
    // 什么都没发生,也没有任何报错。
    requestAnimationFrame(() => {
      document
        .getElementById(`${idPrefix}-source-${index}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    })
  }

  if (role === 'user') {
    return (
      <div className="msg msg-user">
        <div className="msg-bubble">{content}</div>
      </div>
    )
  }

  return (
    <div className="msg msg-assistant">
      <div className="msg-bubble">
        {content ? (
          <AnswerText content={content} sources={sources} onCiteClick={revealSource} />
        ) : (
          // 一个字都还没到。给一个会呼吸的提示,而不是一片空白——
          // 空白会让人以为没点上「发送」。
          <span className="msg-typing" aria-label="正在生成">
            <i />
            <i />
            <i />
          </span>
        )}
        {!streaming && (
          <SourceDisclosure
            idPrefix={idPrefix}
            sources={sources}
            open={open}
            onOpenChange={setOpen}
            highlighted={highlighted}
          />
        )}
      </div>
    </div>
  )
}
