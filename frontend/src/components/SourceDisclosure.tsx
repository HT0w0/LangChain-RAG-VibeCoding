/**
 * 来源的折叠区。
 *
 * **默认折叠**:回答本身是主角,一屏挂五六张来源卡片会把正文挤到看不见。
 *
 * 但折叠之后,「答案到底依据了什么」就只剩一个入口——所以那个入口必须
 * **看得见、数得清**:标题写「引用了 3 条资料」而不是「来源」,因为
 * 「来源」两个字给不出任何信息量,用户不知道点开是三条还是三十条,
 * 于是不会去点。而引用溯源正是这个产品存在的理由。
 *
 * **受控而不是受托管**:点正文里的 `[1]` 要能把它撑开并定位到那一张,
 * 那需要父组件能从外面打开它。
 */

import { DownOutlined, UpOutlined } from '@ant-design/icons'
import { useState } from 'react'
import type { SourceRef } from '../api/types'
import { SourceCards } from './SourceCards'

export interface SourceDisclosureProps {
  sources: SourceRef[]
  /** 传了就是受控,没传就是自己管。 */
  open?: boolean
  onOpenChange?: (open: boolean) => void
  /** 展开后高亮哪一张(由正文里点 `[1]` 触发)。 */
  highlighted?: number | null
  /** 锚点前缀,按消息区分,否则多轮之间会撞 id。见 `SourceCards`。 */
  idPrefix: string
}

export function SourceDisclosure({
  sources,
  open,
  onOpenChange,
  highlighted,
  idPrefix,
}: SourceDisclosureProps) {
  const [uncontrolled, setUncontrolled] = useState(false)
  const isOpen = open ?? uncontrolled

  // 没有来源时整个入口都不出现。挂一个「引用了 0 条资料」是可笑的,
  // 而且会让人以为检索出了问题。
  if (sources.length === 0) return null

  function toggle() {
    const next = !isOpen
    if (open === undefined) setUncontrolled(next)
    onOpenChange?.(next)
  }

  return (
    <div className={`source-disclosure${isOpen ? ' source-disclosure-open' : ''}`}>
      <button
        type="button"
        className="source-disclosure-toggle"
        // `aria-expanded` 不只是给屏幕阅读器看的:测试也靠它判断状态,
        // 而内联样式或 class 都不是可断言的「状态」。
        aria-expanded={isOpen}
        onClick={toggle}
      >
        {isOpen ? <UpOutlined /> : <DownOutlined />}
        <span>引用了 {sources.length} 条资料</span>
      </button>
      {isOpen && (
        <div className="source-disclosure-body">
          <SourceCards sources={sources} highlighted={highlighted} idPrefix={idPrefix} />
        </div>
      )}
    </div>
  )
}
