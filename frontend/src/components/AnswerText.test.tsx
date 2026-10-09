import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { SourceRef } from '../api/types'
import { AnswerText } from './AnswerText'

/**
 * 回答的渲染。
 *
 * 这一层错了**不会报错,只会显示得不对**:`[1]` 变成不可点的纯文本、
 * 或者反过来——本该是字面量的 `arr[1]` 变成了一个链接。两种都不会
 * 让任何东西崩,用户也不会去核对。
 */

function source(index: number): SourceRef {
  return {
    index,
    chunk_id: index,
    doc_id: 1,
    doc_name: '报价.csv',
    kb_id: 1,
    kb_name: '商品库',
    chunk_index: 0,
    content: 'RX9070 显卡,价格 4299 元',
  }
}

const sources = [source(1), source(2)]

describe('AnswerText', () => {
  it('把正文里的引用渲染成可点的标记', () => {
    render(<AnswerText content="价格为 4299 元 [1]。" sources={sources} />)

    expect(screen.getByRole('button', { name: '查看第 1 条来源' })).toBeInTheDocument()
  })

  it('**超出资料条数的编号不渲染成链接**', () => {
    // 模型会臆造编号。渲染成链接的话,点过去找不到对应卡片——
    // 而它在页面上和真引用长得一模一样。
    render(<AnswerText content="成立于 [2024] 年。" sources={sources} />)

    expect(screen.queryByRole('button')).not.toBeInTheDocument()
    expect(screen.getByText(/\[2024\]/)).toBeInTheDocument()
  })

  it('**行内代码里的方括号保持字面量**', () => {
    // 这是「改 markdown 源码」那种做法会错的地方:`arr[1]` 会被当成引用。
    // 走语法树才能把它和真正的引用区分开。
    render(<AnswerText content={'取值写成 `arr[1]` 即可。'} sources={sources} />)

    expect(screen.queryByRole('button')).not.toBeInTheDocument()
    expect(screen.getByText(/arr\[1\]/)).toBeInTheDocument()
  })

  it('点击引用把编号传给回调', async () => {
    const onCiteClick = vi.fn()
    render(
      <AnswerText content="见 [2]。" sources={sources} onCiteClick={onCiteClick} />,
    )

    await userEvent.click(screen.getByRole('button', { name: '查看第 2 条来源' }))

    expect(onCiteClick).toHaveBeenCalledWith(2)
  })

  it('markdown 链接不会被变成引用标记', () => {
    render(<AnswerText content="见 [1](https://example.com)。" sources={sources} />)

    expect(screen.queryByRole('button')).not.toBeInTheDocument()
    expect(screen.getByRole('link')).toHaveAttribute('href', 'https://example.com')
  })
})
