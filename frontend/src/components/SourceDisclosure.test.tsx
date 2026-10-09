import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { SourceRef } from '../api/types'
import { SourceDisclosure } from './SourceDisclosure'

/**
 * 来源的折叠区。
 *
 * **默认折叠是刻意的**:回答本身是主角,一屏挂五六张来源卡片会把正文挤到
 * 看不见。而折叠起来之后,「答案到底依据了什么」这件事就只剩一个入口——
 * 所以那个入口必须**看得见、数得清**(写「引用了 3 条资料」而不是「来源」),
 * 否则用户不会去点,而引用溯源正是这个产品存在的理由。
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
    content: `RX9070 显卡的售价是 4299 元(第 ${index} 条)`,
  }
}

const sources = [source(1), source(2), source(3)]

describe('SourceDisclosure', () => {
  it('**默认折叠**:来源的正文一个字都不渲染出来', () => {
    // 不是「藏起来」而是「不渲染」——用 CSS 藏的话,正文仍然在 DOM 里,
    // 屏幕阅读器和「Ctrl+F 找价格」都会命中一段用户以为没显示的内容。
    render(<SourceDisclosure idPrefix="m1" sources={sources} />)

    expect(screen.queryAllByText(/4299/)).toHaveLength(0)
  })

  it('折叠时也要告诉用户有几条', () => {
    // 「来源」两个字给不出任何信息量,用户不知道点开是三条还是三十条,
    // 于是不会点。带上条数,它才是一个**值得点的入口**。
    render(<SourceDisclosure idPrefix="m1" sources={sources} />)

    expect(screen.getByRole('button', { name: /引用了 3 条资料/ })).toBeInTheDocument()
  })

  it('点一下展开,再点一下收起', async () => {
    render(<SourceDisclosure idPrefix="m1" sources={sources} />)

    await userEvent.click(screen.getByRole('button', { name: /引用了 3 条资料/ }))
    expect(screen.getAllByText(/4299/)).toHaveLength(3)

    await userEvent.click(screen.getByRole('button', { name: /引用了 3 条资料/ }))
    expect(screen.queryAllByText(/4299/)).toHaveLength(0)
  })

  it('`open` 受控:外部可以直接把它打开', () => {
    // 点正文里的 `[1]` 要能展开它并定位到那一张。那是**受控**的用法,
    // 组件自己管状态的话,父组件就没有办法从外面把它撑开。
    render(<SourceDisclosure idPrefix="m1" sources={sources} open />)

    expect(screen.getAllByText(/4299/)).toHaveLength(3)
  })

  it('受控时点击会回调,而不是自己改状态', async () => {
    const onOpenChange = vi.fn()
    render(<SourceDisclosure idPrefix="m1" sources={sources} open={false} onOpenChange={onOpenChange} />)

    await userEvent.click(screen.getByRole('button', { name: /引用了 3 条资料/ }))

    expect(onOpenChange).toHaveBeenCalledWith(true)
    expect(screen.queryByText(/4299/)).not.toBeInTheDocument()
  })

  it('没有来源时整个折叠区不出现', () => {
    // 一轮没有依据的回答(比如「资料中没有找到相关内容」)下面挂一个
    // 「引用了 0 条资料」是可笑的,而且会让人以为检索出了问题。
    render(<SourceDisclosure idPrefix="m1" sources={[]} />)

    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('展开状态下也报对数', () => {
    render(<SourceDisclosure idPrefix="m1" sources={sources} open />)

    expect(screen.getByRole('button', { name: /引用了 3 条资料/ })).toBeInTheDocument()
  })
})
