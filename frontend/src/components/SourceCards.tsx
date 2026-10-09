/**
 * 来源卡片。
 *
 * 这是整个产品存在的理由:**答案必须能被核对**。所以卡片默认展开正文
 * 而不是折叠起来——点开才看到原文,等于把「核对」这个动作藏起来,
 * 而用户不做核对的系统,和一个会编的系统没有区别。
 *
 * 卡片的顺序**必须和编号一致**,不能被「分数高的排前面」之类的排序动过。
 * 正文里的 `[1]` 指向的就是这个列表的第 1 项,排一下就全错位了,而且
 * 错得毫无痕迹——每张卡片看起来都对。
 */

import { Card, Empty, Space, Tag, Typography } from 'antd'
import type { SourceRef } from '../api/types'

export interface SourceCardsProps {
  sources: SourceRef[]
  /** 被点开的引用编号(用于滚动定位与高亮)。 */
  highlighted?: number | null
}

function scoreLabel(source: SourceRef): string | null {
  // 优先显示重排分:它是最终用来排序和截断的那个数。向量相似度和 BM25
  // 分是两路各自尺度的原始值,放在一起会让人误以为可以横向比较。
  if (source.rerank_score != null) return `重排 ${source.rerank_score.toFixed(4)}`
  if (source.fusion_score != null) return `融合 ${source.fusion_score.toFixed(4)}`
  return null
}

export function SourceCards({ sources, highlighted }: SourceCardsProps) {
  if (sources.length === 0) {
    return <Empty description="这一轮没有引用任何资料" image={Empty.PRESENTED_IMAGE_SIMPLE} />
  }

  return (
    <Space orientation="vertical" style={{ width: '100%' }} size={8}>
      {sources.map((source) => {
        const active = highlighted === source.index
        return (
          <Card
            key={source.index}
            // 滚动锚点。**用 `index` 而不是 `chunk_id`**:正文里的 `[1]`
            // 指的是一次回答内的序号,同一个分块在下一轮可能是别的编号。
            id={`source-${source.index}`}
            size="small"
            style={active ? { borderColor: '#1677ff', background: '#f0f7ff' } : undefined}
            title={
              <Space size={6} wrap>
                <Tag color="blue">{source.index}</Tag>
                <Typography.Text strong>{source.doc_name}</Typography.Text>
                <Typography.Text type="secondary">{source.kb_name}</Typography.Text>
              </Space>
            }
          >
            <Space size={4} wrap style={{ marginBottom: 8 }}>
              {source.category && <Tag>{source.category}</Tag>}
              {source.price != null && <Tag color="orange">¥{source.price}</Tag>}
              {scoreLabel(source) && (
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {scoreLabel(source)}
                </Typography.Text>
              )}
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                第 {source.chunk_index + 1} 块
              </Typography.Text>
            </Space>
            {/* `pre-wrap` 是必须的:分块正文里有换行,不保留的话整段会被
                压成一行,而表格类的资料压成一行之后就没法读了。 */}
            <Typography.Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>
              {source.content}
            </Typography.Paragraph>
          </Card>
        )
      })}
    </Space>
  )
}
