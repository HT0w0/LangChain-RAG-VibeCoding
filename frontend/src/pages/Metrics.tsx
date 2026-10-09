/**
 * 请求指标面板。
 *
 * 几处刻意的显示决定,都是为了让「没有数据」和「数据是 0」不会长得一样:
 *
 * * **首 token 无记录时显示 `—`,不是 `0ms`。** `0` 会被读成「首 token 瞬间
 *   出来」——一个漂亮得离谱的数字,而事实是这一轮一个字都没流出来。
 * * **prompt 缓存命中率同理。** `null` 是「平台没报这个读数」,`0` 是
 *   「平台说一次都没命中」。混在一起显示的话,缓存坏了和读数没接上
 *   看起来是一样的。
 */

import { Alert, Card, Col, Descriptions, Row, Segmented, Space, Statistic, Table, Tag, Typography } from 'antd'
import { useCallback, useEffect, useState } from 'react'
import { ApiError } from '../api/client'
import { adminApi } from '../api/endpoints'
import type { MetricsSummary } from '../api/types'

const WINDOWS = [
  { label: '最近 1 小时', value: 60 },
  { label: '最近 6 小时', value: 360 },
  { label: '最近 24 小时', value: 1440 },
]

function ms(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`
  return `${value.toFixed(0)} ms`
}

function pct(value: number | null): string {
  // `null` 表示「没有读数」,不是 0%。两者在这一格上含义相反,
  // 所以不能都渲染成 `0%`。
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`
}

export function MetricsPage() {
  const [windowMinutes, setWindowMinutes] = useState(60)
  const [summary, setSummary] = useState<MetricsSummary | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const load = useCallback(async (minutes: number) => {
    setLoading(true)
    setError('')
    try {
      setSummary(await adminApi.metrics(minutes))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load(windowMinutes)
  }, [load, windowMinutes])

  const stageRows = Object.entries(summary?.stages ?? {}).map(([stage, value]) => ({
    key: stage,
    stage,
    avg_ms: value,
  }))

  return (
    <Space orientation="vertical" style={{ width: '100%' }} size={16}>
      <Space>
        <Segmented
          options={WINDOWS}
          value={windowMinutes}
          onChange={(value) => setWindowMinutes(value as number)}
        />
        <Typography.Text type="secondary">
          窗口内的数据,不是开服以来的累计值——累计值在系统变好时几乎不动,
          看起来像「优化没生效」。
        </Typography.Text>
      </Space>

      {error && <Alert type="error" message={error} showIcon />}

      <Row gutter={16}>
        <Col span={6}>
          <Card size="small" loading={loading}>
            <Statistic title="请求数" value={summary?.requests ?? 0} />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small" loading={loading}>
            <Statistic
              title="错误率"
              value={summary ? (summary.error_rate * 100).toFixed(1) : 0}
              suffix="%"
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card size="small" loading={loading}>
            <Space size={8} wrap>
              {Object.entries(summary?.outcomes ?? {}).map(([outcome, count]) => (
                <Tag key={outcome} color={outcome === 'done' ? 'green' : 'red'}>
                  {outcome}: {count}
                </Tag>
              ))}
              {summary && Object.keys(summary.outcomes).length === 0 && (
                <Typography.Text type="secondary">窗口内没有请求</Typography.Text>
              )}
            </Space>
            <Typography.Paragraph className="hint">
              <Typography.Text strong>失败也计入分母。</Typography.Text>
              只统计成功请求的延迟分布,会把最慢的那部分(慢到出错的)整个从
              分母里删掉——p95 越优化越好看,而用户感受不到。
            </Typography.Paragraph>
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={12}>
          <Card size="small" title="总耗时" loading={loading}>
            <Descriptions column={4} size="small">
              <Descriptions.Item label="p50">{ms(summary?.total_ms.p50 ?? 0)}</Descriptions.Item>
              <Descriptions.Item label="p95">{ms(summary?.total_ms.p95 ?? 0)}</Descriptions.Item>
              <Descriptions.Item label="max">{ms(summary?.total_ms.max ?? 0)}</Descriptions.Item>
              <Descriptions.Item label="avg">{ms(summary?.total_ms.avg ?? 0)}</Descriptions.Item>
            </Descriptions>
            <Typography.Paragraph className="hint">
              分位数用<Typography.Text strong>最近秩</Typography.Text>而不是线性插值:
              n=10 时两种定义给出 5 和 5.5,都「对」。选最近秩是因为它总是返回
              真实存在的观测值。
            </Typography.Paragraph>
          </Card>
        </Col>
        <Col span={12}>
          <Card size="small" title="首 token" loading={loading}>
            {summary?.first_token_ms ? (
              <Descriptions column={4} size="small">
                <Descriptions.Item label="p50">{ms(summary.first_token_ms.p50)}</Descriptions.Item>
                <Descriptions.Item label="p95">{ms(summary.first_token_ms.p95)}</Descriptions.Item>
                <Descriptions.Item label="max">{ms(summary.first_token_ms.max)}</Descriptions.Item>
                <Descriptions.Item label="avg">{ms(summary.first_token_ms.avg)}</Descriptions.Item>
              </Descriptions>
            ) : (
              <Typography.Text type="secondary">
                —(窗口内没有任何一轮流出过 token)
              </Typography.Text>
            )}
          </Card>
        </Col>
      </Row>

      <Card size="small" title="缓存与成本" loading={loading}>
        <Descriptions column={2} size="small">
          <Descriptions.Item label="检索缓存命中率">
            {pct(summary?.cache_hit_rate_retrieval ?? null)}
          </Descriptions.Item>
          <Descriptions.Item label="答案缓存命中率">
            {pct(summary?.cache_hit_rate_answer ?? null)}
          </Descriptions.Item>
          <Descriptions.Item label="有据可依比例">
            {pct(summary?.grounded_rate ?? null)}
          </Descriptions.Item>
          <Descriptions.Item label="prompt 缓存命中率">
            {/* 这一格刻意用 `pct`,它对 `null` 显示 `—`。 */}
            {pct(summary?.cache_read_rate ?? null)}
          </Descriptions.Item>
          <Descriptions.Item label="prompt tokens">
            {summary?.prompt_tokens ?? 0}
          </Descriptions.Item>
          <Descriptions.Item label="其中缓存供给">
            {summary?.cached_prompt_tokens ?? 0}
          </Descriptions.Item>
          <Descriptions.Item label="completion tokens">
            {summary?.completion_tokens ?? 0}
          </Descriptions.Item>
        </Descriptions>
        <Typography.Paragraph className="hint">
          prompt 缓存命中率的分母
          <Typography.Text strong>只算报过读数的请求</Typography.Text>,而且
          <Typography.Text strong>没有读数时显示「—」而不是「0%」</Typography.Text>:
          0% 会被读成「缓存一条都没命中」,而事实可能是平台压根没回这个读数。
          详见 docs/PROMPT-CACHE.md。
        </Typography.Paragraph>
      </Card>

      <Card size="small" title="各阶段平均耗时" loading={loading}>
        <Table
          size="small"
          pagination={false}
          dataSource={stageRows}
          locale={{ emptyText: '窗口内没有请求' }}
          columns={[
            { title: '阶段', dataIndex: 'stage' },
            {
              title: '平均耗时',
              dataIndex: 'avg_ms',
              render: (value: number) => ms(value),
            },
          ]}
        />
        <Typography.Paragraph className="hint">
          每个阶段的分母是「<Typography.Text strong>记过这个阶段</Typography.Text>
          」的请求数,不是总请求数——否则关掉重排之后它的平均值会被那些
          没跑过它的请求拉低,看起来像是「重排变快了」。
        </Typography.Paragraph>
      </Card>
    </Space>
  )
}
