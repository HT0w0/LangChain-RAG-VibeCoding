/**
 * 问答工作台。
 *
 * 三条时序约束决定了这个文件的结构,每一条写错了都不报错:
 *
 * 1. **提问前必须先建会话。** 没有 `session_id` 时后端不落库,而答案
 *    照常流式返回——历史记录就这么丢了,全程无提示。
 * 2. **`sources` 先于 `token` 到达。** 这是服务端刻意的顺序(用户提问后
 *    立刻看到引用了哪些片段,而不是等答案写完)。所以来源卡片要**在流式
 *    过程中就渲染**,不能等 `done`。
 * 3. **`done.answer` 才是权威。** 流式的 token 可能因为一次断连少半句,
 *    而 `done` 带的是完整答案。以 token 为准的话,断流时页面上留着的
 *    是一段被截断的回答,用户看不出来。
 */

import { Alert, Button, Card, Empty, Input, List, Space, Spin, Typography, message } from 'antd'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiError } from '../api/client'
import { streamChat } from '../api/chat'
import { kbApi, sessionApi } from '../api/endpoints'
import type { KBOption, MessageOut, SessionOut, SourceRef } from '../api/types'
import { AnswerText } from '../components/AnswerText'
import { SourceCards } from '../components/SourceCards'

interface Draft {
  answer: string
  sources: SourceRef[]
  cited: number[]
}

export function ChatPage() {
  const [sessions, setSessions] = useState<SessionOut[]>([])
  const [currentId, setCurrentId] = useState<number | null>(null)
  const [messages, setMessages] = useState<MessageOut[]>([])
  const [kbOptions, setKbOptions] = useState<KBOption[]>([])
  const [scope, setScope] = useState<number[]>([])
  const [draft, setDraft] = useState<Draft | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [highlighted, setHighlighted] = useState<number | null>(null)

  //: 用 ref 而不是 state:它不参与渲染,而放进 state 会让「停止」按钮的
  //: 点击闭包里拿到一个过期的 controller。
  const abortRef = useRef<AbortController | null>(null)

  const refreshSessions = useCallback(async () => {
    const result = await sessionApi.list()
    setSessions(result.items)
    return result.items
  }, [])

  useEffect(() => {
    // 知识库选项**失败时不拦人**:普通用户没有知识库时它可能为空,
    // 而空选项的语义就是「全部库」,照样能提问。
    void kbApi
      .options()
      .then((result) => setKbOptions(result.items))
      .catch(() => setKbOptions([]))
    void refreshSessions().catch(() => setSessions([]))
  }, [refreshSessions])

  const openSession = useCallback(async (id: number) => {
    setCurrentId(id)
    setDraft(null)
    setError('')
    try {
      const result = await sessionApi.messages(id)
      setMessages(result.items)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : '加载历史失败')
      setMessages([])
    }
  }, [])

  /** 拿一个可用的会话 id,没有就新建。见文件头第 1 条。 */
  const ensureSession = useCallback(async (): Promise<number> => {
    if (currentId !== null) return currentId
    const created = await sessionApi.create(scope.length > 0 ? scope : undefined)
    setCurrentId(created.id)
    setSessions((prev) => [created, ...prev])
    return created.id
  }, [currentId, scope])

  const send = useCallback(
    async (question: string) => {
      setBusy(true)
      setError('')

      let sessionId: number
      try {
        sessionId = await ensureSession()
      } catch (err) {
        setBusy(false)
        setError(err instanceof ApiError ? err.message : '创建会话失败')
        return
      }

      // 用户消息**立刻上屏**,不等服务端。它的 id 是临时的负数,
      // `done` 之后整段会被服务端返回的真实记录替换掉。
      setMessages((prev) => [
        ...prev,
        {
          id: -Date.now(),
          session_id: sessionId,
          role: 'user',
          content: question,
          sources: [],
          cited: [],
          prompt_tokens: 0,
          completion_tokens: 0,
          latency_ms: 0,
          created_at: new Date().toISOString(),
        },
      ])
      setDraft({ answer: '', sources: [], cited: [] })

      const controller = new AbortController()
      abortRef.current = controller

      try {
        await streamChat(
          {
            question,
            session_id: sessionId,
            kb_ids: scope.length > 0 ? scope : undefined,
          },
          {
            onSources: (event) =>
              setDraft((prev) => ({ answer: prev?.answer ?? '', sources: event.sources, cited: [] })),
            onToken: (text) => setDraft((prev) => ({ ...(prev ?? { sources: [], cited: [] }), answer: (prev?.answer ?? '') + text })),
            onDone: () => {
              /* 收尾统一放到 finally,见下 */
            },
            onError: (event) => setError(event.message),
          },
          controller.signal,
        )
      } catch (err) {
        if ((err as Error)?.name !== 'AbortError') {
          setError(err instanceof ApiError || err instanceof Error ? err.message : '请求失败')
        }
      } finally {
        abortRef.current = null
        setBusy(false)
        setDraft(null)
        // **以服务端为准重取一次。** 本地拼出来的记录缺 `message_id`、
        // 缺来源快照,而且第一轮之后会话标题是服务端生成的。多一次请求
        // 换来的是「界面上的东西和库里的一模一样」,这个交换很划算。
        try {
          const result = await sessionApi.messages(sessionId)
          setMessages(result.items)
          await refreshSessions()
        } catch {
          /* 刷新失败不影响这一轮已经拿到的答案 */
        }
      }
    },
    [ensureSession, refreshSessions, scope],
  )

  const onCiteClick = useCallback((index: number) => {
    setHighlighted(index)
    // 滚动到对应的来源卡片。用 `scrollIntoView` 而不是算偏移量——
    // 卡片高度随正文长短变化,算出来的位置会在长资料上偏掉。
    document.getElementById(`source-${index}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [])

  const latestSources = useMemo(() => {
    if (draft) return draft.sources
    const last = [...messages].reverse().find((m) => m.role === 'assistant')
    return last?.sources ?? []
  }, [draft, messages])

  const latestCited = draft?.cited ?? [...messages].reverse().find((m) => m.role === 'assistant')?.cited ?? []

  return (
    <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
      <Card
        title="会话"
        size="small"
        style={{ width: 260, flexShrink: 0 }}
        extra={
          <Button
            size="small"
            type="link"
            onClick={() => {
              setCurrentId(null)
              setMessages([])
              setDraft(null)
              setError('')
            }}
          >
            新建
          </Button>
        }
      >
        <List
          size="small"
          dataSource={sessions}
          locale={{ emptyText: '还没有会话' }}
          renderItem={(item) => (
            <List.Item
              style={{ cursor: 'pointer', paddingInline: 8 }}
              onClick={() => void openSession(item.id)}
              actions={[
                <a
                  key="del"
                  onClick={async (event) => {
                    event.stopPropagation()
                    await sessionApi.remove(item.id)
                    if (item.id === currentId) {
                      setCurrentId(null)
                      setMessages([])
                    }
                    await refreshSessions()
                    void message.success('已删除')
                  }}
                >
                  删
                </a>,
              ]}
            >
              <Typography.Text
                ellipsis
                strong={item.id === currentId}
                style={{ maxWidth: 150 }}
              >
                {item.title || `会话 ${item.id}`}
              </Typography.Text>
            </List.Item>
          )}
        />
      </Card>

      <div style={{ flex: 1, minWidth: 0 }}>
        <Space direction="vertical" style={{ width: '100%' }} size={16}>
          {error && <Alert type="error" message={error} showIcon closable onClose={() => setError('')} />}

          {messages.length === 0 && draft === null ? (
            <Card>
              <Empty description="在下面输入你的问题,答案会标注引用了哪几条资料" />
            </Card>
          ) : null}

          {messages.map((item) => (
            <Card key={item.id} size="small">
              <Typography.Text type="secondary">
                {item.role === 'user' ? '你' : '助手'}
              </Typography.Text>
              <div style={{ marginTop: 8 }}>
                {item.role === 'assistant' ? (
                  <AnswerText
                    content={item.content}
                    sources={item.sources}
                    onCiteClick={onCiteClick}
                  />
                ) : (
                  <Typography.Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>
                    {item.content}
                  </Typography.Paragraph>
                )}
              </div>
              {item.role === 'assistant' && item.sources.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <SourceCards sources={item.sources} highlighted={highlighted} />
                </div>
              )}
            </Card>
          ))}

          {draft && (
            <Card size="small">
              <Typography.Text type="secondary">助手</Typography.Text>
              <div style={{ marginTop: 8 }}>
                {draft.answer ? (
                  <AnswerText content={draft.answer} sources={draft.sources} onCiteClick={onCiteClick} />
                ) : (
                  <Spin size="small" />
                )}
              </div>
              {draft.sources.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  {/* 来源卡片**在流式过程中就渲染**——服务端刻意先发 sources
                      再流 token,就是为了这个。等 done 才显示的话,那个设计
                      白做了。 */}
                  <SourceCards sources={draft.sources} highlighted={highlighted} />
                </div>
              )}
            </Card>
          )}

          <Composer
            busy={busy}
            kbOptions={kbOptions}
            scope={scope}
            onScopeChange={setScope}
            onSend={(question) => void send(question)}
            onStop={() => abortRef.current?.abort()}
          />

          {latestSources.length > 0 && (
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              这一轮引用了 {latestCited.length} / {latestSources.length} 条资料
            </Typography.Text>
          )}
        </Space>
      </div>
    </div>
  )
}

interface ComposerProps {
  busy: boolean
  kbOptions: KBOption[]
  scope: number[]
  onScopeChange: (ids: number[]) => void
  onSend: (question: string) => void
  onStop: () => void
}

function Composer({ busy, kbOptions, scope, onScopeChange, onSend, onStop }: ComposerProps) {
  const [value, setValue] = useState('')

  function submit() {
    const question = value.trim()
    // 空白问题直接不发:服务端会拒,但走一趟网络再弹个错没有意义。
    if (question === '' || busy) return
    setValue('')
    onSend(question)
  }

  return (
    <Card size="small">
      <Space direction="vertical" style={{ width: '100%' }}>
        {kbOptions.length > 0 && (
          <Space size={4} wrap>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              检索范围:
            </Typography.Text>
            <Button
              size="small"
              type={scope.length === 0 ? 'primary' : 'default'}
              onClick={() => onScopeChange([])}
            >
              全部
            </Button>
            {kbOptions.map((option) => (
              <Button
                key={option.id}
                size="small"
                type={scope.includes(option.id) ? 'primary' : 'default'}
                onClick={() =>
                  onScopeChange(
                    scope.includes(option.id)
                      ? scope.filter((id) => id !== option.id)
                      : [...scope, option.id],
                  )
                }
              >
                {option.name}
              </Button>
            ))}
          </Space>
        )}
        <Input.TextArea
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="问点什么,比如「RX9070 显卡多少钱」"
          autoSize={{ minRows: 2, maxRows: 6 }}
          disabled={busy}
          onPressEnter={(event) => {
            // Enter 发送、Shift+Enter 换行。不判断 shiftKey 的话,
            // 用户永远没法在问题里换行。
            if (!event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
        />
        <Space>
          <Button type="primary" loading={busy} onClick={submit}>
            发送
          </Button>
          {busy && <Button onClick={onStop}>停止</Button>}
        </Space>
      </Space>
    </Card>
  )
}
