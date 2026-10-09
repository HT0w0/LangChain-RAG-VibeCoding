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

import { Alert, Button, Card, Empty, Input, Space, Typography, message } from 'antd'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ApiError } from '../api/client'
import { streamChat } from '../api/chat'
import { kbApi, sessionApi } from '../api/endpoints'
import type { KBOption, MessageOut, SessionOut, SourceRef } from '../api/types'
import { MessageBubble } from '../components/MessageBubble'
import { SidebarList } from '../components/SidebarList'

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

  // ---- 会话 id 进 URL ------------------------------------------------
  //
  // **不这样做的话,刷新一次页面就丢掉正在看的会话**,回到「新会话」。
  // 用户在长对话里按 F5(或者不小心按到)就要从侧栏里重新找回来。
  // 顺带它也让会话可以被直接链接。
  const [searchParams, setSearchParams] = useSearchParams()

  // 挂载时按 URL 打开一次。**只在挂载时跑**:之后 URL 跟着用户的操作走,
  // 两边互相驱动会打架。
  const restored = useRef(false)
  useEffect(() => {
    if (restored.current) return
    restored.current = true
    const id = Number(searchParams.get('session'))
    // `Number(null)` 是 0,`Number('abc')` 是 NaN——两个都要挡掉。
    if (Number.isInteger(id) && id > 0) void openSession(id)
  }, [openSession, searchParams])

  useEffect(() => {
    setSearchParams(currentId === null ? {} : { session: String(currentId) }, {
      replace: true,
    })
  }, [currentId, setSearchParams])

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

  // 新消息进来时自动滚到底。用户刚问完一句,应该看到答复,而不是
  // 自己往下滚——尤其在答案有几屏长的时候。
  const scrollRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const box = scrollRef.current
    if (box) box.scrollTop = box.scrollHeight
  }, [messages, draft?.answer])

  const startNewSession = useCallback(() => {
    setCurrentId(null)
    setMessages([])
    setDraft(null)
    setError('')
  }, [])

  return (
    <div className="chat-shell">
      <aside className="chat-aside">
        <Card
          size="small"
          className="panel panel-flush"
          title="会话"
          extra={
            <Button type="link" size="small" onClick={startNewSession}>
              新建
            </Button>
          }
        >
          <SidebarList
            empty="还没有会话"
            items={sessions.map((item) => ({
              key: item.id,
              title: item.title || `会话 ${item.id}`,
              subtitle: `${item.message_count} 条消息`,
              selected: item.id === currentId,
              onSelect: () => void openSession(item.id),
              action: (
                <a
                  className="sidebar-row-action"
                  onClick={async () => {
                    await sessionApi.remove(item.id)
                    if (item.id === currentId) startNewSession()
                    await refreshSessions()
                    void message.success('已删除')
                  }}
                >
                  删除
                </a>
              ),
            }))}
          />
        </Card>
      </aside>

      <section className="chat-main">
        <div className="chat-scroll" ref={scrollRef}>
          {error && (
            <Alert
              type="error"
              message={error}
              showIcon
              closable
              onClose={() => setError('')}
              style={{ marginBottom: 12 }}
            />
          )}

          {messages.length === 0 && draft === null ? (
            <Empty
              className="chat-empty"
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={
                <span>
                  问点什么吧。答案只依据知识库里的资料,
                  <br />
                  每句话都标出来源,点开就能核对原文。
                </span>
              }
            />
          ) : null}

          {messages.map((item) => (
            <MessageBubble
              key={item.id}
              role={item.role}
              content={item.content}
              sources={item.sources}
              idPrefix={`m${item.id}`}
            />
          ))}

          {draft && (
            // 流式中:**只显示答案,不显示来源**。服务端确实先发 `sources`
            // 再流 token,但那是传输顺序;答案还没写完就铺开一屏来源卡片,
            // 会把用户真正在等的东西挤下去。`sources` 仍然要传——正文里的
            // `[1]` 要靠它才知道上界,不然流式过程中引用标记渲染不出来。
            <MessageBubble
              role="assistant"
              content={draft.answer}
              sources={draft.sources}
              streaming
              idPrefix="draft"
            />
          )}
        </div>

        <div className="chat-composer">
          <Composer
            busy={busy}
            kbOptions={kbOptions}
            scope={scope}
            onScopeChange={setScope}
            onSend={(question) => void send(question)}
            onStop={() => abortRef.current?.abort()}
          />
        </div>
      </section>
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
    <Card size="small" className="panel">
      <Space orientation="vertical" style={{ width: '100%' }} size={10}>
        {kbOptions.length > 0 && (
          <div className="composer-scope">
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              检索范围
            </Typography.Text>
            {/* 用**单选按钮组**的语义而不是一排会变色的普通按钮:
                「全部」和具体库是互斥的,而选中态一眼看得出来。 */}
            <Space size={4} wrap>
              <Button
                size="small"
                type={scope.length === 0 ? 'primary' : 'default'}
                ghost={scope.length === 0}
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
          </div>
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

        <div className="composer-actions">
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            Enter 发送 · Shift+Enter 换行
          </Typography.Text>
          <Space>
            {busy && <Button onClick={onStop}>停止</Button>}
            <Button type="primary" loading={busy} onClick={submit}>
              发送
            </Button>
          </Space>
        </div>
      </Space>
    </Card>
  )
}
