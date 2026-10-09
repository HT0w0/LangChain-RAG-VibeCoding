/**
 * 接口的类型定义。
 *
 * **照 `GET /openapi.json` 抄的,不是猜的。** 后端那边为此专门修过一个
 * 缺陷:`UTCModel` 的通配序列化器把整个响应模型的类型都抹平了,OpenAPI 里
 * 只剩 `{"title": "Id"}`,照它抄出来的类型全是 `any`——而 `any` 不会报错,
 * 它只是让所有的字段访问都不受检查。详见 `backend/tests/test_schemas.py`。
 *
 * 时间统一是**带偏移量的 ISO 字符串**(后端保证),`new Date(...)` 直接能解析。
 */

export type Role = 'admin' | 'user'

export interface UserOut {
  id: number
  username: string
  role: Role
  is_active: boolean
  created_at: string
  last_login_at?: string | null
}

export interface TokenResponse {
  access_token: string
  token_type?: string
  expires_in: number
  user: UserOut
}

export interface UserListResponse {
  total: number
  items: UserOut[]
}

// ---------------------------------------------------------------------------
// 会话与消息
// ---------------------------------------------------------------------------

export interface SessionOut {
  id: number
  user_id: number
  title: string
  kb_ids?: number[] | null
  message_count: number
  created_at: string
  updated_at: string
  last_message_at: string
}

export interface SessionListResponse {
  total: number
  items: SessionOut[]
}

/**
 * 一条被引用的资料。`index` 就是正文里 `[index]` 的那个数字,**从 1 起**。
 *
 * `content` 是完整正文而不是摘要——来源卡片要能展开看原文,而摘要会让
 * 用户看不到支撑结论的那半句。
 */
export interface SourceRef {
  index: number
  chunk_id: number
  doc_id: number
  doc_name: string
  kb_id: number
  kb_name: string
  chunk_index: number
  content: string
  category?: string | null
  price?: number | null
  rerank_score?: number | null
  fusion_score?: number | null
  vector_rank?: number | null
  keyword_rank?: number | null
}

export interface MessageOut {
  id: number
  session_id: number
  role: 'user' | 'assistant'
  content: string
  sources: SourceRef[]
  cited: number[]
  prompt_tokens: number
  completion_tokens: number
  latency_ms: number
  error?: string | null
  created_at: string
}

export interface MessageListResponse {
  total: number
  items: MessageOut[]
}

// ---------------------------------------------------------------------------
// 知识库
// ---------------------------------------------------------------------------

export interface KBOut {
  id: number
  name: string
  description?: string | null
  created_by?: number | null
  doc_count: number
  chunk_count: number
  created_at: string
  updated_at: string
}

export interface KBListResponse {
  total: number
  items: KBOut[]
}

/**
 * `GET /api/kb/options` 的条目(问答页的范围选择器用)。
 *
 * **只有 id 和名字,没有 doc_count 之类的管理信息。** 这条接口普通用户
 * 也能调,它刻意只返回「能选哪几个库」——库里有几篇文档、谁建的,属于
 * 管理面数据。所以它和 `KBOut` 是两个类型,不要合并。
 */
export interface KBOption {
  id: number
  name: string
}

export interface KBOptionListResponse {
  items: KBOption[]
}

/** 文档摄取状态。`pending|parsing|embedding|ready|failed`。 */
export type DocumentStatus =
  | 'pending'
  | 'parsing'
  | 'embedding'
  | 'ready'
  | 'failed'

export interface DocumentOut {
  id: number
  kb_id: number
  filename: string
  mime?: string | null
  size: number
  status: DocumentStatus
  error?: string | null
  warnings?: string[]
  chunk_count: number
  uploaded_by?: number | null
  created_at: string
  updated_at: string
}

export interface DocumentListResponse {
  total: number
  items: DocumentOut[]
}

export interface ChunkPreview {
  id: number
  chunk_index: number
  content: string
  token_count: number
  meta: Record<string, unknown>
}

export interface ChunkPreviewPage {
  document_id: number
  filename: string
  total: number
  items: ChunkPreview[]
}

// ---------------------------------------------------------------------------
// 指标
// ---------------------------------------------------------------------------

export interface LatencyBucket {
  p50: number
  p95: number
  max: number
  avg: number
}

export interface MetricsSummary {
  window_minutes: number
  requests: number
  outcomes: Record<string, number>
  error_rate: number
  total_ms: LatencyBucket
  first_token_ms: LatencyBucket | null
  stages: Record<string, number>
  cache_hit_rate_retrieval: number
  cache_hit_rate_answer: number
  grounded_rate: number
  prompt_tokens: number
  completion_tokens: number
  cached_prompt_tokens: number
  /**
   * `null` 表示**这一窗里没有任何请求报过缓存读数**(平台没说),
   * 不是「命中率 0%」。面板上两者必须能区分——详见 `docs/PROMPT-CACHE.md`。
   */
  cache_read_rate: number | null
}

// ---------------------------------------------------------------------------
// 问答(SSE)
// ---------------------------------------------------------------------------

export interface ChatFilters {
  category?: string | null
  price_min?: number | null
  price_max?: number | null
}

export interface ChatRequest {
  question: string
  kb_ids?: number[] | null
  filters?: ChatFilters | null
  session_id?: number | null
}

export interface SourcesEvent {
  sources: SourceRef[]
  route_sizes: Record<string, number>
  reranked: boolean
  degraded: boolean
}

export interface TokenEvent {
  text: string
}

export interface DoneEvent {
  answer: string
  cited: number[]
  /**
   * `null` 表示这一轮**根本没花 token**(命中答案缓存,或检索为空走固定答复)。
   *
   * `cached_tokens` 是 `null` 时意思是「平台没报」,`0` 才是「平台说没命中」。
   */
  usage: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    cached_tokens: number | null
  } | null
  latency_ms: number
  stage_timings: Record<string, number>
  grounded: boolean
  search_query: string
  history_messages: number
  cache_hit: boolean
  answer_cache_hit: boolean
  session_id?: number | null
  message_id?: number | null
}

export interface ErrorEvent {
  code: string
  message: string
  elapsed_ms: number
}
