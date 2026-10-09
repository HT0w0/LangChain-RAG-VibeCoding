/**
 * 全部接口的调用点。**每个路径只在这里出现一次。**
 *
 * 散到页面里写的话,改动接口时得靠搜索,而漏掉一处不会报错——只会让
 * 那个页面在某个分支下 404。
 */

import { request } from './client'
import type {
  ChunkPreviewPage,
  DocumentListResponse,
  DocumentOut,
  KBOptionListResponse,
  KBListResponse,
  KBOut,
  MessageListResponse,
  MetricsSummary,
  SessionListResponse,
  SessionOut,
  TokenResponse,
  UserListResponse,
  UserOut,
} from './types'

// ---------------------------------------------------------------------------
// 认证
// ---------------------------------------------------------------------------

export const authApi = {
  login: (username: string, password: string) =>
    request<TokenResponse>('/auth/login', { method: 'POST', body: { username, password } }),

  register: (username: string, password: string) =>
    request<TokenResponse>('/auth/register', {
      method: 'POST',
      body: { username, password },
    }),

  me: () => request<UserOut>('/auth/me'),

  changePassword: (oldPassword: string, newPassword: string) =>
    request<null>('/auth/change-password', {
      method: 'POST',
      body: { old_password: oldPassword, new_password: newPassword },
    }),
}

// ---------------------------------------------------------------------------
// 会话
// ---------------------------------------------------------------------------

export const sessionApi = {
  list: () => request<SessionListResponse>('/sessions'),

  /**
   * 新建会话。
   *
   * **问答之前必须先调它。** `/chat/stream` 的 `session_id` 为 `null` 时
   * 后端**不落库**——答案照常流式返回、`done` 事件照常发出,只是
   * `session_id`/`message_id` 都是 `null`,这一轮什么都没存下来。
   * 症状是「历史记录永远只有一条」或者「刷新之后对话没了」,而且
   * 全程没有任何报错。
   */
  create: (kbIds?: number[]) =>
    request<SessionOut>('/sessions', { method: 'POST', body: { kb_ids: kbIds } }),

  get: (id: number) => request<SessionOut>(`/sessions/${id}`),

  messages: (id: number) => request<MessageListResponse>(`/sessions/${id}/messages`),

  rename: (id: number, title: string) =>
    request<SessionOut>(`/sessions/${id}`, { method: 'PATCH', body: { title } }),

  remove: (id: number) => request<null>(`/sessions/${id}`, { method: 'DELETE' }),
}

// ---------------------------------------------------------------------------
// 知识库
// ---------------------------------------------------------------------------

export const kbApi = {
  list: () => request<KBListResponse>('/kb'),

  options: () => request<KBOptionListResponse>('/kb/options'),

  get: (id: number) => request<KBOut>(`/kb/${id}`),

  create: (name: string, description?: string) =>
    request<KBOut>('/kb', { method: 'POST', body: { name, description } }),

  update: (id: number, patch: { name?: string; description?: string }) =>
    request<KBOut>(`/kb/${id}`, { method: 'PATCH', body: patch }),

  remove: (id: number) => request<null>(`/kb/${id}`, { method: 'DELETE' }),

  documents: (id: number) => request<DocumentListResponse>(`/kb/${id}/documents`),

  document: (kbId: number, docId: number) =>
    request<DocumentOut>(`/kb/${kbId}/documents/${docId}`),

  upload: (id: number, files: File[]) => {
    const formData = new FormData()
    // 字段名必须是 `files`(复数):后端签名是 `files: list[UploadFile]`,
    // 写成单数的话 FastAPI 会报 422,而错误信息只说缺 `files` 字段。
    for (const file of files) formData.append('files', file)
    return request<DocumentOut[]>(`/kb/${id}/documents`, { method: 'POST', formData })
  },

  removeDocument: (kbId: number, docId: number) =>
    request<null>(`/kb/${kbId}/documents/${docId}`, { method: 'DELETE' }),

  reindex: (kbId: number, docId: number) =>
    request<DocumentOut>(`/kb/${kbId}/documents/${docId}/reindex`, { method: 'POST' }),

  chunks: (kbId: number, docId: number, page = 1, pageSize = 20) =>
    request<ChunkPreviewPage>(
      `/kb/${kbId}/documents/${docId}/chunks?page=${page}&page_size=${pageSize}`,
    ),
}

// ---------------------------------------------------------------------------
// 管理
// ---------------------------------------------------------------------------

export const adminApi = {
  users: (page = 1, pageSize = 20) =>
    request<UserListResponse>(`/admin/users?page=${page}&page_size=${pageSize}`),

  setActive: (userId: number, isActive: boolean) =>
    request<UserOut>(`/admin/users/${userId}/active`, {
      method: 'PATCH',
      body: { is_active: isActive },
    }),

  setRole: (userId: number, role: 'admin' | 'user') =>
    request<UserOut>(`/admin/users/${userId}/role`, { method: 'PATCH', body: { role } }),

  metrics: (windowMinutes = 60) =>
    request<MetricsSummary>(`/admin/metrics?window_minutes=${windowMinutes}`),
}
