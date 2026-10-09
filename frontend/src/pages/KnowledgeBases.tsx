/**
 * 知识库管理。
 *
 * **摄取是后台任务**,上传接口立刻返回 202 而文档此时还是 `pending`。
 * 所以这个页面必须**轮询**:不轮询的话,用户传完文件看到的是「待处理」,
 * 刷新才变「就绪」——他会以为要手动刷新,而更糟的是他可能就此认为
 * 上传失败了,再传一遍。
 *
 * 轮询只在**有文档处于非终态时**进行,全部就绪就停。一直轮询的话,
 * 一个开着这个页面的浏览器会每秒敲一次服务端,而它什么都没在等。
 */

import {
  Alert,
  Button,
  Card,
  Empty,
  Form,
  Input,
  List,
  Modal,
  Popconfirm,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
  Upload,
  message,
} from 'antd'
import { UploadOutlined } from '@ant-design/icons'
import type { RcFile } from 'antd/es/upload/interface'
import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../api/client'
import { kbApi } from '../api/endpoints'
import type { ChunkPreviewPage, DocumentOut, DocumentStatus, KBOut } from '../api/types'

//: 非终态:还会自己往前走的状态。终态是 `ready` 与 `failed`。
const IN_FLIGHT: DocumentStatus[] = ['pending', 'parsing', 'embedding']

const STATUS_META: Record<DocumentStatus, { color: string; text: string }> = {
  pending: { color: 'default', text: '排队中' },
  parsing: { color: 'processing', text: '解析中' },
  embedding: { color: 'processing', text: '向量化中' },
  ready: { color: 'success', text: '已就绪' },
  failed: { color: 'error', text: '失败' },
}

export function KnowledgeBasesPage() {
  const [kbs, setKbs] = useState<KBOut[]>([])
  const [selected, setSelected] = useState<KBOut | null>(null)
  const [docs, setDocs] = useState<DocumentOut[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [preview, setPreview] = useState<ChunkPreviewPage | null>(null)
  const [creating, setCreating] = useState(false)
  const [form] = Form.useForm<{ name: string; description?: string }>()

  const loadKbs = useCallback(async () => {
    try {
      const result = await kbApi.list()
      setKbs(result.items)
      return result.items
    } catch (err) {
      setError(err instanceof ApiError ? err.message : '加载知识库失败')
      return []
    }
  }, [])

  const loadDocs = useCallback(async (kb: KBOut) => {
    try {
      const result = await kbApi.documents(kb.id)
      setDocs(result.items)
      return result.items
    } catch (err) {
      setError(err instanceof ApiError ? err.message : '加载文档失败')
      return []
    }
  }, [])

  useEffect(() => {
    void loadKbs()
  }, [loadKbs])

  useEffect(() => {
    if (selected) void loadDocs(selected)
  }, [selected, loadDocs])

  // ---- 轮询:只在有文档还在跑的时候开 ----
  const timer = useRef<number | null>(null)
  useEffect(() => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current)
      timer.current = null
    }
    if (!selected || !docs.some((doc) => IN_FLIGHT.includes(doc.status))) return

    timer.current = window.setTimeout(() => {
      void loadDocs(selected).then((next) => {
        // 全部进终态时顺手刷一下知识库列表:分块数/文档数变了。
        if (!next.some((doc) => IN_FLIGHT.includes(doc.status))) void loadKbs()
      })
    }, 1500)

    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current)
      timer.current = null
    }
  }, [docs, selected, loadDocs, loadKbs])

  async function createKb(values: { name: string; description?: string }) {
    setBusy(true)
    try {
      const created = await kbApi.create(values.name, values.description)
      setCreating(false)
      form.resetFields()
      await loadKbs()
      setSelected(created)
      void message.success('知识库已创建')
    } catch (err) {
      void message.error(err instanceof ApiError ? err.message : '创建失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
      <Card
        title="知识库"
        size="small"
        style={{ width: 280, flexShrink: 0 }}
        extra={
          <Button size="small" type="link" onClick={() => setCreating(true)}>
            新建
          </Button>
        }
      >
        <List
          size="small"
          dataSource={kbs}
          locale={{ emptyText: '还没有知识库' }}
          renderItem={(kb) => (
            <List.Item
              style={{
                cursor: 'pointer',
                paddingInline: 8,
                background: selected?.id === kb.id ? '#f0f7ff' : undefined,
              }}
              onClick={() => setSelected(kb)}
              actions={[
                <Popconfirm
                  key="del"
                  title={`删除「${kb.name}」?`}
                  description="其中的文档、分块和索引都会一并删除,不可恢复。"
                  onConfirm={async (event) => {
                    event?.stopPropagation()
                    try {
                      await kbApi.remove(kb.id)
                      if (selected?.id === kb.id) {
                        setSelected(null)
                        setDocs([])
                      }
                      await loadKbs()
                      void message.success('已删除')
                    } catch (err) {
                      void message.error(err instanceof ApiError ? err.message : '删除失败')
                    }
                  }}
                >
                  <a onClick={(event) => event.stopPropagation()}>删</a>
                </Popconfirm>,
              ]}
            >
              <List.Item.Meta
                title={<Typography.Text strong={selected?.id === kb.id}>{kb.name}</Typography.Text>}
                description={
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    {kb.doc_count} 篇 · {kb.chunk_count} 块
                  </Typography.Text>
                }
              />
            </List.Item>
          )}
        />
      </Card>

      <div style={{ flex: 1, minWidth: 0 }}>
        <Space direction="vertical" style={{ width: '100%' }} size={16}>
          {error && <Alert type="error" message={error} showIcon closable onClose={() => setError('')} />}

          {selected === null ? (
            <Card>
              <Empty description="左边选一个知识库,或者新建一个" />
            </Card>
          ) : (
            <Card
              title={selected.name}
              size="small"
              extra={
                <Upload
                  multiple
                  showUploadList={false}
                  beforeUpload={() => false}
                  onChange={({ fileList }) => {
                    // `beforeUpload` 返回 false 时 antd 不自己上传,这里接手。
                    // 交给它自己传的话,请求不带我们的 Authorization 头。
                    // `originFileObj` 的类型是 `RcFile`(antd 的 File 子类,
                    // 多了 uid/lastModifiedDate)。谓词写成 `file is File`
                    // 会报 TS2677——`RcFile` 比 `File` 多字段,反向不成立。
                    const files = fileList
                      .map((item) => item.originFileObj)
                      .filter((file): file is RcFile => file != null)
                    if (files.length === 0) return
                    void (async () => {
                      try {
                        await kbApi.upload(selected.id, files)
                        await loadDocs(selected)
                        await loadKbs()
                        void message.success(`已接收 ${files.length} 个文件,正在后台处理`)
                      } catch (err) {
                        void message.error(err instanceof ApiError ? err.message : '上传失败')
                      }
                    })()
                  }}
                >
                  <Button size="small" icon={<UploadOutlined />}>
                    上传文档
                  </Button>
                </Upload>
              }
            >
              <Table<DocumentOut>
                rowKey="id"
                size="small"
                dataSource={docs}
                pagination={false}
                locale={{ emptyText: '还没有文档' }}
                columns={[
                  {
                    title: '文件名',
                    dataIndex: 'filename',
                    render: (value: string, row) => (
                      <Space direction="vertical" size={0}>
                        <Typography.Text>{value}</Typography.Text>
                        {row.error && (
                          <Typography.Text type="danger" style={{ fontSize: 12 }}>
                            {row.error}
                          </Typography.Text>
                        )}
                        {/* 警告不是错误:文档解析出来了、也入库了,只是有
                            些内容没取到(比如扫描件没有文字层)。
                            混进错误里会让人以为文档不能用了。 */}
                        {(row.warnings ?? []).map((warning) => (
                          <Typography.Text key={warning} type="warning" style={{ fontSize: 12 }}>
                            {warning}
                          </Typography.Text>
                        ))}
                      </Space>
                    ),
                  },
                  {
                    title: '状态',
                    dataIndex: 'status',
                    width: 110,
                    render: (status: DocumentStatus) => {
                      const meta = STATUS_META[status]
                      return <Tag color={meta.color}>{meta.text}</Tag>
                    },
                  },
                  { title: '分块', dataIndex: 'chunk_count', width: 70 },
                  {
                    title: '操作',
                    width: 200,
                    render: (_value, row) => (
                      <Space size={8}>
                        <Button
                          size="small"
                          disabled={row.status !== 'ready'}
                          onClick={async () => {
                            try {
                              setPreview(await kbApi.chunks(selected.id, row.id, 1, 20))
                            } catch (err) {
                              void message.error(
                                err instanceof ApiError ? err.message : '读取分块失败',
                              )
                            }
                          }}
                        >
                          看分块
                        </Button>
                        <Tooltip title="重新解析并向量化。改过配置(分块大小、embedding 模型)之后需要它">
                          <Button
                            size="small"
                            onClick={async () => {
                              try {
                                await kbApi.reindex(selected.id, row.id)
                                await loadDocs(selected)
                                void message.success('已加入重建队列')
                              } catch (err) {
                                void message.error(
                                  err instanceof ApiError ? err.message : '重建失败',
                                )
                              }
                            }}
                          >
                            重建
                          </Button>
                        </Tooltip>
                        <Popconfirm
                          title={`删除「${row.filename}」?`}
                          onConfirm={async () => {
                            try {
                              await kbApi.removeDocument(selected.id, row.id)
                              await loadDocs(selected)
                              await loadKbs()
                            } catch (err) {
                              void message.error(
                                err instanceof ApiError ? err.message : '删除失败',
                              )
                            }
                          }}
                        >
                          <a>删</a>
                        </Popconfirm>
                      </Space>
                    ),
                  },
                ]}
              />
              <Typography.Paragraph
                type="secondary"
                style={{ fontSize: 12, marginTop: 12, marginBottom: 0 }}
              >
                处理在后台进行,页面会自动刷新状态。全部就绪后才会停止轮询。
              </Typography.Paragraph>
            </Card>
          )}
        </Space>
      </div>

      <Modal
        open={creating}
        title="新建知识库"
        onCancel={() => setCreating(false)}
        onOk={() => form.submit()}
        confirmLoading={busy}
        destroyOnHidden
      >
        <Form form={form} layout="vertical" onFinish={createKb}>
          <Form.Item name="name" label="名称" rules={[{ required: true, message: '请输入名称' }]}>
            <Input autoFocus />
          </Form.Item>
          <Form.Item name="description" label="说明">
            <Input.TextArea autoSize={{ minRows: 2, maxRows: 4 }} />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        open={preview !== null}
        title={preview ? `分块预览 · ${preview.filename}` : ''}
        footer={null}
        width={720}
        onCancel={() => setPreview(null)}
        destroyOnHidden
      >
        <Typography.Paragraph type="secondary">
          共 {preview?.total ?? 0} 块,下面显示前 20 块。
          <Typography.Text strong>分块是检索的最小单位</Typography.Text>
          ——答案溯源到的那「一条资料」就是其中一块。
        </Typography.Paragraph>
        <List
          size="small"
          dataSource={preview?.items ?? []}
          renderItem={(chunk) => (
            <List.Item>
              <Space direction="vertical" size={4} style={{ width: '100%' }}>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  第 {chunk.chunk_index + 1} 块 · {chunk.token_count} token
                </Typography.Text>
                <Typography.Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>
                  {chunk.content}
                </Typography.Paragraph>
              </Space>
            </List.Item>
          )}
        />
      </Modal>
    </div>
  )
}
