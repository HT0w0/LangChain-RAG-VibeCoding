/**
 * 用户管理。
 *
 * 表上那两个开关**对当前登录的账号是禁用的**——后端也拦(见
 * `UserAdminService.set_active` 的 `cannot_disable_self` / `last_admin`)。
 * 前端这层不是为了安全(后端才是防线),而是为了**不要给出一个必然失败的
 * 按钮**:点了弹一句「不能禁用当前登录的账号」,用户只会觉得系统在刁难他。
 */

import { Alert, Button, Card, Popconfirm, Space, Table, Tag, Tooltip, Typography, message } from 'antd'
import { useCallback, useEffect, useState } from 'react'
import { ApiError } from '../api/client'
import { adminApi } from '../api/endpoints'
import type { UserOut } from '../api/types'
import { useAuth } from '../store/auth'

export function UsersPage() {
  const { user: me } = useAuth()
  const [users, setUsers] = useState<UserOut[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = useCallback(async (target: number) => {
    setLoading(true)
    setError('')
    try {
      const result = await adminApi.users(target, 20)
      setUsers(result.items)
      setTotal(result.total)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load(page)
  }, [load, page])

  async function apply(action: () => Promise<unknown>, success: string) {
    try {
      await action()
      await load(page)
      void message.success(success)
    } catch (err) {
      // 服务端会拒掉「禁用自己」「撤掉最后一个管理员」这类操作,
      // 它的理由比前端能编的任何一句话都准,直接显示。
      void message.error(err instanceof ApiError ? err.message : '操作失败')
    }
  }

  return (
    <Card title="用户管理" size="small">
      {error && <Alert type="error" message={error} showIcon style={{ marginBottom: 16 }} />}
      <Table<UserOut>
        rowKey="id"
        size="small"
        loading={loading}
        dataSource={users}
        pagination={{
          current: page,
          pageSize: 20,
          total,
          onChange: setPage,
          showSizeChanger: false,
        }}
        columns={[
          { title: 'ID', dataIndex: 'id', width: 70 },
          {
            title: '用户名',
            dataIndex: 'username',
            render: (value: string, row) =>
              row.id === me?.id ? (
                <Space size={4}>
                  <Typography.Text strong>{value}</Typography.Text>
                  <Tag>我自己</Tag>
                </Space>
              ) : (
                value
              ),
          },
          {
            title: '角色',
            dataIndex: 'role',
            width: 120,
            render: (role: string) =>
              role === 'admin' ? <Tag color="gold">管理员</Tag> : <Tag>普通用户</Tag>,
          },
          {
            title: '状态',
            dataIndex: 'is_active',
            width: 100,
            render: (active: boolean) =>
              active ? <Tag color="green">启用</Tag> : <Tag color="red">已禁用</Tag>,
          },
          {
            title: '注册时间',
            dataIndex: 'created_at',
            render: (value: string) => new Date(value).toLocaleString(),
          },
          {
            title: '操作',
            width: 220,
            render: (_value, row) => {
              const isSelf = row.id === me?.id
              return (
                <Space size={8}>
                  <Tooltip title={isSelf ? '不能禁用当前登录的账号' : ''}>
                    <Popconfirm
                      title={row.is_active ? `禁用「${row.username}」?` : `启用「${row.username}」?`}
                      disabled={isSelf}
                      onConfirm={() =>
                        apply(
                          () => adminApi.setActive(row.id, !row.is_active),
                          row.is_active ? '已禁用' : '已启用',
                        )
                      }
                    >
                      <Button size="small" disabled={isSelf}>
                        {row.is_active ? '禁用' : '启用'}
                      </Button>
                    </Popconfirm>
                  </Tooltip>
                  <Tooltip title={isSelf ? '不能改自己的角色' : ''}>
                    <Button
                      size="small"
                      disabled={isSelf}
                      onClick={() =>
                        apply(
                          () => adminApi.setRole(row.id, row.role === 'admin' ? 'user' : 'admin'),
                          '角色已更新',
                        )
                      }
                    >
                      {row.role === 'admin' ? '降为普通用户' : '设为管理员'}
                    </Button>
                  </Tooltip>
                </Space>
              )
            },
          },
        ]}
      />
      <Typography.Paragraph type="secondary" style={{ fontSize: 12, marginTop: 12, marginBottom: 0 }}>
        后端还守着一条:系统必须保留至少一个启用状态的管理员。所以撤销最后一个
        管理员会被拒绝——那一类的失败信息直接来自服务端,前端不另写一份。
      </Typography.Paragraph>
    </Card>
  )
}
