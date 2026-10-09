import { Alert, Button, Card, Descriptions, Form, Input, Typography, message } from 'antd'
import { useState } from 'react'
import { ApiError } from '../api/client'
import { authApi } from '../api/endpoints'
import { useAuth } from '../store/auth'

interface FormValues {
  oldPassword: string
  newPassword: string
  confirm: string
}

function formatTime(value?: string | null): string {
  if (!value) return '—'
  // 后端给的是**带偏移量的** ISO 串,直接交给 Date 就是对的。
  // 早期不带偏移量时这里要按 UTC 手动补,那个坑见
  // `backend/app/schemas/common.py` 的模块文档。
  return new Date(value).toLocaleString()
}

export function ProfilePage() {
  const { user } = useAuth()
  const [form] = Form.useForm<FormValues>()
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function onFinish(values: FormValues) {
    setBusy(true)
    setError('')
    try {
      await authApi.changePassword(values.oldPassword, values.newPassword)
      form.resetFields()
      void message.success('密码已修改')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : '修改失败,请稍后重试')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ maxWidth: 640, margin: '0 auto' }}>
      <Card title="账号信息" size="small" style={{ marginBottom: 16 }}>
        <Descriptions column={1} size="small">
          <Descriptions.Item label="用户名">{user?.username ?? '—'}</Descriptions.Item>
          <Descriptions.Item label="角色">
            {user?.role === 'admin' ? '管理员' : '普通用户'}
          </Descriptions.Item>
          <Descriptions.Item label="注册时间">{formatTime(user?.created_at)}</Descriptions.Item>
          <Descriptions.Item label="上次登录">{formatTime(user?.last_login_at)}</Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title="修改密码" size="small">
        {error && <Alert type="error" message={error} showIcon style={{ marginBottom: 16 }} />}
        <Form<FormValues> form={form} layout="vertical" onFinish={onFinish} disabled={busy}>
          <Form.Item
            name="oldPassword"
            label="当前密码"
            rules={[{ required: true, message: '请输入当前密码' }]}
          >
            <Input.Password autoComplete="current-password" />
          </Form.Item>
          <Form.Item
            name="newPassword"
            label="新密码"
            rules={[
              { required: true, message: '请输入新密码' },
              { min: 6, message: '密码至少 6 位' },
            ]}
          >
            <Input.Password autoComplete="new-password" />
          </Form.Item>
          <Form.Item
            name="confirm"
            label="确认新密码"
            dependencies={['newPassword']}
            rules={[
              { required: true, message: '请再输入一次新密码' },
              ({ getFieldValue }) => ({
                validator: (_rule, value: string) =>
                  value === getFieldValue('newPassword')
                    ? Promise.resolve()
                    : Promise.reject(new Error('两次输入的密码不一致')),
              }),
            ]}
          >
            <Input.Password autoComplete="new-password" />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={busy}>
            保存
          </Button>
        </Form>
        <Typography.Paragraph type="secondary" style={{ marginTop: 16, marginBottom: 0, fontSize: 12 }}>
          改密之后**当前这把 token 仍然有效**——它是一枚签名过的通行证,
          服务端不查密码。要立刻失效得等 token 自然过期(默认 12 小时)。
        </Typography.Paragraph>
      </Card>
    </div>
  )
}
