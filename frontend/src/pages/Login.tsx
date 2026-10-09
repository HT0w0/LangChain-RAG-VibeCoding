import { Alert, Button, Card, Form, Input, Typography } from 'antd'
import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { ApiError } from '../api/client'
import { useAuth } from '../store/auth'

interface FormValues {
  username: string
  password: string
}

export function LoginPage() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function onFinish(values: FormValues) {
    setBusy(true)
    setError('')
    try {
      await login(values.username, values.password)
      // 回到被踢出去之前在的那一页;没有的话去问答页。
      const from = (location.state as { from?: { pathname: string } } | null)?.from
      navigate(from?.pathname ?? '/chat', { replace: true })
    } catch (err) {
      // 登录失败的文案**必须来自后端**。前端自己写「用户名或密码错误」
      // 会把「账号被停用」也说成密码错,而这两种情况的下一步完全不同。
      setError(err instanceof ApiError ? err.message : '登录失败,请稍后重试')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ display: 'grid', placeItems: 'center', minHeight: '100vh', padding: 24 }}>
      <Card style={{ width: 380 }}>
        <Typography.Title level={4} style={{ textAlign: 'center' }}>
          电商知识库问答
        </Typography.Title>
        {error && <Alert type="error" message={error} showIcon style={{ marginBottom: 16 }} />}
        <Form<FormValues> layout="vertical" onFinish={onFinish} disabled={busy}>
          <Form.Item name="username" label="用户名" rules={[{ required: true, message: '请输入用户名' }]}>
            <Input autoComplete="username" autoFocus />
          </Form.Item>
          <Form.Item name="password" label="密码" rules={[{ required: true, message: '请输入密码' }]}>
            <Input.Password autoComplete="current-password" />
          </Form.Item>
          <Button type="primary" htmlType="submit" block loading={busy}>
            登录
          </Button>
        </Form>
        <div style={{ marginTop: 12, textAlign: 'center' }}>
          <Link to="/register">没有账号?注册一个</Link>
        </div>
      </Card>
    </div>
  )
}
