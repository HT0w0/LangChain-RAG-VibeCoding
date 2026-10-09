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
    <div className="auth-page">
      <Card className="auth-card">
        <div className="auth-mark">知</div>
        <Typography.Title level={4} className="auth-title">
          电商知识库问答
        </Typography.Title>
        {/* 副标题说清「这是什么」——登录页是唯一一个用户还不知道
            这个系统能干什么的地方。 */}
        <Typography.Text className="auth-subtitle">
          基于企业自有资料作答,每句结论都可溯源
        </Typography.Text>

        {error && <Alert type="error" message={error} showIcon style={{ marginBottom: 16 }} />}
        <Form<FormValues> layout="vertical" onFinish={onFinish} disabled={busy}>
          <Form.Item
            name="username"
            label="用户名"
            rules={[{ required: true, message: '请输入用户名' }]}
          >
            <Input size="large" autoComplete="username" autoFocus placeholder="admin" />
          </Form.Item>
          <Form.Item
            name="password"
            label="密码"
            rules={[{ required: true, message: '请输入密码' }]}
          >
            <Input.Password size="large" autoComplete="current-password" />
          </Form.Item>
          <Button type="primary" size="large" htmlType="submit" block loading={busy}>
            登录
          </Button>
        </Form>
        <div style={{ marginTop: 14, textAlign: 'center' }}>
          <Link to="/register">没有账号?注册一个</Link>
        </div>
      </Card>
    </div>
  )
}
