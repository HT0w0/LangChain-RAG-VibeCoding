import { Alert, Button, Card, Form, Input, Typography } from 'antd'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ApiError } from '../api/client'
import { useAuth } from '../store/auth'

interface FormValues {
  username: string
  password: string
  confirm: string
}

//: 与后端 `app/schemas/auth.py` 的规则保持一致。
//:
//: **这里是提前反馈,不是校验。** 后端的校验才是真的——前端少一条规则,
//: 用户提交后拿到后端的中文提示;前端多一条规则,用户会被拦在一个
//: 后端其实允许的输入上。两边不一致时以后端为准,所以文案也照后端抄。
const USERNAME_RULE = /^[A-Za-z0-9_\-一-鿿]{3,32}$/

export function RegisterPage() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function onFinish(values: FormValues) {
    setBusy(true)
    setError('')
    try {
      await register(values.username, values.password)
      navigate('/chat', { replace: true })
    } catch (err) {
      setError(err instanceof ApiError ? err.message : '注册失败,请稍后重试')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-page">
      <Card className="auth-card">
        <div className="auth-mark">知</div>
        <Typography.Title level={4} className="auth-title">
          注册
        </Typography.Title>
        <Typography.Text className="auth-subtitle">
          注册后即可提问;知识库由管理员维护
        </Typography.Text>

        {error && <Alert type="error" message={error} showIcon style={{ marginBottom: 16 }} />}
        <Form<FormValues> layout="vertical" onFinish={onFinish} disabled={busy}>
          <Form.Item
            name="username"
            label="用户名"
            rules={[
              { required: true, message: '请输入用户名' },
              { pattern: USERNAME_RULE, message: '3-32 位,仅限中英文、数字、下划线和连字符' },
            ]}
          >
            <Input autoComplete="username" autoFocus />
          </Form.Item>
          <Form.Item
            name="password"
            label="密码"
            rules={[
              { required: true, message: '请输入密码' },
              { min: 6, message: '密码至少 6 位' },
            ]}
          >
            <Input.Password autoComplete="new-password" />
          </Form.Item>
          <Form.Item
            name="confirm"
            label="确认密码"
            dependencies={['password']}
            rules={[
              { required: true, message: '请再输入一次密码' },
              // 用 `getFieldValue` 而不是把密码绑到 state 上:多存一份
              // 密码就多一份被无意中渲染进 DOM 或日志的机会。
              ({ getFieldValue }) => ({
                validator: (_rule, value: string) =>
                  value === getFieldValue('password')
                    ? Promise.resolve()
                    : Promise.reject(new Error('两次输入的密码不一致')),
              }),
            ]}
          >
            <Input.Password autoComplete="new-password" />
          </Form.Item>
          <Button type="primary" htmlType="submit" block loading={busy}>
            注册并登录
          </Button>
        </Form>
        <div style={{ marginTop: 12, textAlign: 'center' }}>
          <Link to="/login">已有账号?去登录</Link>
        </div>
      </Card>
    </div>
  )
}
