#!/usr/bin/env node
/**
 * 闸门 B —— Claude Code 的 PreToolUse 钩子，拦 Claude 自己发起的 `git commit`。
 *
 * 和闸门 A（git 的 pre-commit）**不是重复劳动**：
 *   · 闸门 A 谁提交都拦，**包括你在终端里手敲 git commit**。它管的是「结果」。
 *   · 闸门 B 只拦 Claude。它管的是「体验」—— 把闸门 A 那段英文报错换成一句中文，
 *     并且**提前**拦下，让 Claude 去补检查，而不是提交到一半撞一鼻子灰。
 *
 * 判据不在这里，在 `../gate/lib.mjs` 的 inspectPassports —— 与闸门 A 是同一份。
 * 两边各写一套的话，会出现「终端里拦得住、Claude 却提交成功」这种没人想得到的事。
 *
 * ## 实测记录（2026-10-06）
 *
 * 官方文档在本机被网络策略挡住，连不上。所以这里的字段名与拒绝格式
 * **都是实测出来的，不是抄的**（做法：写个只存 stdin 的探针钩子，跑一次看真实内容）：
 *
 *   · stdin 是 JSON，命令在 `tool_input.command`。
 *     顶层还有 `session_id` / `cwd` / `hook_event_name` / `tool_name` / `tool_use_id`。
 *   · 钩子进程的工作目录 = 项目根；PATH 里有 node。
 *   · 拒绝必须写 `hookSpecificOutput.permissionDecision`，
 *     而且**必须带 `hookEventName`** —— 实测不带它**完全不生效**：
 *     命令照跑，钩子白装，一声不吭。（这条最坑，所以量过。）
 *   · 拦截时界面上一律显示成 `hook error:`，这是这个版本的表现，改不掉。
 *     **所以理由文字必须自己把话说全**，不能指望外面那层壳帮你解释。
 */
import { HOW_TO_FIX, inspectPassports } from '../gate/lib.mjs'
import { detectGitCommit } from './detect.mjs'

/* ---------- 读 stdin ---------- */

let raw = ''
process.stdin.setEncoding('utf8')
process.stdin.on('data', (c) => {
  raw += c
})
process.stdin.on('end', () => {
  main()
})

/* ---------- 主流程 ---------- */

function main() {
  let payload = null
  try {
    payload = JSON.parse(raw)
  } catch {
    return // 读不懂就放行 —— 钩子自己出问题不该拦住人干活
  }

  if (payload?.tool_name !== 'Bash') return
  const command = String(payload?.tool_input?.command ?? '')
  if (command.trim() === '') return

  const commitArgs = detectGitCommit(command)
  if (commitArgs === null) return

  // 只是看看帮助、或者演练，不真提交 —— 不管
  if (commitArgs.some((a) => a === '--help' || a === '-h' || a === '--dry-run')) return

  // 第一条：不许用 --no-verify 绕闸门。
  // 这是本方案**唯一一条防人的**，其它都是防手滑。git 自带的逃生门堵不死，
  // 但至少不能让 Claude 顺手用了它 —— 那等于把整个装置拆掉。
  const noVerify = commitArgs.some(
    (a) => a === '--no-verify' || a === '-n' || a === '--no-verify=true'
  )
  if (noVerify) {
    deny(
      '不许用 --no-verify 跳过检查。\n\n' +
        '这道闸门是「提交前必须过单元测试和质量评审」的装置，--no-verify 正好就是绕过它的开关 ——\n' +
        '从 Claude 这边用，等于把整个装置拆掉。\n\n' +
        '要正经提交：说一句 /gitcommit 。\n' +
        '确实需要紧急跳过（比如软件已经坏了、必须先存档再说）：请**你自己**在终端里敲\n' +
        '  git commit --no-verify\n' +
        '这件事得由你来做，我不代劳。'
    )
    return
  }

  // 第二条：查通行证（与闸门 A 同一份判据）
  let result
  try {
    result = inspectPassports()
  } catch {
    // 钩子自己坏了 —— 放行，交给闸门 A 去拦。绝不能因为钩子出问题就替用户做决定。
    return
  }

  if (result.fatal !== null) return // 同上，交给闸门 A
  if (result.problems.length === 0) return // 有证、且是这份代码的、没有「必须改」→ 放行

  deny(
    '提交被拦下 —— 没有有效的通行证。\n\n' +
      result.problems.map((p) => `  ${p}`).join('\n') +
      '\n\n' +
      HOW_TO_FIX +
      '\n  （闸门 B 只是提前告诉你一声。真要硬提交，git 那边的 pre-commit 也会拦住。）'
  )
}

function deny(reason) {
  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        permissionDecision: 'deny',
        permissionDecisionReason: reason
      }
    })
  )
}
