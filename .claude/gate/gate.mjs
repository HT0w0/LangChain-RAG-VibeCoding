#!/usr/bin/env node
/**
 * 闸门 A —— 由 git 的 pre-commit 钩子调用。
 *
 * 它**只做一件事：查通行证**。
 *   · 在不在？
 *   · 是不是**这份代码**的？（指纹对不对）
 *   · 里面有没有写着「必须改」？
 * 全过 → 放行（退出码 0）。任何一条不过 → 拒绝（退出码 1）。
 *
 * ⚠️ 它**不跑任何检查**、**不懂什么是测试**。
 * 这是刻意的：闸门越简单越好，它永远不会因为「找不到 python」「网络抽风」
 * 而误拦一份本来没问题的代码。
 *
 * 代价（必须认）：**闸门的可信度 = 通行证的可信度。**
 * 通行证要是白条，闸门就是个摆设。所以「单测」那张证由 `stamp.mjs`
 * **自己跑一遍 pytest / ruff / vitest / tsc** 才发。
 *
 * 判据在 lib.mjs 的 inspectPassports 里 —— 闸门 B 用的是同一份。
 * **别在这里另写一套判断**，两边跑偏了没人会发现。
 */
import { HOW_TO_FIX, inspectPassports } from './lib.mjs'

const { fatal, problems } = inspectPassports()

if (fatal !== null) {
  process.stderr.write(
    `\n✗ 提交被拦下（闸门自己出了问题，不是你代码的问题）\n\n  ${fatal}\n\n` +
      `  紧急时跳过这次拦截：git commit --no-verify\n\n`
  )
  process.exit(1)
}

if (problems.length > 0) {
  process.stderr.write('\n✗ 提交被拦下 —— 没有有效的通行证\n\n')
  for (const p of problems) process.stderr.write(`  ${p}\n`)
  process.stderr.write(
    '\n' + HOW_TO_FIX + '\n  紧急时跳过这次拦截（自己承担风险）：git commit --no-verify\n\n'
  )
  process.exit(1)
}

process.exit(0)
