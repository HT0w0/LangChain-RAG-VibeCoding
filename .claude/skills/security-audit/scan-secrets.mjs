#!/usr/bin/env node
// 安全审计的机械部分 —— 口径写死在这里，不许调用方凭印象说「我 grep 过了」。
//
// 为什么要有这个脚本：2026-10-06 基线实测，三个 Claude 对同一个仓库各报各的，
// 没有一份说清楚「我扫的是工作区还是历史」「远端上到底有什么」。这两个问题的
// 答案对公开仓库来说是天差地别的：
//   · 工作区干净 ≠ 历史干净（密钥提交过又删掉，仍然躺在历史里）
//   · 本地有什么 ≠ 远端有什么（本机可能有一条永远不该推的分支）
//
// 本脚本只做「机械可判定」的两件事，剩下靠脑子判断（读 SKILL.md）：
//   1. 按已知密钥样式扫工作区 + 扫全部 git 历史
//   2. 列出「本地有、远端没有」的引用，以及它们各自多带了哪些文件
//
// 用法：node .claude/skills/security-audit/scan-secrets.mjs [仓库目录，默认当前目录]

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const repo = path.resolve(process.argv[2] ?? '.')

function git(args, opts = {}) {
  // `-c core.quotepath=false`：不加这条，git 会把中文文件名转义成
  // `"docs/\345\274\200\345\217\221\350\256\260\345\275\225.md"` 这种样子，
  // 用户根本认不出是哪个文件 —— 而这一栏报出来的文件名恰恰是他要判断风险的全部依据。
  // `-C repo` 让所有命令都在目标仓库里跑，调用方不必先 cd。
  return execFileSync('git', ['-c', 'core.quotepath=false', '-C', repo, ...args], {
    encoding: 'utf8',
    maxBuffer: 256 * 1024 * 1024,
    ...opts
  })
}

function tryGit(args) {
  try {
    // `stdio` 必须显式指定。不指的话，`execFileSync` 会把子进程的 stderr
    // **直通到本进程的 stderr** —— 于是一句探路的 `fatal: Needed a single revision`
    // 就漏到用户眼前，看起来像脚本坏了。探路的失败是**预期内**的，不该出声。
    return git(args, { stdio: ['ignore', 'pipe', 'pipe'] })
  } catch {
    return null
  }
}

// ── 已知密钥样式 ───────────────────────────────────────────────
// 只放「一望即知是凭据」的样式。宽泛的 password/secret 由下面 GENERIC 单独处理，
// 因为它们在正常代码里大量出现（表单字段名、变量名），会淹掉真正的命中。
const PATTERNS = [
  ['GitHub token', /\b(ghp|gho|ghs|ghu)_[A-Za-z0-9]{36,}\b/],
  ['GitHub PAT', /\bgithub_pat_[A-Za-z0-9_]{20,}\b/],
  ['AWS Access Key', /\bAKIA[0-9A-Z]{16}\b/],
  ['Google API key', /\bAIza[0-9A-Za-z_-]{35}\b/],
  ['Slack token', /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/],
  // 百炼（DashScope）工作空间 key 是本项目的**主凭据**，必须单列一条。
  // 不能指望下面那条通用的 `sk-` 样式：它是 `sk-` 后接 20 个以上的
  // `[A-Za-z0-9_-]`，而百炼的 key 形如 `sk-ws-<段>.<段>` ——
  // **中间有点号**，通用样式在那里就断了，长度也未必够 20。
  // （这条不是猜的：写通用样式时正因为它漏掉了点号，才专门补了这一条。）
  ['百炼 DashScope key', /\bsk-(?:ws-)?[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_-]+)*/],
  ['OpenAI 风格 key', /\bsk-[A-Za-z0-9_-]{20,}\b/],
  ['私钥文件内容', /-----BEGIN [A-Z ]*PRIVATE KEY-----/],
  ['npm token', /\/\/[^\s]*:_authToken=\S+/]
]

// 赋值式的凭据：要求右侧是被引号包起来的字面量，避免命中 `password: string` 这类类型声明
const GENERIC = [
  ['赋值的 password 字面量', /\b(password|passwd|pwd)\s*[:=]\s*['"][^'"]{3,}['"]/i],
  ['赋值的 secret 字面量', /\b(secret|api[_-]?key|apikey|access[_-]?key)\s*[:=]\s*['"][^'"]{6,}['"]/i]
]

// GitHub Actions 里这两种写法是**正确**的，不是泄露 —— 报出来就是假警报。
const FINE = [/\$\{\{\s*secrets\.[A-Za-z0-9_]+\s*\}\}/, /\$\{\{\s*github\.token\s*\}\}/]

// 占位符与测试哨兵，**不是泄露**。
//
// 为什么本项目必须单独列这一组：`.env.example` 里那行本来就是给克隆的人照抄的模板，
// `conftest.py` 里的哨兵更是**刻意**写成「一眼看出不是真钥匙」的样子。
// 不滤掉的话，每次扫描都稳定报 6 条噪音 —— 而**一道总在响的警报等于没有警报**，
// 真出了事，那条会淹在这 6 条里。
//
// ⚠️ 判据要收紧到「一眼就是假的」：`sk-test-`、`sk-ws-xxxx`、`your_`、`<...>`。
// **别往前缀上放**（比如整条 `sk-ws-` 全放行）—— 那正好把真的放过。
const PLACEHOLDER = [
  /sk-(?:test|fake|dummy|your|example|placeholder)[A-Za-z0-9_-]*/i,
  /sk-[A-Za-z-]*x{4,}/i,
  /(?:API_KEY|SECRET|TOKEN|PASSWORD)\s*[:=]\s*["']?(?:your|xxx|placeholder|changeme|\*{3,}|<[^>]*>)/i,
  /请替换|占位|示例|替换为/
]

function scan(text, label) {
  const hits = []
  for (const line of text.split(/\r?\n/)) {
    if (FINE.some((re) => re.test(line))) continue
    if (PLACEHOLDER.some((re) => re.test(line))) continue
    for (const [name, re] of [...PATTERNS, ...GENERIC]) {
      if (re.test(line)) {
        hits.push({ name, line: line.trim().slice(0, 160) })
        break
      }
    }
  }
  return hits
}

console.log(`\n仓库：${repo}\n`)

// ── 1. 工作区 ────────────────────────────────────────────────
// 用 `git grep` 而不是递归读文件：它天然只扫**被 git 跟踪**的文件，
// 于是 node_modules / release / out 这些没跟踪的目录不会拖慢扫描、也不会造成噪音。
const worktree = tryGit(['grep', '-nI', '-E', '.', '--', '.']) ?? ''
const wtHits = scan(worktree, 'worktree')
console.log('=== 1. 密钥样式扫描 · 工作区（只扫被 git 跟踪的文件）===')
if (wtHits.length === 0) console.log('  无命中')
else for (const h of wtHits) console.log(`  [${h.name}] ${h.line}`)

// ── 2. 历史 ──────────────────────────────────────────────────
// 这一步是基线全都漏掉的：工作区删干净了，历史里可能还留着。
const history = tryGit(['log', '-p', '--all', '--no-color'])
console.log('\n=== 2. 密钥样式扫描 · 全部 git 历史（含本地未推送的分支）===')
if (history === null) {
  console.log('  读不到历史（可能不是 git 仓库）')
} else {
  const hisHits = scan(history, 'history')
  if (hisHits.length === 0) console.log('  无命中')
  else for (const h of hisHits) console.log(`  [${h.name}] ${h.line}`)
}

// ── 3. 本地引用 vs 远端 ──────────────────────────────────────
// 这是「公开仓库」最要命的一栏：本机可能有分支带着不该公开的文件。
console.log('\n=== 3. 本地引用 vs 远端 ===')

const remotes = (tryGit(['remote']) ?? '').split('\n').filter(Boolean)
const remote = tryGit(['rev-parse', '--verify', 'origin/main'])
if (remotes.length === 0) {
  // 没配远端时 `rev-parse` 会往 stderr 甩一句 `fatal: Needed a single revision`，
  // 直接漏出去会让人以为脚本坏了。这里说清楚：**没远端不是问题，但要知道**。
  console.log('  这个仓库没有配置任何远端 —— 所以「远端上有什么」这一问现在没有答案。')
  console.log('  （作品集仓库迟早要推上去，推之前**必须**再跑一次本节。）')
} else if (remote === null) {
  console.log(`  配了远端（${remotes.join(', ')}），但没有 origin/main —— 分支名可能不叫这个`)
} else {
  const remoteFiles = new Set(
    (tryGit(['ls-tree', '-r', '--name-only', 'origin/main']) ?? '').split('\n').filter(Boolean)
  )
  console.log(`  远端 origin/main：${remoteFiles.size} 个文件`)

  const refs = (tryGit(['for-each-ref', '--format=%(refname)', 'refs/heads', 'refs/original']) ?? '')
    .split('\n')
    .filter(Boolean)

  // 每个引用都要表态。「干净的不打印」看似清爽，实则分不清
  // 「查过、没风险」和「压根没查」—— 这一栏存在的全部意义就是让人知道查没查。
  console.log('  本地引用逐个核对（与远端比，多出来的文件才是风险）：')
  const risky = []
  for (const ref of refs) {
    const files = (tryGit(['ls-tree', '-r', '--name-only', ref]) ?? '').split('\n').filter(Boolean)
    const extra = files.filter((f) => !remoteFiles.has(f))
    if (extra.length === 0) {
      console.log(`    ${ref.padEnd(42)}与远端一致，无可疑文件`)
    } else {
      risky.push({ ref, extra })
      console.log(`    ${ref.padEnd(42)}⚠️ 多 ${extra.length} 个文件`)
    }
  }

  for (const { ref, extra } of risky) {
    console.log(`\n  ⚠️  ${ref} —— 比远端多出这些文件，只存在本机：`)
    for (const f of extra.slice(0, 15)) console.log(`        ${f}`)
    if (extra.length > 15) console.log(`        …还有 ${extra.length - 15} 个`)
  }
  if (risky.length > 0) {
    console.log(
      `\n  ⚠️ 上面这 ${risky.length} 个引用**一旦被推上去，那些文件就公开了**。` +
        `\n     绝对不要用 git push --all / --mirror / --tags。`
    )
  }
}

// ── 4. 运行时数据有没有进仓库 ──────────────────────────────────
//
// 这一节是本项目特有的，也是**最容易出事的一节**。
// `data/` 下面是真实的东西：开发用的 SQLite 库（里面有真实注册的用户、
// 真实对话历史）、上传的原始文档、Chroma 的向量索引。
// 它们一旦进了公开仓库，泄露的不是「一个密钥」，而是**用户的全部对话记录**
// —— 而删掉文件也撤不回来（历史里还在）。
//
// 密钥有肉眼可见的样式（上面那些正则），**数据库文件没有** ——
// 它看起来就是个普通的二进制。所以这一节不能靠「扫样式」，只能直接问 git。
console.log('\n=== 4. 运行时数据（data/）有没有进过仓库 ===')

const DATA_PATTERNS = ['data', '*.db', '*.db-wal', '*.db-shm', '*.sqlite3']
let dirty = false

for (const pat of DATA_PATTERNS) {
  const tracked = (tryGit(['ls-files', '--', pat]) ?? '').split('\n').filter(Boolean)
  if (tracked.length > 0) {
    dirty = true
    console.log(`  ⚠️  ${pat} —— 当前被 git 跟踪 ${tracked.length} 个文件：`)
    for (const f of tracked.slice(0, 10)) console.log(`        ${f}`)
    if (tracked.length > 10) console.log(`        …还有 ${tracked.length - 10} 个`)
  }
}

// 更阴的一问：现在没跟踪，不代表**过去**没跟踪过。
const everData = (tryGit(['log', '--all', '--oneline', '--', 'data']) ?? '').split('\n').filter(Boolean)
if (everData.length > 0) {
  dirty = true
  console.log(`  ⚠️  data/ 在历史里出现过（${everData.length} 个提交）—— **现在删掉也撤不回来**：`)
  for (const l of everData.slice(0, 5)) console.log(`        ${l}`)
  console.log('     处置：改写历史（filter-repo / BFG），或**换一把新钥匙、换一个仓库**。')
}

// 只读一遍 .gitignore，报事实。这里不替用户判断「忽略得对不对」。
const gi = (() => {
  try {
    return fs.readFileSync(path.join(repo, '.gitignore'), 'utf8')
  } catch {
    return null
  }
})()
if (gi === null) {
  console.log('  ⚠️  读不到 .gitignore —— 没有它的话 `git add -A` 会把 data/ 整个收进去。')
} else {
  const covers = DATA_PATTERNS.filter((p) => gi.includes(p))
  console.log(
    covers.length > 0
      ? `  .gitignore 覆盖了：${covers.join(' · ')}`
      : '  ⚠️  .gitignore 里**没有**覆盖 data/ —— git add -A 会把它整个收进去。'
  )
}

if (!dirty) console.log('  ✅ 运行时数据没有被跟踪过（工作区与历史都干净）')

console.log('')
