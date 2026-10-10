#!/usr/bin/env node
// 注释比例统计 —— 口径写死在这里，不许调用方自己心算。
//
// 为什么要有这个脚本：同一个仓库，三个人各数各的，会数出三个不一样的数
// （块注释的续行算不算、行尾行内注释算不算、测试文件算不算）。
// 数字不可复现，标准就是空的。
//
// ── 本项目的口径 ────────────────────────────────────────────────
//
//   TS/TSX：`//` 独占一行、`/* */` 整块。行尾的行内注释不算。
//
//   Python：**`#` 独占一行 + docstring 的正文行。**
//
//   ⚠️ 「docstring 算注释」这一条是本项目特有的，且是必须的 ——
//   实测（2026-10-09）backend/app 只数 `#` 是 12.7%，把 docstring 算上才是 33.6%。
//   这个项目的「为什么」几乎全写在 docstring 里（`store_upload` 为什么用 uuid、
//   `security.py` 为什么 JWT 不放 role 都是）。只数 `#` 的话，工具会对着一个
//   注释写得比谁都认真的目录报「需要补注释」—— **假警报比没有警报更坏**，
//   它会逼出 `# 定义变量` 那种凑数的废话，把真正有用的注释淹掉。
//
//   docstring 的判定：**三引号开在一行的行首**（允许 `r"""` 这类前缀）。
//   `x = """…"""` 这种赋值不算 —— 那是数据，不是注释。
//
// 用法：node .claude/skills/comments-check/count-comments.mjs [仓库根，默认 .]

import fs from 'node:fs'
import path from 'node:path'

const argv = process.argv.slice(2)
const root = path.resolve(argv.find((a) => !a.startsWith('--')) ?? '.')

// 分档标准（跟 SKILL.md 里那张表必须一致）。
//
// 阈值是**实测值向下取整到 5 的倍数**，不是拍脑袋定的：
// 实测 33.6% / 17.6% / 29.6% → 门槛 30% / 15% / 25%。
// 留出余量，是为了让它平时是绿的 —— **一道总在响的警报等于没有警报**。
const BUCKETS = [
  { key: 'backend', label: '后端 app', min: 30, dirs: ['backend/app'], exts: ['.py'] },
  { key: 'frontend', label: '前端 src', min: 15, dirs: ['frontend/src'], exts: ['.ts', '.tsx'] }
]
const OVERALL_MIN = 25

// 这两块只报数、不设要求：脚本是探针与演示（联网才跑得起来），
// 测试文件的注释回答的是另一个问题（这条用例防什么）。
const REFERENCE = [
  { label: '后端 scripts', dirs: ['backend/scripts'], exts: ['.py'] },
  { label: '前端 src 同目录测试', dirs: ['frontend/src'], exts: ['.ts', '.tsx'], tests: true }
]

const isTest = (p) => /\.test\.(ts|tsx|js|jsx)$/.test(p) || /(^|\/)tests?\//.test(p)

function walk(dir, exts, acc = []) {
  let entries
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true })
  } catch {
    return acc // 目录不存在就当空的，由下面统一报「没扫到文件」
  }
  for (const e of entries) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) {
      if (e.name === 'node_modules' || e.name === '__pycache__') continue
      walk(p, exts, acc)
    } else if (exts.some((x) => e.name.endsWith(x)) && !e.name.endsWith('.d.ts')) {
      acc.push(p.split(path.sep).join('/'))
    }
  }
  return acc
}

function countTs(text) {
  let comment = 0
  let code = 0
  let inBlock = false
  for (const raw of text.split(/\r?\n/)) {
    const t = raw.trim()
    if (!t) continue
    if (inBlock) {
      comment++
      if (t.includes('*/')) inBlock = false
      continue
    }
    if (t.startsWith('/*')) {
      comment++
      if (!t.includes('*/')) inBlock = true
      continue
    }
    if (t.startsWith('//')) {
      comment++
      continue
    }
    code++
  }
  return { comment, code, total: comment + code }
}

// 用**命名组**，不用编号 —— 编号踩过坑，见下面 countPy 的注释。
const DOC_OPEN = /^[rubfRUBF]{0,2}(?<quote>"""|''')/

/**
 * 数一个 Python 文件的注释与代码行。
 *
 * ⚠️ 这里**必须**用命名组。踩过一次：写成了 `m[2]`，而 `[rubfRUBF]{0,2}`
 * 是字符类、不占捕获组，于是拿到 `undefined` —— 它既不等于 `null`（关不掉
 * docstring），又永远不会出现在任何一行里（`t.includes(undefined)` 找的是
 * 字符串 "undefined"）。症状是**整个文件 100% 算成注释**，而脚本一声不吭
 * 地报出一个漂亮的数字。**这就是本项目最讨厌的那类失败**，所以下面加了断言。
 */
function countPy(text) {
  let comment = 0
  let code = 0
  let inDoc = null // 开着的是哪个三引号（""" 或 '''）—— 必须记住它，别用另一个关
  let n = 0
  for (const raw of text.split(/\r?\n/)) {
    n++
    const t = raw.trim()
    if (!t) continue
    if (inDoc !== null) {
      comment++
      if (t.includes(inDoc)) inDoc = null
      continue
    }
    const m = t.match(DOC_OPEN)
    if (m) {
      const quote = m.groups.quote
      if (typeof quote !== 'string') {
        throw new Error(`docstring 判定坏了：${file}:${n} —— 捕获组没拿到引号`)
      }
      comment++
      // 单行的 `"""一句话"""` 里有两个引号 → 不是块的开头。用 split 数，
      // 不用 indexOf —— 后者对 `""""""` 这种会数错。
      if (t.split(quote).length - 1 < 2) inDoc = quote
      continue
    }
    if (t.startsWith('#')) {
      comment++
      continue
    }
    code++
  }
  return { comment, code, total: comment + code }
}

const count = (file) => {
  const text = fs.readFileSync(file, 'utf8')
  return file.endsWith('.py') ? countPy(text) : countTs(text)
}

/* ---------- 自检 ---------- */
//
//   node .claude/skills/comments-check/count-comments.mjs --selftest
//
// 为什么要有：这个脚本**出错的后果是静默的** —— 报出一个漂亮的百分比，
// 而它是错的。（2026-01-09 实测踩到过：捕获组读错一位，整个 Python 目录
// 被算成 100% 注释，脚本一声不吭。）
//
// 所以下面两类用例都要有：
//   · 该数对的（数错 = 报出假数字）
//   · **不该数进去的**（数错 = 逼着人去补废话注释）

/** [语言, 说明, 源码, 期望注释行数, 期望代码行数] */
const SELF_CASES = [
  ['py', '单行 docstring', '"""一句话。"""\n', 1, 0],
  ['py', '块 docstring（开、正文、关都算）', '"""开头\n正文\n"""\n', 3, 0],
  ['py', '`#` 注释', '# 说明\n', 1, 0],
  ['py', '普通代码行', 'x = 1\n', 0, 1],
  ['py', '赋值的三引号是**数据**不是注释', 'SQL = """SELECT 1"""\nreturn SQL\n', 0, 2],
  // 这条是**判别性**的：若「任意三引号都能关掉」，`'''` 会在第 2 行关掉、
  // 第 3 行的 `"""` 又开一个新的，于是 `x = 1` 被算成注释（4/0 而不是 3/1）。
  ['py', '只有**配对**的引号关得掉', '"""开着\n\'\'\'\n"""\nx = 1\n', 3, 1],
  // 注意这里 docstring 只有**两**行（开的 `"""模块` + 关的 `文档"""`），
  // 所以是 3 条注释，不是 4 —— 一开始期望写成 4 的是我，脚本是对的。
  ['py', '混合', '"""模块\n文档"""\nimport os\n\n# 为什么\nx = 1\n', 3, 2],
  ['py', '带 r 前缀的 docstring', 'r"""原样\n字符串"""\ny = 2\n', 2, 1],
  ['ts', '行注释', '// 说明\n', 1, 0],
  ['ts', '块注释（每行都算）', '/* 块\n续行 */\n', 2, 0],
  ['ts', '行尾的行内注释**不算**', 'const x = 1 // 行尾\n', 0, 1],
  ['ts', '混合', '// 说明\nconst x = 1\n', 1, 1]
]

if (argv.includes('--selftest')) {
  let bad = 0
  console.log('\n注释统计口径自检\n')
  for (const [lang, note, src, wantC, wantK] of SELF_CASES) {
    const got = lang === 'py' ? countPy(src) : countTs(src)
    const ok = got.comment === wantC && got.code === wantK
    if (!ok) bad++
    console.log(
      `  ${ok ? '✅' : '❌'} ${note}\n` +
        `     期望 注释 ${wantC} / 代码 ${wantK}；实际 注释 ${got.comment} / 代码 ${got.code}`
    )
  }
  console.log(`\n合计 ${SELF_CASES.length} 项：通过 ${SELF_CASES.length - bad}，失败 ${bad}\n`)
  process.exit(bad > 0 ? 1 : 0)
}

function collect({ dirs, exts, tests }) {
  const files = dirs.flatMap((d) => walk(path.join(root, d), exts))
  const picked = files.filter((f) => (tests ? isTest(f) : !isTest(f)))
  const rows = picked.map((f) => ({ f, ...count(f) })).filter((r) => r.total > 0)
  return {
    rows,
    comment: rows.reduce((s, r) => s + r.comment, 0),
    total: rows.reduce((s, r) => s + r.total, 0)
  }
}

const pct = (c, t) => (t ? (c / t) * 100 : 0)
const flag = (v, min) => (v >= min ? '✅' : '❌')
const fmt = (c, t) => `${pct(c, t).toFixed(1).padStart(5)}%`

console.log(`\n注释比例（口径：非空行为分母，只算独占一行的注释；Python 计入 docstring）\n`)

const buckets = BUCKETS.map((b) => ({ ...b, ...collect(b) }))
const overallC = buckets.reduce((s, b) => s + b.comment, 0)
const overallT = buckets.reduce((s, b) => s + b.total, 0)

console.log(
  `  整体（产品代码）  ${fmt(overallC, overallT)}   要求 ${OVERALL_MIN}%  ${flag(pct(overallC, overallT), OVERALL_MIN)}`
)
for (const b of buckets) {
  console.log(
    `  ${b.label.padEnd(14)}  ${fmt(b.comment, b.total)}   要求 ${b.min}%  ${flag(pct(b.comment, b.total), b.min)}`
  )
}

console.log('')
for (const r of REFERENCE) {
  const got = collect(r)
  console.log(`  [参考] ${r.label.padEnd(20)} ${fmt(got.comment, got.total)}   （不计入上表，不设要求）`)
}

const below = []
for (const b of buckets) {
  for (const r of b.rows) {
    const v = pct(r.comment, r.total)
    if (v < b.min) below.push({ ...r, bucket: b.label, min: b.min, pct: v })
  }
}
below.sort((a, b) => a.pct - b.pct)

if (below.length === 0) {
  console.log('\n  没有低于门槛的源文件。\n')
} else {
  console.log(`\n  低于门槛的源文件（${below.length} 个）—— ⚠️ 列在这里是让人去看，不是让人去补废话：\n`)
  // **只列最低的 15 个。** 本项目实测有 33 个文件低于各自的档位 ——
  // 超过一半的文件都上榜的话，这列表就不再是「去看哪里」，而是一屏噪音，
  // 没人会读。真要看全的就照下面这个阈值自己排。
  for (const r of below.slice(0, 15)) {
    console.log(
      `    ${r.pct.toFixed(1).padStart(5)}%  ${String(r.comment).padStart(4)}/${String(r.total).padStart(4)}  ${r.f}`
    )
  }
  if (below.length > 15) console.log(`    …还有 ${below.length - 15} 个`)
  console.log('\n  「注释少」和「注释该补」是两件事。判断一条注释该不该存在，只问一句：')
  console.log('  删掉它，读代码的人会失去什么？ —— 见 SKILL.md 第三节。\n')
}
