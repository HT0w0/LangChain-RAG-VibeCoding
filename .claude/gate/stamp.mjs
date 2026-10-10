#!/usr/bin/env node
/**
 * 发通行证。检查员跑完之后调它，它写出一张带指纹的证。
 *
 *   node .claude/gate/stamp.mjs --kind unit
 *   node .claude/gate/stamp.mjs --kind quality --findings <一个 JSON 文件的路径>
 *
 * 用法（--kind unit）：
 *   它**自己跑四条命令**，全过才发证：
 *     · `pytest`        —— 后端单元测试（离线，不打真实 API）
 *     · `ruff check`    —— 后端质量门禁（**不跑 `ruff format`**，见 pyproject.toml）
 *     · `npm run test`  —— 前端 vitest
 *     · `npx tsc -b`    —— 前端类型检查
 *   任何一条不过 —— **不发证，并且把旧证删掉**（否则一张上次留下的旧证还能顶用）。
 *
 *   让「单测过了」从**模型的自我报告**变成**机器判定**，是这个装置的全部意义。
 *   你说你跑过了不算数，它自己跑出来的才算。
 *
 * 用法（--kind quality）：
 *   findings 文件长这样：
 *     { "blocking": ["必须改的"], "advisory": ["建议改的"], "summary": "一句话总评" }
 *   只有 `blocking` 非空会被闸门拦 —— 评审员的产出是「发现」而不是「对错」，
 *   实测它几乎每次都报点什么，全拦就没法用了。
 *
 * ⚠️ 它**自己会跑 `git add -A`**（冻住当前改动），并把冻了什么打印出来。
 * 为什么由它来做而不是要求调用者先 add：见下面 freeze() 的注释 ——
 * 一句话，「顺序搞反」这个坑从根上就不该存在。
 */
import { spawnSync } from 'node:child_process'
import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { fingerprint, KINDS, passportPath, repoRoot } from './lib.mjs'

const root = repoRoot()

/* ---------- 参数 ---------- */

function parseArgs(argv) {
  const out = { kind: null, findings: null }
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--kind') out.kind = argv[++i]
    else if (argv[i] === '--findings') out.findings = argv[++i]
  }
  return out
}

const { kind, findings } = parseArgs(process.argv.slice(2))

if (kind === null || !(kind in KINDS)) {
  process.stderr.write(
    `\n✗ 用法不对：--kind 只能是 ${Object.keys(KINDS).join(' 或 ')}\n\n` +
      `  例：node .claude/gate/stamp.mjs --kind unit\n\n`
  )
  process.exit(2)
}

const label = KINDS[kind].label
const file = passportPath(kind)

/* ---------- 第一道关：改动冻住了没有 ---------- */

/**
 * 为什么要在**发证的这一刻**就拦：指纹算的是「已冻结的内容」。
 * 如果调用者先发证、后 `git add`，那这张证记的是**旧**指纹，
 * 提交时闸门会拒 —— 而报错只会说「指纹对不上」，完全指不到「你顺序反了」。
 *
 * 把这个坑挡在源头，是本方案里最值钱的一道防线：**把静默失败变成响亮的失败。**
 */
function freeze() {
  // `-c core.quotePath=false` 是必须的：不加的话，中文文件名会被 git 转义成
  // 八进制（`".\346\234\252\345\206\273..."`），提示里就成了看不懂的东西 ——
  // 而这条提示的用处正是告诉人**哪些文件**被冻进去了。
  const res = spawnSync('git', ['-c', 'core.quotePath=false', 'add', '-A'], {
    encoding: 'utf8',
    cwd: root,
    shell: false
  })
  if (res.status !== 0) {
    process.stderr.write(`\n✗ 冻不住改动（git add -A 失败）：\n${res.stderr}\n\n`)
    process.exit(1)
  }

  // 把冻了什么列出来。不列的话，用户看不到这一步动了他的暂存区 ——
  // 一个会悄悄动你东西的工具，比一个多嘴的工具可怕得多。
  const st = spawnSync('git', ['-c', 'core.quotePath=false', 'status', '--porcelain'], {
    encoding: 'utf8',
    cwd: root,
    shell: false
  })
  const lines = (st.stdout ?? '').split('\n').filter((l) => l.trim() !== '')
  const added = lines.filter((l) => l.slice(0, 2) === 'A ' || l.slice(0, 2) === 'A')
  const modified = lines.filter((l) => l.slice(0, 2) === 'M ')
  process.stdout.write(
    `\n── 已把当前改动冻起来（git add -A）──\n` +
      `   新文件 ${added.length} 个，改动 ${modified.length} 个\n`
  )
  for (const l of [...added, ...modified].slice(0, 12)) {
    process.stdout.write(`     ${l}\n`)
  }
  if (added.length + modified.length > 12) {
    process.stdout.write(`     …还有 ${added.length + modified.length - 12} 个\n`)
  }
}

/**
 * 为什么**由本脚本自己**做冻结，而不是要求调用者先 `git add -A`：
 *
 * 通行证绑的是「已冻结内容」的指纹。要是让调用者自己冻，他就会把顺序搞反
 * （先发证、后 add），证记的指纹就是旧的 —— 提交时被拒，而报错只会说
 * 「指纹对不上」，**完全指不到真正的原因是顺序反了**。
 *
 * 更要命的是**并行**：tester 会新建测试文件，quality-engineer 可能在那些文件
 * 出现**之前**就发完证了。这时候谁先谁后都有问题。
 *
 * 所以冻结这一步收到本脚本里来做 —— 坑从根上就不存在了。
 * 编排者（/gitcommit）在两个检查员都回来之后会**重跑一遍发证**，
 * 让最终的指纹一定是「两个人都干完之后」的那一份。
 */
freeze()

/* ---------- 发证 ---------- */

function writePassport(extra) {
  const pass = {
    kind,
    label,
    fingerprint: fingerprint(),
    at: new Date().toISOString(),
    ...extra
  }
  writeFileSync(file, JSON.stringify(pass, null, 2) + '\n', 'utf8')
  return pass
}

/**
 * 检查没过时把旧证删掉。
 * 不删的话，上次留下的那张证（哪怕指纹已经对不上了）还躺在那里 ——
 * 闸门虽然会按指纹判它无效，但**看到文件还在的人会以为「证还在，应该没事」**。
 * 宁可不留痕迹。
 */
function discardOldPassport() {
  try {
    rmSync(file, { force: true })
  } catch {
    /* 删不掉就算了，反正闸门会按「没有证」处理 */
  }
}

/* ---------- 跑命令 ---------- */

/**
 * 找到虚拟环境里的 python。
 *
 * **不用裸 `python`**：那可能是系统里任何一个 python，装没装 pytest / ruff 都难说，
 * 而症状是「闸门说测试没过」—— 指不到真正的原因是解释器挑错了。
 * 找不到就明说，别猜。
 */
function venvPython() {
  const candidates =
    process.platform === 'win32' ? ['.venv/Scripts/python.exe'] : ['.venv/bin/python']
  for (const rel of candidates) {
    const abs = join(root, rel)
    if (existsSync(abs)) return abs
    process.stdout.write(`  （没有 ${rel}）\n`)
  }
  return null
}

/**
 * 跑一条命令，打印它输出的尾巴。
 *
 * `shell` 的取舍：
 *   · `npm` / `npx` 在 Windows 上是 `.cmd` 外壳脚本，**必须**过一层 shell，
 *     否则 CreateProcess 直接 ENOENT（这条踩过）。
 *   · venv 里的 python 是货真价实的 `.exe`，且路径里带空格
 *     （本仓库路径就带）—— 走 shell 反而会被空格切成两截。
 * 所以由调用点显式声明走不走 shell，不靠猜。
 */
function run(title, { cmd, args, cwd = root, shell = false }) {
  process.stdout.write(`\n── 正在跑：${title} ──\n`)
  const res = spawnSync(cmd, args, {
    encoding: 'utf8',
    cwd,
    shell,
    // 全量测试的输出可以很大；默认 1MB 会 ENOBUFS，而那个错误看起来像「测试挂了」
    maxBuffer: 64 * 1024 * 1024
  })

  // 把固定噪音滤掉。它们的共同点是**恰好排在输出的最后** ——
  // 不滤的话，「跑成功了」看起来就是「跑出来三条警告」，像是测试根本没跑。
  // 这一类错误信息本身也是静默的：看到的人在怀疑测试，而测试其实全过了。
  //
  // 两类噪音：
  //   · npm 的配置警告（`.npmrc` 里的镜像变量）
  //   · jsdom 的 `Not implemented: Window's getComputedStyle()` —— antd 会用到，
  //     只在 stderr 上吵，与测试结果无关（实测：62 条全过时它照样刷三条）
  const NOISE = /^(npm (warn|notice)|Not implemented:)/i
  const text = `${res.stdout ?? ''}${res.stderr ?? ''}`
    .split('\n')
    .filter((l) => !NOISE.test(l.trim()))
    .join('\n')

  const lines = text.split('\n').filter((l) => l.trim() !== '')
  if (res.status !== 0) {
    // 只留尾巴：失败信息通常在最后，全贴出来会淹掉重点
    process.stdout.write(lines.slice(-25).join('\n') + '\n')
  } else {
    process.stdout.write(lines.slice(-3).join('\n') + '\n')
  }
  return res.status === 0
}

/** 一条不过就拒发证，并把旧证删掉。 */
function fail(reason) {
  discardOldPassport()
  process.stderr.write(
    `\n✗ 没发「${label}」通行证 —— **${reason}**。\n` +
      `  （如果之前留着一张旧证，已经删掉了。）\n\n` +
      `  上面就是原始输出，我不改写它。修完再发证。\n\n`
  )
  process.exit(1)
}

/* ---------- 单测 + 静态检查 ---------- */

if (kind === 'unit') {
  const py = venvPython()
  if (py === null) {
    process.stderr.write(
      `\n✗ 没发「${label}」通行证 —— **找不到 .venv 里的 python**。\n\n` +
        `  先建虚拟环境并装依赖：\n` +
        `    python -m venv .venv\n` +
        `    .venv/Scripts/python.exe -m pip install -e ".[dev]"\n\n`
    )
    process.exit(1)
  }
  // `frontend/` 单独算一个工作目录：vitest 与 tsc 的配置都在那里。
  const web = join(root, 'frontend')

  // `-X utf8` 是必须的。Windows 下 python 往管道里写中文时会按 GBK 编码，
  // 而这里按 utf8 解码 —— 不加的话，错误信息本身就是一串乱码，
  // 而**错误信息正是这个装置存在的意义**。
  if (!run('pytest（后端单元测试）', { cmd: py, args: ['-X', 'utf8', '-m', 'pytest', '-q'] })) {
    fail('后端单元测试没过')
  }

  if (
    !run('ruff check（后端质量门禁）', {
      cmd: py,
      // **不跑 `ruff format`** —— 见 pyproject.toml 里那段：代码是按 100 列手工
      // 折行的，format 会把刻意的折行重新合并，产生一次覆盖 30+ 文件的 diff。
      args: ['-X', 'utf8', '-m', 'ruff', 'check', 'backend/app', 'backend/tests', 'backend/scripts']
    })
  ) {
    fail('ruff check 没过')
  }

  if (
    !run('npm run test（前端单元测试）', {
      cmd: 'npm',
      args: ['run', 'test'],
      cwd: web,
      shell: process.platform === 'win32'
    })
  ) {
    fail('前端单元测试没过')
  }

  if (
    !run('tsc -b（前端类型检查）', {
      cmd: 'npx',
      args: ['tsc', '-b'],
      cwd: web,
      shell: process.platform === 'win32'
    })
  ) {
    fail('前端类型检查没过')
  }

  const pass = writePassport({
    blocking: [],
    advisory: [],
    note: 'pytest / ruff check / vitest / tsc 都由发证脚本自己跑过，四条全过才发的这张证。'
  })
  process.stdout.write(
    `\n✓ 已发「${label}」通行证\n` +
      `  指纹 ${pass.fingerprint}\n` +
      `  （这张证绑着这份代码。改一行代码它就作废，得重新跑。）\n\n`
  )
  process.exit(0)
}

/* ---------- 质量评审 ---------- */

if (findings === null) {
  process.stderr.write(
    `\n✗ 发「${label}」通行证必须带 --findings <JSON 文件路径>\n\n` +
      `  文件长这样：\n` +
      `    { "blocking": ["必须改的"], "advisory": ["建议改的"], "summary": "一句话总评" }\n\n` +
      `  为什么不让评审员直接写通行证：通行证上的**指纹只能由本脚本填**，\n` +
      `  评审员碰不到 —— 否则它给一份旧代码发证也无人察觉。\n\n`
  )
  process.exit(2)
}

let raw
try {
  raw = JSON.parse(readFileSync(findings, 'utf8'))
} catch (e) {
  process.stderr.write(`\n✗ 读不了 findings 文件（或它不是合法 JSON）：\n  ${String(e.message ?? e)}\n\n`)
  process.exit(2)
}

const asList = (v) => (Array.isArray(v) ? v.filter((x) => typeof x === 'string' && x.trim() !== '') : null)
const blocking = asList(raw.blocking)
const advisory = asList(raw.advisory)

if (blocking === null || advisory === null) {
  process.stderr.write(
    `\n✗ findings 文件格式不对：blocking 和 advisory 都必须**是字符串数组**（可以是空数组）。\n` +
      `  （写成一句话、漏掉其中一个字段，都会被这里挡下。）\n\n`
  )
  process.exit(2)
}

const pass = writePassport({
  blocking,
  advisory,
  summary: typeof raw.summary === 'string' ? raw.summary : ''
})

process.stdout.write(
  `\n✓ 已发「${label}」通行证\n` +
    `  指纹 ${pass.fingerprint}\n` +
    `  必须改 ${blocking.length} 条，建议 ${advisory.length} 条\n` +
    `  ${blocking.length > 0 ? '⚠️ 有「必须改」，闸门会拦下这次提交。' : '没有「必须改」，闸门放行。'}\n\n`
)
process.exit(0)
