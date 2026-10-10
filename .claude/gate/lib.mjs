/**
 * 闸门与发证脚本共用的一小块。
 *
 * 只放两样东西：**怎么算指纹**、**怎么读通行证**。
 * 这两样必须只有一份实现 —— 闸门和发证各写一份的话，
 * 两边只要算得不一样（差一个空格、差一次 cwd），症状就是
 * 「明明发过证了，提交却永远被拒」，而且报错只会说「指纹对不上」，
 * 完全指不到真正的原因。这正是这个项目里最要紧的一类 bug：
 * **写错不报错、只是静默失效**（CLAUDE.md §5.1）。
 */
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

/** `.claude/gate/` 的绝对路径（就是本文件所在目录） */
export const GATE_DIR = dirname(fileURLToPath(import.meta.url))

/** 仓库根目录。不靠 cwd 猜 —— git 自己知道。 */
export function repoRoot() {
  return execFileSync('git', ['rev-parse', '--show-toplevel'], { encoding: 'utf8' }).trim()
}

/**
 * 指纹 = 「即将提交的内容」的哈希，交给 git 自己算（`git write-tree`）。
 *
 * 为什么用它：
 * - 它由 git 计算，检查员**填不了**（通行证里的指纹是发证脚本自己填的）。
 * - 它只认**已冻结（git add 过）**的内容，所以「改一行代码」必然让旧证作废。
 * - 它是内容寻址的：同样的内容永远算出同样的哈希，不受时间、路径影响。
 *
 * ⚠️ 用它就必须守一个顺序（见 CLAUDE.md / 计划里的「决定 3」）：
 *    **所有检查跑完 → git add -A → 发证 → 提交**
 * 顺序反了（先冻结、后让检查员跑），检查员新建的文件不在指纹里，
 * 提交那一刻指纹就变了 —— 症状是**每一次提交都被自己的闸门拦下**。
 */
export function fingerprint() {
  return execFileSync('git', ['write-tree'], {
    encoding: 'utf8',
    cwd: repoRoot()
  }).trim()
}

/** 两张通行证的定义。label 是给用户看的名字。 */
export const KINDS = {
  unit: { file: 'pass-unit.json', label: '单元测试 + 类型检查' },
  quality: { file: 'pass-quality.json', label: '质量评审' }
}

export function passportPath(kind) {
  return join(GATE_DIR, KINDS[kind].file)
}

/**
 * 读一张通行证。**不抛错** —— 读不到就返回 null，由调用方决定怎么说给用户听。
 * 文件损坏（半截 JSON）也返回 null：宁可当「没有证」，也不要带着一个坏对象往下走。
 */
export function readPassport(kind) {
  const file = passportPath(kind)
  if (!existsSync(file)) return null
  try {
    const parsed = JSON.parse(readFileSync(file, 'utf8'))
    if (parsed === null || typeof parsed !== 'object') return null
    return parsed
  } catch {
    return null
  }
}

/**
 * **两处闸门共用的判据** —— 只此一份。
 *
 * 闸门 A（git 钩子）和闸门 B（Claude 钩子）必须得出**完全一样**的结论。
 * 各写一份的话，两边只要有一处判据不一样（比如一边忘了看 `blocking`），
 * 症状就是「终端里提交被拦、Claude 里提交却放行」，而且谁都不会想到是两份代码。
 * 所以判据只写在这里，两边都调它。
 *
 * 返回 { fatal, problems }：
 *   fatal    —— 闸门自己坏了（算不出指纹之类），不是你代码的问题
 *   problems —— 为什么不让提交；空的就表示放行
 */
export function inspectPassports() {
  const problems = []

  let fp
  try {
    fp = fingerprint()
  } catch (e) {
    return {
      fatal: `算不出「即将提交的内容」的指纹，闸门没法工作。\n  原始错误：${String(e.message ?? e)}`,
      problems: []
    }
  }

  for (const kind of Object.keys(KINDS)) {
    const label = KINDS[kind].label
    const pass = readPassport(kind)

    if (pass === null) {
      problems.push(`缺少「${label}」通行证。`)
      continue
    }
    if (pass.fingerprint !== fp) {
      problems.push(
        `「${label}」通行证是**旧代码**的（代码改过之后没重新检查）——` +
          `它记的指纹是 ${String(pass.fingerprint).slice(0, 12)}…，当前是 ${fp.slice(0, 12)}…。`
      )
      continue
    }
    const blocking = Array.isArray(pass.blocking) ? pass.blocking : []
    if (blocking.length > 0) {
      problems.push(`「${label}」报了 ${blocking.length} 条**必须改**的问题：`)
      for (const item of blocking) problems.push(`    · ${String(item)}`)
    }
  }

  return { fatal: null, problems }
}

/**
 * 哪几张通行证**已经过期**（没有、或者指纹对不上当前冻结内容）。
 *
 * 为什么需要它：两个检查员是**并行**跑的，而 tester 会新建测试文件。
 * 如果 quality-engineer 在那些文件出现**之前**就发完了证，那张证记的指纹
 * 立刻就是旧的 —— 而它自己不会知道（它干完活的时候，世界还是对的）。
 *
 * 所以编排者在两人都回来之后再看一眼：谁过期了，谁重发一次。
 * 没人过期就什么都不做（不发证的那一方不会白跑一遍测试）。
 */
export function staleKinds() {
  let fp
  try {
    fp = fingerprint()
  } catch {
    // 连指纹都算不出来 —— 那就当两张都过期，让编排者去跑，由发证脚本自己报错
    return Object.keys(KINDS)
  }
  const out = []
  for (const kind of Object.keys(KINDS)) {
    const p = readPassport(kind)
    if (p === null || p.fingerprint !== fp) out.push(kind)
  }
  return out
}

/** 拒绝时那段「怎么办」的说明。两处闸门共用，措辞必须一致。 */
export const HOW_TO_FIX =
  '  这是怎么一回事：提交之前必须先过两位检查员（单元测试+类型检查、质量评审），\n' +
  '  跑完之后各发一张「通行证」；通行证绑着**当时那份代码**的指纹，\n' +
  '  所以改过代码就得重新跑一遍。\n' +
  '\n' +
  '  怎么办：在 Claude Code 里说一句 /gitcommit 。\n'
