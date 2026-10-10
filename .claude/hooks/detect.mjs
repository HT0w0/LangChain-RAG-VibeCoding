/**
 * 「这条命令是不是一句 git commit？」—— 单独拆出来，是为了能被客观检查。
 *
 * 为什么值得单独一个文件：这段逻辑出错的后果是**静默的**。
 *   · 误判（把无关命令当成提交）→ 用户正常干活被拦，看得见，还好。
 *   · **漏判（真正的提交没被认出来）→ 闸门被绕过，一声不吭。**
 * 后者才是要命的，而它恰恰是「看起来一切正常」的那种错。
 * 所以这里不能凭手感，要有 `.claude/gate/check-detect.mjs` 一组用例钉着。
 */

/** 把一条命令行按 shell 的分隔符切成若干段。**必须认引号**—— */
/** 否则 `git commit -m "修好了; 顺便删掉日志"` 里的分号会被当成命令分隔符。 */
export function splitSegments(cmd) {
  const segs = []
  let cur = ''
  let quote = null
  for (let i = 0; i < cmd.length; i++) {
    const ch = cmd[i]
    if (quote !== null) {
      if (ch === '\\' && quote === '"') {
        cur += ch + (cmd[++i] ?? '')
        continue
      }
      if (ch === quote) quote = null
      cur += ch
      continue
    }
    if (ch === '"' || ch === "'") {
      quote = ch
      cur += ch
      continue
    }
    if (ch === '\\') {
      cur += ch + (cmd[++i] ?? '')
      continue
    }
    const two = ch + (cmd[i + 1] ?? '')
    if (two === '&&' || two === '||') {
      segs.push(cur)
      cur = ''
      i++
      continue
    }
    if (ch === ';' || ch === '|' || ch === '\n') {
      segs.push(cur)
      cur = ''
      continue
    }
    cur += ch
  }
  segs.push(cur)
  return segs.filter((s) => s.trim() !== '')
}

/** 把一段切成词，去掉引号。 */
export function tokenize(seg) {
  const out = []
  let cur = ''
  let quote = null
  let started = false
  for (let i = 0; i < seg.length; i++) {
    const ch = seg[i]
    if (quote !== null) {
      if (ch === '\\' && quote === '"') {
        cur += seg[++i] ?? ''
        continue
      }
      if (ch === quote) {
        quote = null
        continue
      }
      cur += ch
      continue
    }
    if (ch === '"' || ch === "'") {
      quote = ch
      started = true
      continue
    }
    if (ch === '\\') {
      cur += seg[++i] ?? ''
      started = true
      continue
    }
    if (ch === ' ' || ch === '\t') {
      if (started) {
        out.push(cur)
        cur = ''
        started = false
      }
      continue
    }
    cur += ch
    started = true
  }
  if (started) out.push(cur)
  return out
}

/** git 的全局选项里，这些会「吃掉」后面一个词 */
const GIT_OPTS_TAKING_VALUE = new Set([
  '-C',
  '-c',
  '--git-dir',
  '--work-tree',
  '--namespace',
  '--exec-path',
  '--config-env'
])

/**
 * 这段是不是一句 `git commit`？是就返回 commit 之后的参数，不是就返回 null。
 *
 * 判据刻意收得很紧：**要求第一个非选项词就是 `commit`**，
 * 而不是「命令里出现过 commit 这个词」。松一点的话，`git log --grep commit`
 * 也会被当成提交拦下来 —— 误判会让用户觉得软件坏了。
 */
export function asGitCommit(seg) {
  const t = tokenize(seg)
  if (t.length === 0) return null

  const bin = t[0].replace(/\\/g, '/').split('/').pop().toLowerCase()
  if (bin !== 'git' && bin !== 'git.exe') return null

  let i = 1
  while (i < t.length && t[i].startsWith('-')) {
    i += GIT_OPTS_TAKING_VALUE.has(t[i]) ? 2 : 1
  }
  if (t[i] !== 'commit') return null
  return t.slice(i)
}

/**
 * 一条完整命令里，有没有哪一段是 git commit？
 * 返回那一段的参数（后面的段优先，因为 `git add -A && git commit` 里
 * 真正要提交的是最后那句），没有就返回 null。
 */
export function detectGitCommit(command) {
  let found = null
  for (const seg of splitSegments(command)) {
    const args = asGitCommit(seg)
    if (args !== null) found = args
  }
  return found
}
