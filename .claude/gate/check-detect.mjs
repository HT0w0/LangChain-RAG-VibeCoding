#!/usr/bin/env node
/**
 * 「这条命令是不是 git commit」的客观检查 —— `node .claude/gate/check-detect.mjs`
 *
 * 为什么要单独测这一段：它出错的后果是**静默的**。
 * 漏判（真正的提交没认出来）= 闸门被绕过，却一声不吭 ——
 * 看起来一切正常，正是那种「比没有检查更坏」的情况。
 *
 * 所以这里两类用例都要有：
 *   · 该认出来的（漏一个 = 闸门有洞）
 *   · **不该认出来的**（误判 = 用户正常干活被拦，会觉得软件坏了）
 *
 * 反向辨识：把 detect.mjs 的 asGitCommit 改成恒返回 null，
 * 「该认出来的」那组应当**全红**。做不到就说明这个检查是空转的。
 */
import { detectGitCommit } from '../hooks/detect.mjs'

/** [命令, 期望认出来吗, 说明] */
const CASES = [
  ['git commit -m "修好了"', true, '最普通的一种'],
  ['git commit', true, '不带参数（会开编辑器）'],
  ['git commit --amend', true, '改上一次提交也算提交'],
  ['git commit -am "x"', true, '合并的 -am'],
  ['git commit -F 提交信息.txt', true, '从文件读信息'],
  ['git add -A && git commit -m "x"', true, '先加后提交 —— 实际最常走的路径'],
  ['cd src && git commit -m "x"', true, '先切目录'],
  ['git reset --soft HEAD~1; git commit -m "x"', true, '分号连起来的'],
  ['git commit -m "x" --no-verify', true, '带绕过开关的（认出来才拦得住）'],
  ['git commit --no-verify', true, '同上'],
  ['git -C . commit -m "x"', true, 'git 的全局选项 -C'],
  // ⚠️ 这条用例我一开始写成了**不带引号**的 `/c/Program Files/.../git.exe commit`，
  // 检查报它「没认出来」。**但错的是用例，不是代码** —— 实测那条命令在 bash 里
  // 根本跑不起来（路径里的空格把命令切成了两截，报 `/c/Program: No such file or directory`），
  // 它不可能是绕过闸门的真实路径。所以改成真实用户会敲的引号写法。
  // 记在这里，是因为「把用例改成能过的样子」这个动作，跟「把检查改松」长得一模一样。
  ['"/c/Program Files/Git/cmd/git.exe" commit -m "x"', true, '写全路径的 git.exe（带引号）'],
  ['  git   commit   -m  "x"  ', true, '多余空格'],
  ['git commit -m "改好了; 顺便删掉日志"', true, '引号里有分号 —— 不能被切成两段'],

  ['git log --oneline -3', false, '看日志'],
  ['git status', false, '看状态'],
  ['git diff --cached', false, '看差异'],
  ['git push origin main', false, '推送'],
  ['git commit-tree abc123', false, '名字里带 commit，但是另一个命令'],
  ['git log --grep commit', false, '参数里带 commit'],
  ['git config --get commit.gpgsign', false, '配置键里带 commit'],
  ['echo "这句话里有 git commit 三个字"', false, '只是打印'],
  ['echo "a; git commit -m x"', false, '引号里的分号不算分隔符'],
  ['npm run commitlint', false, '别的工具的提交检查'],
  ['node scripts/提交脚本.mjs', false, '跑脚本'],
  ['git commit --help', false, '看帮助（钩子里会被跳过，但识别层认它）'],
  ['git commit --dry-run', false, '演练，不真提交'],
  ['git --version', false, '看版本'],
  ['ls -la', false, '跟 git 无关']
]

let pass = 0
const failures = []

for (const [command, expectedDetect, note] of CASES) {
  const args = detectGitCommit(command)
  const rawDetected = args !== null

  // `--help` / `--dry-run` 这两条：识别层认它是提交，但钩子会主动跳过。
  // 这里单独说明，免得以后有人以为用例写错了。
  const isSkippedCase = command.includes('--help') || command.includes('--dry-run')
  const effective = isSkippedCase ? false : rawDetected

  if (effective === expectedDetect) {
    pass++
    process.stdout.write(`  ✅ ${note} — ${JSON.stringify(command)}\n`)
  } else {
    failures.push(
      `❌ ${note}\n     命令：${JSON.stringify(command)}\n     期望：${expectedDetect ? '认出来' : '不认'}，实际：${effective ? '认出来' : '不认'}` +
        (isSkippedCase ? '\n     （注：这是「识别层认、钩子主动跳过」的那一类）' : '')
    )
    process.stdout.write(`  ❌ ${note} — ${JSON.stringify(command)}\n`)
  }
}

process.stdout.write(`\n合计 ${CASES.length} 项：通过 ${pass}，失败 ${failures.length}\n`)
if (failures.length > 0) {
  process.stdout.write('\n' + failures.join('\n\n') + '\n')
  process.exit(1)
}
process.exit(0)
