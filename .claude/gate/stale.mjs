#!/usr/bin/env node
/**
 * 打印「哪几张通行证过期了」—— 一行一个，没过期就什么都不打印。
 *
 *   node .claude/gate/stale.mjs
 *   → unit
 *   → quality
 *
 * 给 `/gitcommit` 用：两个检查员并行跑完之后，谁过期了就补发谁。
 *
 * 为什么需要这一步：tester 会新建测试文件，而 quality-engineer 可能在那些文件
 * 出现**之前**就发完了证 —— 它干完活的时候世界是对的，所以它自己发现不了。
 * 只有等两个人都停下来、回头再看一眼，才知道谁的那张是旧的了。
 */
import { staleKinds } from './lib.mjs'

for (const kind of staleKinds()) {
  process.stdout.write(kind + '\n')
}
