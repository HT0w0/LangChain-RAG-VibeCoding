/**
 * 侧栏的一列可点行(会话列表、知识库列表)。
 *
 * **不用 antd 的 `List`。** v6 把它标成废弃、下个大版本移除,指定的替代品
 * `Listy` 却是个**数据驱动的虚拟化列表**(要 `items` / `rowKey` /
 * `itemRender`,没有 `List.Item.Meta`)——拿它渲染五条侧栏记录是语义错位,
 * 而且迟早要再改一次。这里要的本来只是「一列能点的行」,直接写出来
 * 比套一个正在被淘汰的组件更稳。
 */

import { Flex, Typography } from 'antd'
import type { ReactNode } from 'react'

export interface SidebarItem {
  key: number
  title: ReactNode
  subtitle?: ReactNode
  selected?: boolean
  onSelect?: () => void
  /** 右侧的操作区(通常是一个删除链接)。 */
  action?: ReactNode
}

export interface SidebarListProps {
  items: SidebarItem[]
  empty: string
}

export function SidebarList({ items, empty }: SidebarListProps) {
  if (items.length === 0) {
    return (
      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
        {empty}
      </Typography.Text>
    )
  }

  return (
    <Flex orientation="vertical" gap={2}>
      {items.map((item) => (
        <div
          key={item.key}
          className={`sidebar-row${item.selected ? ' sidebar-row-selected' : ''}`}
          onClick={item.onSelect}
        >
          <div style={{ flex: 1, minWidth: 0 }}>
            <Typography.Text strong={item.selected} ellipsis style={{ display: 'block' }}>
              {item.title}
            </Typography.Text>
            {item.subtitle && (
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                {item.subtitle}
              </Typography.Text>
            )}
          </div>
          {/* 操作区**不吃点击**。不拦住冒泡的话,点「删」会先选中那一项
              再弹确认框,而用户以为自己只是点了个删除。 */}
          {item.action && (
            <div onClick={(event) => event.stopPropagation()}>{item.action}</div>
          )}
        </div>
      ))}
    </Flex>
  )
}
