/**
 * V2.5 Traffic-Light System：统一视觉降维色彩 token。
 * 与 index.css 中的 CSS 变量保持一一对应，供 TS / 内联 style / 图表配置复用。
 */

export const TRAFFIC_LIGHT = {
  GREEN_OPTIMAL: '#10B981',
  YELLOW_APPROACHING: '#F59E0B',
  RED_DEVIATED: '#EF4444',
} as const

export type TrafficLightKey = keyof typeof TRAFFIC_LIGHT

export type TrafficLightLevel = 'green' | 'yellow' | 'red' | 'pending'

/** 状态 -> 十六进制色 */
export const TRAFFIC_HEX: Record<Exclude<TrafficLightLevel, 'pending'>, string> = {
  green: TRAFFIC_LIGHT.GREEN_OPTIMAL,
  yellow: TRAFFIC_LIGHT.YELLOW_APPROACHING,
  red: TRAFFIC_LIGHT.RED_DEVIATED,
}

/** 状态 -> Tailwind / 工具类（点状指示灯、文字、卡片发光） */
export const TRAFFIC_CLASS: Record<
  TrafficLightLevel,
  { dot: string; text: string; glow: string; border: string; bg: string; label: string }
> = {
  green: {
    dot: 'bg-[var(--GREEN_OPTIMAL)]',
    text: 'text-[var(--GREEN_OPTIMAL)]',
    glow: 'traffic-glow-green',
    border: 'border-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_35%,transparent)]',
    bg: 'bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_12%,transparent)]',
    label: '达标',
  },
  yellow: {
    dot: 'bg-[var(--YELLOW_APPROACHING)]',
    text: 'text-[var(--YELLOW_APPROACHING)]',
    glow: 'traffic-glow-yellow',
    border: 'border-[color-mix(in_srgb,var(--YELLOW_APPROACHING)_35%,transparent)]',
    bg: 'bg-[color-mix(in_srgb,var(--YELLOW_APPROACHING)_12%,transparent)]',
    label: '接近',
  },
  red: {
    dot: 'bg-[var(--RED_DEVIATED)]',
    text: 'text-[var(--RED_DEVIATED)]',
    glow: 'traffic-glow-red',
    border: 'border-[color-mix(in_srgb,var(--RED_DEVIATED)_35%,transparent)]',
    bg: 'bg-[color-mix(in_srgb,var(--RED_DEVIATED)_12%,transparent)]',
    label: '偏离',
  },
  pending: {
    dot: 'bg-slate-500',
    text: 'text-slate-400',
    glow: '',
    border: 'border-slate-700',
    bg: 'bg-slate-800/60',
    label: '待机',
  },
}

export function levelToTraffic(level: 'green' | 'yellow' | 'red' | null | undefined): TrafficLightLevel {
  if (level === 'green' || level === 'yellow' || level === 'red') return level
  return 'pending'
}
