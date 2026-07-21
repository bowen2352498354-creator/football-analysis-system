import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Download, Wifi, Activity, Settings2, ChevronDown, School as SchoolIcon, Plus, Check, HardDrive } from 'lucide-react'
import type { ApiStatus, GlobalSettings, ViewMode } from '../types'
import {
  getClassGroupDisplayName,
  getSchoolDisplayName,
  loadCustomClassGroupNames,
  loadCustomSchoolNames,
  PRESET_CLASS_GROUP_NAMES,
  PRESET_SCHOOL_NAMES,
  saveCustomClassGroupName,
  saveCustomSchoolName,
} from '../mockData'

interface NavTab {
  id: ViewMode
  label: string
}

const NAV_TABS: NavTab[] = [
  { id: 'realtime', label: '实时反馈系统 (实验A组)' },
  { id: 'zen', label: '延时反馈系统 (实验B组)' },
  { id: 'coach', label: '教练端数据看板' },
]

interface NavbarProps {
  activeView: ViewMode
  onChangeView: (view: ViewMode) => void
  apiStatus: ApiStatus
  onDownloadTestData: () => void
  /** 全局教学环境设置（学校 + 班级/组别），贯穿全站各工作台 */
  globalSettings: GlobalSettings
  onChangeGlobalSettings: (settings: GlobalSettings) => void
}

/** API 状态指示灯颜色映射 */
const API_STATUS_STYLE: Record<ApiStatus, { color: string; label: string }> = {
  online: { color: 'bg-emerald-400', label: 'API 在线' },
  connecting: { color: 'bg-amber-400', label: 'API 连接中' },
  offline: { color: 'bg-rose-500', label: 'API 离线' },
}

/** 全局顶部导航栏：毛玻璃拟态风格，居中三段式切换 + 右侧状态区 + 全局教学环境设置下拉 */
export default function Navbar({
  activeView,
  onChangeView,
  apiStatus,
  onDownloadTestData,
  globalSettings,
  onChangeGlobalSettings,
}: NavbarProps) {
  const [isOnline] = useState(true)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const settingsRef = useRef<HTMLDivElement>(null)
  const statusStyle = API_STATUS_STYLE[apiStatus]

  // 自定义学校 / 班级分组列表：首次挂载时从 localStorage 读取教师此前保存过的记录，
  // 新增时同步写回 localStorage，实现"一次录入，长期复用"的持久化体验。
  const [customSchoolNames, setCustomSchoolNames] = useState<string[]>([])
  const [customClassGroupNames, setCustomClassGroupNames] = useState<string[]>([])

  useEffect(() => {
    setCustomSchoolNames(loadCustomSchoolNames())
    setCustomClassGroupNames(loadCustomClassGroupNames())
  }, [])

  // 点击面板外部区域时自动收起下拉设置面板
  useEffect(() => {
    if (!isSettingsOpen) return
    function handleClickOutside(event: MouseEvent) {
      if (settingsRef.current && !settingsRef.current.contains(event.target as Node)) {
        setIsSettingsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isSettingsOpen])

  const summaryText = `${getSchoolDisplayName(globalSettings)} · ${getClassGroupDisplayName(globalSettings)}`

  return (
    <header className="sticky top-0 z-50 w-full flex-shrink-0 border-b border-slate-700/80 bg-slate-900/90 backdrop-blur-md">
      <div className="flex h-14 w-full items-center justify-between gap-4 px-3 sm:px-4 lg:px-5">
        {/* 左侧品牌标识 */}
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400 to-sky-500 text-sm font-bold text-black">
            ⚽
          </div>
          <span className="hidden text-sm font-semibold tracking-wide text-white/90 sm:inline">
            足球AI可视化反馈系统
          </span>
        </div>

        {/* 中间导航选项卡 */}
        <nav className="relative flex items-center gap-1 rounded-full bg-black/20 p-1">
          {NAV_TABS.map((tab) => {
            const isActive = tab.id === activeView
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => onChangeView(tab.id)}
                className={`relative rounded-full px-3 py-1.5 text-xs font-medium transition-colors sm:px-4 sm:text-sm ${
                  isActive ? 'text-white' : 'text-white/50 hover:text-white/80'
                }`}
              >
                {isActive && (
                  <motion.span
                    layoutId="nav-active-pill"
                    className="absolute inset-0 rounded-full bg-white/15 shadow-inner"
                    transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                  />
                )}
                <span className="relative z-10 whitespace-nowrap">{tab.label}</span>
                {isActive && (
                  <motion.span
                    layoutId="nav-active-underline"
                    className="absolute -bottom-1 left-1/2 h-0.5 w-6 -translate-x-1/2 rounded-full bg-emerald-400"
                    transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                  />
                )}
              </button>
            )
          })}
        </nav>

        {/* 右侧状态区 */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* 全局教学环境设置下拉入口 */}
          <div ref={settingsRef} className="relative">
            <button
              type="button"
              onClick={() => setIsSettingsOpen((prev) => !prev)}
              className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/10 px-2.5 py-1.5 text-xs font-medium text-white/80 transition hover:bg-white/20 active:scale-95 sm:px-3 sm:text-sm"
              title="全局教学环境设置"
            >
              <span className="inline-flex flex-shrink-0">
                <Settings2 className="h-3.5 w-3.5 text-emerald-400" />
              </span>
              <span className="hidden max-w-[10rem] truncate lg:inline">{summaryText}</span>
              <span className="inline-flex flex-shrink-0">
                <ChevronDown className={`h-3.5 w-3.5 transition-transform ${isSettingsOpen ? 'rotate-180' : ''}`} />
              </span>
            </button>

            <AnimatePresence>
              {isSettingsOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -8, scale: 0.96 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -8, scale: 0.96 }}
                  transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                  className="absolute right-0 top-[calc(100%+10px)] w-80 rounded-3xl border border-white/10 bg-black/80 p-4 shadow-2xl backdrop-blur-2xl"
                >
                  <div className="mb-3 flex items-center gap-2">
                    <span className="inline-flex flex-shrink-0">
                      <SchoolIcon className="h-4 w-4 text-emerald-400" />
                    </span>
                    <p className="text-sm font-semibold text-white/90">全局教学环境设置</p>
                  </div>
                  <p className="mb-4 text-xs leading-relaxed text-white/40">
                    一次设置，全站生效。支持 100% 自定义录入并自动持久化保存，以下选择将自动同步至
                    「实时反馈」「延时反馈」与「教练端看板」各工作台。
                  </p>

                  {/* 学校 / 机构：预设 + 100% 自定义录入并持久化保存 */}
                  <EnvOptionPicker
                    label="学校 / 机构"
                    presetOptions={PRESET_SCHOOL_NAMES}
                    customOptions={customSchoolNames}
                    value={globalSettings.schoolName}
                    placeholder="请输入学校 / 机构全称"
                    addButtonLabel="+ 新增自定义学校/机构"
                    onSelectValue={(val) => onChangeGlobalSettings({ ...globalSettings, schoolName: val })}
                    onAddCustomValue={(val) => {
                      const nextList = saveCustomSchoolName(val)
                      setCustomSchoolNames(nextList)
                    }}
                  />

                  {/* 班级 / 实验组别：预设 + 100% 自定义录入并持久化保存 */}
                  <EnvOptionPicker
                    label="班级 / 组别"
                    presetOptions={PRESET_CLASS_GROUP_NAMES}
                    customOptions={customClassGroupNames}
                    value={globalSettings.classGroupName}
                    placeholder="请输入班级 / 分组名称，如「五年三班-实验A组」"
                    addButtonLabel="+ 新增自定义分组/班级"
                    onSelectValue={(val) => onChangeGlobalSettings({ ...globalSettings, classGroupName: val })}
                    onAddCustomValue={(val) => {
                      const nextList = saveCustomClassGroupName(val)
                      setCustomClassGroupNames(nextList)
                    }}
                  />

                  {/* 【核心新增】全局归档总闸：极其显眼的 Apple 风格切换开关，
                      控制本次训练数据是否自动本地落盘归档 + 同步至教练看板 */}
                  <div className="mt-4 border-t border-white/10 pt-4">
                    <div
                      className={`overflow-hidden rounded-2xl ring-1 transition-colors ${
                        globalSettings.enableDataArchiving
                          ? 'bg-emerald-500/10 ring-emerald-400/30'
                          : 'bg-white/5 ring-white/10'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3 px-4 py-3">
                        <div className="flex min-w-0 items-center gap-2.5">
                          <span
                            className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl ${
                              globalSettings.enableDataArchiving ? 'bg-emerald-500/20' : 'bg-white/10'
                            }`}
                          >
                            <HardDrive
                              className={`h-4 w-4 ${
                                globalSettings.enableDataArchiving ? 'text-emerald-300' : 'text-white/40'
                              }`}
                            />
                          </span>
                          <p className="truncate text-sm font-semibold text-white/90">
                            💾 本次训练数据本地落盘归档
                          </p>
                        </div>
                        <button
                          type="button"
                          role="switch"
                          aria-checked={globalSettings.enableDataArchiving}
                          onClick={() =>
                            onChangeGlobalSettings({
                              ...globalSettings,
                              enableDataArchiving: !globalSettings.enableDataArchiving,
                            })
                          }
                          className={`relative flex h-7 w-[4.5rem] flex-shrink-0 items-center rounded-full px-1 text-[10px] font-bold transition-colors duration-300 ${
                            globalSettings.enableDataArchiving ? 'bg-emerald-500 justify-start' : 'bg-white/15 justify-end'
                          }`}
                        >
                          <motion.span
                            layout
                            transition={{ type: 'spring', stiffness: 500, damping: 32 }}
                            className={`absolute top-0.5 h-6 w-6 rounded-full bg-white shadow ${
                              globalSettings.enableDataArchiving ? 'left-[calc(100%-1.625rem)]' : 'left-0.5'
                            }`}
                          />
                          <span
                            className={`z-10 ${globalSettings.enableDataArchiving ? 'ml-1 text-black' : 'mr-1 text-white/70'}`}
                          >
                            {globalSettings.enableDataArchiving ? '🟢 开启' : '⚪ 关闭'}
                          </span>
                        </button>
                      </div>
                      <p className="border-t border-white/10 px-4 py-2.5 text-[11px] leading-relaxed text-white/50">
                        开启后，实时与延时组的所有测试结果将自动生成 Word 并同步至教练看板。
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* 网络连接状态 */}
          <div className="hidden items-center gap-1.5 text-white/60 md:flex" title="网络连接状态">
            <span className="inline-flex flex-shrink-0">
              <Wifi className={`h-4 w-4 ${isOnline ? 'text-emerald-400' : 'text-rose-500'}`} />
            </span>
          </div>

          {/* API 状态指示灯 */}
          <div className="hidden items-center gap-1.5 rounded-full bg-black/20 px-2.5 py-1 md:flex" title={statusStyle.label}>
            <span className={`h-2 w-2 rounded-full ${statusStyle.color} animate-pulse`} />
            <span className="flex items-center gap-1 text-xs text-white/60">
              <span className="inline-flex flex-shrink-0">
                <Activity className="h-3 w-3" />
              </span>
              {statusStyle.label}
            </span>
          </div>

          {/* 下载测试数据按钮 */}
          <button
            type="button"
            onClick={onDownloadTestData}
            className="flex items-center gap-1.5 rounded-full bg-white/10 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-white/20 active:scale-95 sm:text-sm"
          >
            <span className="inline-flex flex-shrink-0">
              <Download className="h-3.5 w-3.5" />
            </span>
            <span className="hidden sm:inline">下载测试数据</span>
          </button>
        </div>
      </div>
    </header>
  )
}

interface EnvOptionPickerProps {
  /** 字段标签，例如「学校 / 机构」 */
  label: string
  /** 内置常用预设选项 */
  presetOptions: string[]
  /** 教师此前保存过的自定义选项（来自 localStorage） */
  customOptions: string[]
  /** 当前生效值（可能是预设值、历史自定义值，或尚未同步进列表的最新自定义值） */
  value: string
  /** 新增自定义输入框的占位提示文案 */
  placeholder: string
  /** "新增自定义…" 触发按钮文案 */
  addButtonLabel: string
  onSelectValue: (value: string) => void
  onAddCustomValue: (value: string) => void
}

/**
 * 全局教学环境「学校 / 班级组别」100% 自定义选择器：
 * 下拉框汇总「内置常用预设 + 教师历史自定义记录」供快速复用，
 * 同时保留一个随时可展开的文本输入框，允许教师输入任意名称并一键持久化保存，
 * 保存后立即成为当前生效值，且会永久出现在下拉列表中供下次直接选用。
 */
function EnvOptionPicker({
  label,
  presetOptions,
  customOptions,
  value,
  placeholder,
  addButtonLabel,
  onSelectValue,
  onAddCustomValue,
}: EnvOptionPickerProps) {
  const [isAdding, setIsAdding] = useState(false)
  const [draftValue, setDraftValue] = useState('')

  // 合并预设与自定义选项，去重后统一展示在下拉列表里
  const mergedOptions = [...presetOptions, ...customOptions.filter((item) => !presetOptions.includes(item))]
  const isValueKnown = mergedOptions.includes(value)

  function handleConfirmAdd() {
    const trimmed = draftValue.trim()
    if (!trimmed) return
    onAddCustomValue(trimmed)
    onSelectValue(trimmed)
    setDraftValue('')
    setIsAdding(false)
  }

  function handleInputKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Enter') {
      event.preventDefault()
      handleConfirmAdd()
    } else if (event.key === 'Escape') {
      setIsAdding(false)
      setDraftValue('')
    }
  }

  return (
    <div className="mb-3 last:mb-0">
      <label className="mb-1.5 block text-xs font-medium text-white/50">{label}</label>
      <div className="relative">
        <select
          value={isValueKnown ? value : '__unsynced_custom__'}
          onChange={(e) => onSelectValue(e.target.value)}
          className="w-full appearance-none rounded-2xl border border-white/10 bg-white/5 px-3.5 py-2.5 text-sm text-white outline-none transition focus:border-emerald-400/50 [&>option]:bg-zinc-900"
        >
          {/* 极端情况兜底：当前生效值尚未出现在合并列表中（例如数据刚迁移），也要能正常显示 */}
          {!isValueKnown && (
            <option value="__unsynced_custom__">{value ? `${value}（自定义）` : '请选择或新增'}</option>
          )}
          {mergedOptions.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        <span className="pointer-events-none absolute right-3.5 top-1/2 inline-flex -translate-y-1/2">
          <ChevronDown className="h-3.5 w-3.5 text-white/40" />
        </span>
      </div>

      <AnimatePresence initial={false} mode="wait">
        {!isAdding ? (
          <motion.button
            key="add-trigger"
            type="button"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsAdding(true)}
            className="mt-2 flex items-center gap-1.5 text-xs font-medium text-emerald-300/80 transition hover:text-emerald-300"
          >
            <span className="inline-flex flex-shrink-0">
              <Plus className="h-3.5 w-3.5" />
            </span>
            {addButtonLabel}
          </motion.button>
        ) : (
          <motion.div
            key="add-input"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2 flex items-center gap-2 overflow-hidden"
          >
            <input
              autoFocus
              type="text"
              value={draftValue}
              onChange={(e) => setDraftValue(e.target.value)}
              onKeyDown={handleInputKeyDown}
              placeholder={placeholder}
              className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-3.5 py-2 text-sm text-white placeholder:text-white/30 outline-none transition focus:border-emerald-400/50"
            />
            <button
              type="button"
              onClick={handleConfirmAdd}
              title="保存并使用"
              className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-emerald-500 text-black transition hover:bg-emerald-400 active:scale-95"
            >
              <span className="inline-flex flex-shrink-0">
                <Check className="h-4 w-4" />
              </span>
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
