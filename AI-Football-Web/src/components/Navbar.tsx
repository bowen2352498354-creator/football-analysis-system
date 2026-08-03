import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent, MouseEvent as ReactMouseEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Download, Wifi, Activity, Settings2, ChevronDown, School as SchoolIcon, Plus, Check, HardDrive, X } from 'lucide-react'
import type { ApiStatus, GlobalSettings, ViewMode } from '../types'
import {
  getClassGroupDisplayName,
  getSchoolDisplayName,
  loadClassGroupOptions,
  loadSchoolOptions,
  removeClassGroupOption,
  removeSchoolOption,
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

  // 学校 / 班级完整可选项：预设（含已隐藏过滤）+ 自定义，首次挂载从 localStorage 恢复
  const [schoolList, setSchoolList] = useState<string[]>([])
  const [classList, setClassList] = useState<string[]>([])

  useEffect(() => {
    setSchoolList(loadSchoolOptions())
    setClassList(loadClassGroupOptions())
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

  /**
   * 删除学校/机构（预设与自定义均可）：拦截冒泡防止误选中，同步 localStorage，
   * 若删的是当前选中项则回退到剩余列表首项（或空串），避免幽灵选中态。
   */
  function handleDeleteSchool(schoolName: string, e: ReactMouseEvent) {
    e.stopPropagation()
    e.preventDefault()
    const nextList = removeSchoolOption(schoolName)
    setSchoolList(nextList)
    if (globalSettings.schoolName === schoolName) {
      onChangeGlobalSettings({ ...globalSettings, schoolName: nextList[0] ?? '' })
    }
  }

  /**
   * 删除班级/组别（预设与自定义均可）：拦截冒泡防止误选中，同步 localStorage，
   * 若删的是当前选中项则回退到剩余列表首项（或空串），避免幽灵选中态。
   */
  function handleDeleteClass(className: string, e: ReactMouseEvent) {
    e.stopPropagation()
    e.preventDefault()
    const nextList = removeClassGroupOption(className)
    setClassList(nextList)
    if (globalSettings.classGroupName === className) {
      onChangeGlobalSettings({ ...globalSettings, classGroupName: nextList[0] ?? '' })
    }
  }

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
                    options={schoolList}
                    value={globalSettings.schoolName}
                    placeholder="请输入学校 / 机构全称"
                    addButtonLabel="+ 新增自定义学校/机构"
                    onSelectValue={(val) => onChangeGlobalSettings({ ...globalSettings, schoolName: val })}
                    onAddCustomValue={(val) => {
                      setSchoolList(saveCustomSchoolName(val))
                    }}
                    onDeleteOption={handleDeleteSchool}
                  />

                  {/* 班级 / 实验组别：预设 + 100% 自定义录入并持久化保存 */}
                  <EnvOptionPicker
                    label="班级 / 组别"
                    options={classList}
                    value={globalSettings.classGroupName}
                    placeholder="请输入班级 / 分组名称，如「五年三班-实验A组」"
                    addButtonLabel="+ 新增自定义分组/班级"
                    onSelectValue={(val) => onChangeGlobalSettings({ ...globalSettings, classGroupName: val })}
                    onAddCustomValue={(val) => {
                      setClassList(saveCustomClassGroupName(val))
                    }}
                    onDeleteOption={handleDeleteClass}
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
  /** 当前完整可选项（预设 + 自定义，已过滤被删除项） */
  options: string[]
  /** 当前生效值（可能是列表中的值，或尚未同步进列表的最新自定义值） */
  value: string
  /** 新增自定义输入框的占位提示文案 */
  placeholder: string
  /** "新增自定义…" 触发按钮文案 */
  addButtonLabel: string
  onSelectValue: (value: string) => void
  onAddCustomValue: (value: string) => void
  /** 删除选项；调用方须在开头执行 e.stopPropagation / e.preventDefault */
  onDeleteOption: (name: string, e: ReactMouseEvent) => void
}

/**
 * 全局教学环境「学校 / 班级组别」选择器：
 * 每一项右侧常驻淡灰删除图标；点击删除不会触发选中（由调用方 stopPropagation）。
 */
function EnvOptionPicker({
  label,
  options,
  value,
  placeholder,
  addButtonLabel,
  onSelectValue,
  onAddCustomValue,
  onDeleteOption,
}: EnvOptionPickerProps) {
  const [isAdding, setIsAdding] = useState(false)
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const [draftValue, setDraftValue] = useState('')
  const pickerRef = useRef<HTMLDivElement>(null)

  const isValueKnown = options.includes(value)
  const displayLabel = value ? (isValueKnown ? value : `${value}（自定义）`) : '请选择或新增'

  // 点击选择器外部时收起下拉，避免与全局设置面板的外层点击逻辑冲突
  useEffect(() => {
    if (!isDropdownOpen) return
    function handleClickOutside(event: MouseEvent) {
      if (pickerRef.current && !pickerRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isDropdownOpen])

  function handleConfirmAdd() {
    const trimmed = draftValue.trim()
    if (!trimmed) return
    onAddCustomValue(trimmed)
    onSelectValue(trimmed)
    setDraftValue('')
    setIsAdding(false)
    setIsDropdownOpen(false)
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

  function handleSelectOption(option: string) {
    onSelectValue(option)
    setIsDropdownOpen(false)
  }

  return (
    <div className="mb-3 last:mb-0">
      <label className="mb-1.5 block text-xs font-medium text-white/50">{label}</label>
      <div ref={pickerRef} className="relative">
        <button
          type="button"
          onClick={() => setIsDropdownOpen((prev) => !prev)}
          className="flex w-full items-center justify-between rounded-2xl border border-white/10 bg-white/5 px-3.5 py-2.5 text-left text-sm text-white outline-none transition hover:bg-white/[0.08] focus:border-emerald-400/50"
        >
          <span className={`truncate ${value ? 'text-white' : 'text-white/40'}`}>{displayLabel}</span>
          <ChevronDown
            className={`h-3.5 w-3.5 flex-shrink-0 text-white/40 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`}
          />
        </button>

        <AnimatePresence>
          {isDropdownOpen && (
            <motion.ul
              initial={{ opacity: 0, y: -6, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6, scale: 0.98 }}
              transition={{ type: 'spring', stiffness: 420, damping: 32 }}
              className="absolute left-0 right-0 top-[calc(100%+6px)] z-20 max-h-48 overflow-y-auto rounded-2xl border border-white/10 bg-zinc-900/95 py-1 shadow-xl backdrop-blur-xl"
              role="listbox"
            >
              {/* 极端情况兜底：当前生效值尚未出现在列表中时，也先展示出来便于确认/删除 */}
              {!isValueKnown && value && (
                <li
                  role="option"
                  aria-selected
                  className="group flex cursor-pointer items-center justify-between gap-2 px-3 py-2 text-sm text-emerald-300 transition hover:bg-white/10"
                  onClick={() => handleSelectOption(value)}
                >
                  <span className="min-w-0 truncate">{value}（自定义）</span>
                  <button
                    type="button"
                    title="删除此项"
                    aria-label={`删除 ${value}`}
                    onClick={(e) => onDeleteOption(value, e)}
                    className="inline-flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md text-white/35 transition hover:bg-rose-500/20 hover:text-rose-300"
                  >
                    <X className="h-3.5 w-3.5" strokeWidth={2.25} />
                  </button>
                </li>
              )}
              {options.map((option) => {
                const isSelected = option === value
                return (
                  <li
                    key={option}
                    role="option"
                    aria-selected={isSelected}
                    className={`group flex cursor-pointer items-center justify-between gap-2 px-3 py-2 text-sm transition hover:bg-white/10 ${
                      isSelected ? 'text-emerald-300' : 'text-white/85'
                    }`}
                    onClick={() => handleSelectOption(option)}
                  >
                    <span className="min-w-0 truncate">{option}</span>
                    {/* 每一项右侧常驻淡灰 ✖；悬浮时略微高亮，不抢选中主交互 */}
                    <button
                      type="button"
                      title="删除此项"
                      aria-label={`删除 ${option}`}
                      onClick={(e) => onDeleteOption(option, e)}
                      className="inline-flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md text-white/35 transition hover:bg-rose-500/20 hover:text-rose-300"
                    >
                      <X className="h-3.5 w-3.5" strokeWidth={2.25} />
                    </button>
                  </li>
                )
              })}
              {options.length === 0 && !value && (
                <li className="px-3 py-2 text-xs text-white/35">暂无可选项，请先新增</li>
              )}
            </motion.ul>
          )}
        </AnimatePresence>
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
