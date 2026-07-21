import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Navbar from './components/Navbar'
import RealtimeWorkspace from './components/RealtimeWorkspace'
import ZenWorkspace from './components/ZenWorkspace'
import CoachDashboard from './components/CoachDashboard'
import { DEFAULT_GLOBAL_SETTINGS } from './mockData'
import type { ApiStatus, GlobalSettings, ViewMode } from './types'

/** 主应用架构：深色沉浸主题 + 三视图切换 */
function App() {
  const [activeView, setActiveView] = useState<ViewMode>('realtime')
  const [apiStatus] = useState<ApiStatus>('online')
  // 全局教学环境设置（学校 + 班级/组别）：在 App 顶层统一管理，供 Navbar 编辑、各工作台只读消费
  const [globalSettings, setGlobalSettings] = useState<GlobalSettings>(DEFAULT_GLOBAL_SETTINGS)

  const handleDownloadTestData = () => {
    const payload = {
      generatedAt: new Date().toISOString(),
      note: '本文件为模拟测试数据导出，正式版将对接本地边缘计算引擎生成的匿名化结构文本。',
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'test-data.json'
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="app-shell text-slate-100">
      <Navbar
        activeView={activeView}
        onChangeView={setActiveView}
        apiStatus={apiStatus}
        onDownloadTestData={handleDownloadTestData}
        globalSettings={globalSettings}
        onChangeGlobalSettings={setGlobalSettings}
      />

      <AnimatePresence mode="wait">
        <motion.main
          key={activeView}
          className="workbench-shell min-h-0 flex-1"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
        >
          {activeView === 'realtime' && <RealtimeWorkspace globalSettings={globalSettings} />}
          {activeView === 'zen' && <ZenWorkspace globalSettings={globalSettings} />}
          {activeView === 'coach' && <CoachDashboard />}
        </motion.main>
      </AnimatePresence>
    </div>
  )
}

export default App
