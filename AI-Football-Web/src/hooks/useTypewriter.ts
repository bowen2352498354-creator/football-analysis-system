import { useEffect, useState } from 'react'

/**
 * 打字机效果 Hook：逐字显示传入文本，模拟 DeepSeek 流式吐字效果。
 * @param text 目标文本，变化时会重新开始打字
 * @param speed 每个字符的间隔时间（毫秒）
 */
export function useTypewriter(text: string, speed = 60) {
  const [displayText, setDisplayText] = useState('')
  const [isDone, setIsDone] = useState(false)

  useEffect(() => {
    setDisplayText('')
    setIsDone(false)

    let index = 0
    const timer = window.setInterval(() => {
      index += 1
      setDisplayText(text.slice(0, index))
      if (index >= text.length) {
        window.clearInterval(timer)
        setIsDone(true)
      }
    }, speed)

    return () => window.clearInterval(timer)
  }, [text, speed])

  return { displayText, isDone }
}
