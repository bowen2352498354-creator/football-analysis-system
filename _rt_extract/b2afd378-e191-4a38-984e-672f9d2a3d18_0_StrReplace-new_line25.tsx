    if (message.type === 'frame') {
      // 【容错防呆】只有拿到规范的 "data:image/..." 格式字符串才更新画面；
      // 万一某一条消息的 image 字段异常（空字符串/undefined/格式不对），
      // 直接忽略这一帧、保留上一帧画面，绝不能让 <img src="" /> 这种空/非法
      // src 把画面"闪"成黑屏——这正是之前排查"点击开始分析后黑屏"问题的关键点。
      // 倒带复盘模式下不覆盖中栏画面，避免 live 帧冲掉历史切片预览。
      if (
        reviewSnapshot == null &&
        typeof message.image === 'string' &&
        message.image.startsWith('data:image')
      ) {
        setFrameImage(message.image)
      }

      if (message.angle !== null && message.status !== null) {
        setKneeAngle(message.angle)
        setBackendStatus(message.status)
        const nextLevel = statusToLevel(message.status)
        if (nextLevel) {
          setHitStats((stats) => ({ ...stats, [nextLevel]: stats[nextLevel] + 1 }))
        }
      }

      // 实时动力链角速度监控：把新样本追加进滚动窗口，并淘汰掉 5 秒之前的旧样本。
      // 【容错防呆】额外用 Number.isFinite 校验，避免后端在极端情况下（例如第一帧
      // 尚未计算出角速度、或除法结果出现 NaN/Infinity）推来非法数值，导致后续
      // SVG 波形图渲染出无法解析的坐标，静默影响整张卡片渲染。
      if (typeof message.angular_velocity === 'number' && Number.isFinite(message.angular_velocity)) {
        const now = Date.now()
        const nextVelocity = message.angular_velocity
        setVelocityHistory((prev) => {
          const next = [...prev, { t: now, v: nextVelocity }]
          const cutoff = now - VELOCITY_WINDOW_MS
          return next.filter((sample) => sample.t >= cutoff)
        })
      }
      if (typeof message.stability_index === 'number' && Number.isFinite(message.stability_index)) {
        setStabilityIndex(message.stability_index)
      }

      // 【Pro-Studio 播控台】仅本地视频分析模式下后端才会回填这三个字段，
      // 摄像头直播模式下始终是 undefined/null，前端组件已经对 null 做了兜底展示。
      if (typeof message.position_ms === 'number') setVideoPositionMs(message.position_ms)
      if (typeof message.duration_ms === 'number') setVideoDurationMs(message.duration_ms)
      if (typeof message.is_paused === 'boolean') setIsPaused(message.is_paused)

      // 【V3·模块三】FSM 广播：更新 HUD 脉冲灯，并在 RECORDING/PROCESSING 时挂载链占位卡
      if (message.fsm_state === 'IDLE' || message.fsm_state === 'RECORDING' || message.fsm_state === 'PROCESSING') {
        setFsmState(message.fsm_state)
        syncLiveAttemptPlaceholder(message.fsm_state, message.attempt_count ?? 0)
      }
      if (typeof message.attempt_count === 'number') setAutoAttemptCount(message.attempt_count)
      if (message.latest_score === null || typeof message.latest_score === 'number') {
        setLatestAutoScore(message.latest_score ?? null)
      }
      return
    }

    if (message.type === 'attempt_captured') {
      setFsmState(message.fsm_state || 'IDLE')
      setAutoAttemptCount(message.attempt_count)
      setLatestAutoScore(message.latest_score)
      setForceCaptureBusy(false)
      const videoUrl = message.video_url
        ? message.video_url.startsWith('http')
          ? message.video_url
          : `${API_BASE_URL}${message.video_url}`
        : null
      const readyAttempt: AutoCaptureAttempt = {
        id: `attempt-${message.attempt_number}-${Date.now()}`,
        attemptNumber: message.attempt_number,
        score: message.score,
        status: 'ready',
        videoUrl,
        impactFrameImage: message.impactFrameImage ?? null,
        errorCodes: message.errorCodes ?? [],
        biomechanicsMetrics: message.biomechanicsMetrics ?? null,
        spatialTrajectoryData: message.spatialTrajectoryData ?? null,
        quantified5dScores: message.quantified5dScores ?? null,
        radar5d: message.radar5d ?? null,
        manual: Boolean(message.manual),
      }
      setAutoAttempts((prev) => {
        const withoutPlaceholder = prev.filter(
          (a) =>
            !(
              (a.status === 'capturing' || a.status === 'processing') &&
              a.attemptNumber === message.attempt_number
            ) && a.attemptNumber !== message.attempt_number,
        )
        return [...withoutPlaceholder, readyAttempt].sort((a, b) => a.attemptNumber - b.attemptNumber)
      })
      // 自动倒带至刚捕获的这一脚，右侧雷达同步刷新
      applyAttemptReview(readyAttempt)
      const landing = message.spatialTrajectoryData?.support_foot_landing_pos
      if (Array.isArray(landing) && landing.length >= 2) {
        setSessionLandings((prev) => [...prev, [landing[0], landing[1]]])
      }
      return
    }

    if (message.type === 'force_capture_ack') {
      setForceCaptureBusy(false)
      if (message.fsm_state) setFsmState(message.fsm_state)
      if (typeof message.attempt_count === 'number') setAutoAttemptCount(message.attempt_count)
      if (!message.success) {
        setDiagnosticNotice(message.message || '强制截取失败')
      }
      return
    }

    if (message.type === 'started') {
      setSessionId(message.session_id)
      return
    }

    if (message.type === 'stopped') {
      wsRef.current?.close()
      wsRef.current = null
      setIsConnected(false)
      void fetchGeneratedReport(message.session_id)
      return
    }

    if (message.type === 'error') {
      setConnectionError(message.message)
      setAnalysisStatus('idle')
      wsRef.current?.close()
      wsRef.current = null
      setIsConnected(false)
      return
    }

    if (message.type === 'notice') {
      // 非致命提醒：只展示提示条，绝不中断当前分析会话、绝不关闭 WebSocket 连接
      setDiagnosticNotice(message.message)
    }
  }

  /** RECORDING/PROCESSING 时在 Attempt Chain 挂载「抓取中…」占位卡 */
  function syncLiveAttemptPlaceholder(state: AutoCaptureFsmState, attemptCount: number) {
    if (state !== 'RECORDING' && state !== 'PROCESSING') {
      setAutoAttempts((prev) => prev.filter((a) => a.status === 'ready' || a.status === 'failed'))
      return
    }
    const nextNumber = Math.max(attemptCount + 1, 1)
    setAutoAttempts((prev) => {
      const ready = prev.filter((a) => a.status === 'ready' || a.status === 'failed')
      const placeholder: AutoCaptureAttempt = {
        id: `live-${nextNumber}`,
        attemptNumber: nextNumber,
        score: null,
        status: state === 'PROCESSING' ? 'processing' : 'capturing',
      }
      return [...ready, placeholder]
    })
  }

  /** 点击 Attempt Chain：中栏视口 + 右侧五维雷达倒带至该切片 */
  function applyAttemptReview(attempt: AutoCaptureAttempt) {
    setSelectedAttemptNumber(attempt.attemptNumber)
    setReviewSnapshot({
      image: attempt.impactFrameImage ?? null,
      score: attempt.score,
      quantified5dScores: attempt.quantified5dScores ?? null,
      errorCodes: attempt.errorCodes ?? [],
      spatialTrajectoryData: attempt.spatialTrajectoryData ?? null,
      videoUrl: attempt.videoUrl ?? null,
    })
    if (attempt.impactFrameImage) {
      setFrameImage(attempt.impactFrameImage)
    }
    if (attempt.score != null || attempt.quantified5dScores) {
      setFinalReport((prev) => ({
        score: attempt.score ?? prev?.score ?? 0,
        totalAttempts: autoAttemptCount || prev?.totalAttempts || attempt.attemptNumber,
        painPoint: prev?.painPoint ?? `零感捕获 Attempt #${attempt.attemptNumber} 五维诊断快照`,
        prescription: prev?.prescription ?? '继续保持发力节奏，系统已自动归档本脚切片。',
        fullText:
          prev?.fullText ??
          `Attempt #${attempt.attemptNumber} · 得分 ${attempt.score ?? '--'}（零感自动捕获）`,
        generatedAt: prev?.generatedAt ?? new Date().toLocaleString(),
        errorCodes: attempt.errorCodes ?? prev?.errorCodes ?? [],
        biomechanicsMetrics: attempt.biomechanicsMetrics ?? prev?.biomechanicsMetrics ?? null,
        quantified5dScores: attempt.quantified5dScores ?? prev?.quantified5dScores ?? null,
        spatialTrajectoryData: attempt.spatialTrajectoryData ?? prev?.spatialTrajectoryData ?? null,
        impactFrameImage: attempt.impactFrameImage ?? prev?.impactFrameImage ?? null,
        scoringEngine: prev?.scoringEngine ?? 'deterministic_biomechanics_v3_auto_capture',
      }))
    }
  }

  function handleSelectAttempt(attempt: AutoCaptureAttempt) {
    if (attempt.status !== 'ready') return
    applyAttemptReview(attempt)
  }

  function handleForceCapture() {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    setForceCaptureBusy(true)
    wsRef.current.send(JSON.stringify({ action: 'force_capture' }))
  }

  function handleReturnToLive() {
    setSelectedAttemptNumber(null)
    setReviewSnapshot(null)
  }