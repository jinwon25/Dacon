import { useEffect, useMemo, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import { toPng } from 'html-to-image'
import { createGenericSquad, createOpponentSquad } from '../data/generic'
import { formationPositions, roleOptions } from '../data/match'
import type { FormationKey, Player, Tactics } from '../types'

const formations: FormationKey[] = ['4-3-3', '4-2-3-1', '3-4-3', '5-3-2']
const storageKey = 'retactic-universal-studio-v1'

interface MatchContext {
  teamName: string
  opponentName: string
  minute: number
  ourScore: number
  theirScore: number
  phase: '경기 전' | '전반전' | '하프타임' | '후반전' | '연장전'
  objective: '균형 유지' | '득점 필요' | '리드 보호' | '압박 탈출' | '상대 역습 차단'
}

interface SavedStudio {
  context: MatchContext
  formation: FormationKey
  tactics: Tactics
  squad: Player[]
  opponentFormation?: FormationKey
  opponentVisible?: boolean
  opponents?: Player[]
  planSlots?: PlanSlots
}

interface PlanSnapshot {
  savedAt: string
  context: MatchContext
  formation: FormationKey
  tactics: Tactics
  squad: Player[]
  opponentFormation: FormationKey
  opponents: Player[]
  teamWidth: number
  defensiveLine: number
  transitionRisk: number
}

type PlanSlots = { A?: PlanSnapshot; B?: PlanSnapshot }
type LayoutSnapshot = { squad: Player[]; opponents: Player[]; formation: FormationKey; opponentFormation: FormationKey }

const defaultContext: MatchContext = {
  teamName: '우리 팀',
  opponentName: '상대 팀',
  minute: 0,
  ourScore: 0,
  theirScore: 0,
  phase: '경기 전',
  objective: '균형 유지',
}

const defaultTactics: Tactics = { pressing: 50, width: 55, tempo: 50, risk: 45 }
const clamp = (value: number, min = 0, max = 100) => Math.min(max, Math.max(min, value))

function loadSavedStudio(): SavedStudio | null {
  try {
    const raw = window.localStorage.getItem(storageKey)
    return raw ? JSON.parse(raw) as SavedStudio : null
  } catch {
    return null
  }
}

export default function UniversalStudio() {
  const saved = useRef(loadSavedStudio()).current
  const [context, setContext] = useState<MatchContext>(saved?.context ?? defaultContext)
  const [formation, setFormation] = useState<FormationKey>(saved?.formation ?? '4-3-3')
  const [tactics, setTactics] = useState<Tactics>(saved?.tactics ?? defaultTactics)
  const [squad, setSquad] = useState<Player[]>(saved?.squad ?? createGenericSquad())
  const [opponentFormation, setOpponentFormation] = useState<FormationKey>(saved?.opponentFormation ?? '4-3-3')
  const [opponentVisible, setOpponentVisible] = useState(saved?.opponentVisible ?? true)
  const [opponents, setOpponents] = useState<Player[]>(saved?.opponents ?? createOpponentSquad())
  const [planSlots, setPlanSlots] = useState<PlanSlots>(saved?.planSlots ?? {})
  const [layoutHistory, setLayoutHistory] = useState<LayoutSnapshot[]>([])
  const [selectedId, setSelectedId] = useState<string>('generic-st')
  const [draggingId, setDraggingId] = useState<string | null>(null)
  const [hasDragged, setHasDragged] = useState(false)
  const [lastSaved, setLastSaved] = useState('방금')
  const [exportStatus, setExportStatus] = useState<'idle' | 'working' | 'done' | 'error'>('idle')
  const pitchRef = useRef<HTMLDivElement>(null)
  const exportRef = useRef<HTMLDivElement>(null)

  const onPitch = squad.filter((player) => player.onPitch)
  const bench = squad.filter((player) => !player.onPitch)
  const selectedPlayer = squad.find((player) => player.id === selectedId) ?? null

  useEffect(() => {
    const timer = window.setTimeout(() => {
      window.localStorage.setItem(storageKey, JSON.stringify({ context, formation, tactics, squad, opponentFormation, opponentVisible, opponents, planSlots }))
      setLastSaved(new Intl.DateTimeFormat('ko', { hour: '2-digit', minute: '2-digit' }).format(new Date()))
    }, 350)
    return () => window.clearTimeout(timer)
  }, [context, formation, tactics, squad, opponentFormation, opponentVisible, opponents, planSlots])

  const diagnostics = useMemo(() => {
    const defenders = onPitch.filter((player) => player.position === 'DF')
    const teamWidth = Math.round(Math.max(...onPitch.map((player) => player.x)) - Math.min(...onPitch.map((player) => player.x)))
    const defensiveLine = defenders.length ? Math.round(defenders.reduce((sum, player) => sum + player.y, 0) / defenders.length) : 75
    const forwardSupport = onPitch.filter((player) => player.y < 45).length
    const transitionRisk = Math.round(tactics.risk * 0.42 + tactics.pressing * 0.28 + (formation === '3-4-3' ? 18 : formation === '5-3-2' ? -8 : 4))

    return {
      teamWidth,
      defensiveLine,
      forwardSupport,
      transitionRisk: clamp(transitionRisk),
      checks: [
        { tone: teamWidth >= 58 ? 'good' : 'warn', title: '공격 폭', detail: teamWidth >= 58 ? `좌우 간격 ${teamWidth} · 측면 활용 가능` : `좌우 간격 ${teamWidth} · 공격 간격이 좁습니다` },
        { tone: forwardSupport >= 3 ? 'good' : 'warn', title: '전방 지원', detail: `공격 구역에 ${forwardSupport}명 배치` },
        { tone: transitionRisk < 62 ? 'good' : 'risk', title: '전환 안전', detail: transitionRisk < 62 ? '공을 잃은 뒤 복귀 가능한 범위' : '압박과 위험 감수가 함께 높습니다' },
        { tone: opponentVisible ? 'good' : 'warn', title: '상대 대형', detail: opponentVisible ? `${opponentFormation} 오버레이와 간격 비교 중` : '상대 표시를 켜면 수적 관계를 확인할 수 있습니다' },
      ],
    }
  }, [formation, onPitch, opponentFormation, opponentVisible, tactics])

  const applyFormation = (next: FormationKey) => {
    pushLayoutHistory()
    setFormation(next)
    setSquad((current) => current.map((player) => {
      if (!player.onPitch || player.slot === null) return player
      return { ...player, ...formationPositions[next][player.slot] }
    }))
    setHasDragged(false)
  }

  const handlePointerDown = (event: ReactPointerEvent<HTMLButtonElement>, playerId: string) => {
    pushLayoutHistory()
    event.currentTarget.setPointerCapture(event.pointerId)
    setDraggingId(playerId)
    setSelectedId(playerId)
  }

  const handlePointerMove = (event: ReactPointerEvent<HTMLButtonElement>, playerId: string) => {
    if (draggingId !== playerId || !pitchRef.current) return
    const rect = pitchRef.current.getBoundingClientRect()
    const x = clamp((event.clientX - rect.left) / rect.width * 100, 7, 93)
    const y = clamp((event.clientY - rect.top) / rect.height * 100, 6, 94)
    setSquad((current) => current.map((player) => player.id === playerId ? { ...player, x, y } : player))
    setHasDragged(true)
  }

  const handlePointerUp = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
    setDraggingId(null)
  }

  const handleOpponentPointerMove = (event: ReactPointerEvent<HTMLButtonElement>, playerId: string) => {
    if (draggingId !== playerId || !pitchRef.current) return
    const rect = pitchRef.current.getBoundingClientRect()
    const x = clamp((event.clientX - rect.left) / rect.width * 100, 7, 93)
    const y = clamp((event.clientY - rect.top) / rect.height * 100, 6, 94)
    setOpponents((current) => current.map((player) => player.id === playerId ? { ...player, x, y } : player))
    setHasDragged(true)
  }

  const applyOpponentFormation = (next: FormationKey) => {
    pushLayoutHistory()
    setOpponentFormation(next)
    setOpponents(createOpponentSquad(next))
  }

  const updatePlayer = (changes: Partial<Player>) => {
    if (!selectedPlayer) return
    setSquad((current) => current.map((player) => player.id === selectedPlayer.id ? { ...player, ...changes } : player))
  }

  const swapWithBench = (incomingId: string) => {
    if (!selectedPlayer?.onPitch || selectedPlayer.slot === null) {
      setSelectedId(incomingId)
      return
    }
    pushLayoutHistory()
    const slot = selectedPlayer.slot
    const coordinate = { x: selectedPlayer.x, y: selectedPlayer.y }
    setSquad((current) => current.map((player) => {
      if (player.id === selectedPlayer.id) return { ...player, onPitch: false, slot: null, x: 0, y: 0 }
      if (player.id === incomingId) return { ...player, onPitch: true, slot, ...coordinate }
      return player
    }))
    setSelectedId(incomingId)
  }

  const resetStudio = () => {
    if (!window.confirm('현재 전술안을 지우고 기본 4-3-3으로 초기화할까요?')) return
    setContext(defaultContext)
    setFormation('4-3-3')
    setTactics(defaultTactics)
    setSquad(createGenericSquad())
    setOpponentFormation('4-3-3')
    setOpponentVisible(true)
    setOpponents(createOpponentSquad())
    setSelectedId('generic-st')
    setHasDragged(false)
    setPlanSlots({})
    setLayoutHistory([])
    window.localStorage.removeItem(storageKey)
  }

  function pushLayoutHistory() {
    const snapshot: LayoutSnapshot = {
      squad: squad.map((player) => ({ ...player })),
      opponents: opponents.map((player) => ({ ...player })),
      formation,
      opponentFormation,
    }
    setLayoutHistory((current) => [...current.slice(-9), snapshot])
  }

  const undoLayout = () => {
    const previous = layoutHistory[layoutHistory.length - 1]
    if (!previous) return
    setSquad(previous.squad)
    setOpponents(previous.opponents)
    setFormation(previous.formation)
    setOpponentFormation(previous.opponentFormation)
    setLayoutHistory((current) => current.slice(0, -1))
  }

  const savePlan = (slot: 'A' | 'B') => {
    const snapshot: PlanSnapshot = {
      savedAt: new Intl.DateTimeFormat('ko', { hour: '2-digit', minute: '2-digit' }).format(new Date()),
      context: { ...context },
      formation,
      tactics: { ...tactics },
      squad: squad.map((player) => ({ ...player })),
      opponentFormation,
      opponents: opponents.map((player) => ({ ...player })),
      teamWidth: diagnostics.teamWidth,
      defensiveLine: diagnostics.defensiveLine,
      transitionRisk: diagnostics.transitionRisk,
    }
    setPlanSlots((current) => ({ ...current, [slot]: snapshot }))
  }

  const loadPlan = (snapshot: PlanSnapshot) => {
    pushLayoutHistory()
    setContext(snapshot.context)
    setFormation(snapshot.formation)
    setTactics(snapshot.tactics)
    setSquad(snapshot.squad)
    setOpponentFormation(snapshot.opponentFormation)
    setOpponents(snapshot.opponents)
  }

  const exportPng = async () => {
    if (!exportRef.current || exportStatus === 'working') return
    setExportStatus('working')
    try {
      await document.fonts.ready
      const dataUrl = await toPng(exportRef.current, {
        cacheBust: true,
        pixelRatio: 2,
        backgroundColor: '#07130f',
        width: 1200,
        height: 675,
        style: { position: 'static', left: '0', top: '0', zIndex: '0' },
      })
      const anchor = document.createElement('a')
      const safeTeam = (context.teamName || 'team').replace(/[^가-힣a-zA-Z0-9-_]/g, '-')
      anchor.download = `retactic-${safeTeam}-${formation}.png`
      anchor.href = dataUrl
      anchor.click()
      setExportStatus('done')
      window.setTimeout(() => setExportStatus('idle'), 2200)
    } catch {
      setExportStatus('error')
      window.setTimeout(() => setExportStatus('idle'), 3000)
    }
  }

  return (
    <main className="universal-studio page-wrap wide">
      <header className="studio-heading">
        <div>
          <p className="eyebrow">UNIVERSAL TACTICS STUDIO</p>
          <h1><span>모든 경기를 위한</span><span>범용 전술 보드</span></h1>
          <p>경기 정보와 선수를 입력하세요.<br />전술의 구조적 위험을 바로 확인할 수 있습니다.</p>
        </div>
        <div className="studio-actions">
          <span className="save-status"><i /> 자동 저장 {lastSaved}</span>
          <button className="secondary-button compact" type="button" onClick={resetStudio}>초기화</button>
          <button className="primary-button compact" type="button" onClick={exportPng} disabled={exportStatus === 'working'}>{exportStatus === 'working' ? '이미지 생성 중…' : 'PNG 저장'}</button>
        </div>
      </header>

      <div className="studio-steps" aria-label="사용 순서">
        <span className="active"><b>1</b><i>경기 정보</i><small>팀과 목표 입력</small></span>
        <span className="active"><b>2</b><i>선수 배치</i><small>포메이션과 위치</small></span>
        <span className="active"><b>3</b><i>팀 지시</i><small>압박·폭·템포</small></span>
        <span className="active"><b>4</b><i>검토·저장</i><small>리스크 확인</small></span>
      </div>

      <div className="studio-layout">
        <aside className="studio-left">
          <section className="panel studio-panel" id="studio-context">
            <div className="panel-title"><span>01</span><div><small>MATCH CONTEXT</small><h2>경기 정보</h2></div></div>
            <div className="form-grid">
              <label>우리 팀<input value={context.teamName} onChange={(event) => setContext({ ...context, teamName: event.target.value })} /></label>
              <label>상대 팀<input value={context.opponentName} onChange={(event) => setContext({ ...context, opponentName: event.target.value })} /></label>
              <label>경기 단계<select value={context.phase} onChange={(event) => setContext({ ...context, phase: event.target.value as MatchContext['phase'] })}>{['경기 전', '전반전', '하프타임', '후반전', '연장전'].map((item) => <option key={item}>{item}</option>)}</select></label>
              <label>분<input type="number" min="0" max="130" value={context.minute} onChange={(event) => setContext({ ...context, minute: Number(event.target.value) })} /></label>
            </div>
            <div className="score-editor">
              <label><span>{context.teamName || '우리 팀'}</span><input type="number" min="0" max="20" value={context.ourScore} onChange={(event) => setContext({ ...context, ourScore: Number(event.target.value) })} /></label>
              <b>:</b>
              <label><span>{context.opponentName || '상대 팀'}</span><input type="number" min="0" max="20" value={context.theirScore} onChange={(event) => setContext({ ...context, theirScore: Number(event.target.value) })} /></label>
            </div>
            <label className="full-field">이번 전술의 목표<select value={context.objective} onChange={(event) => setContext({ ...context, objective: event.target.value as MatchContext['objective'] })}>{['균형 유지', '득점 필요', '리드 보호', '압박 탈출', '상대 역습 차단'].map((item) => <option key={item}>{item}</option>)}</select></label>
          </section>

          <section className="panel studio-panel">
            <div className="panel-title"><span>03</span><div><small>TEAM INSTRUCTIONS</small><h2>팀 지시</h2></div></div>
            <RangeControl label="압박 강도" low="기다리기" high="즉시 압박" value={tactics.pressing} onChange={(value) => setTactics({ ...tactics, pressing: value })} />
            <RangeControl label="공격 폭" low="좁게" high="넓게" value={tactics.width} onChange={(value) => setTactics({ ...tactics, width: value })} />
            <RangeControl label="공격 템포" low="차분하게" high="빠르게" value={tactics.tempo} onChange={(value) => setTactics({ ...tactics, tempo: value })} />
            <RangeControl label="위험 감수" low="안전하게" high="과감하게" value={tactics.risk} onChange={(value) => setTactics({ ...tactics, risk: value })} />
          </section>
        </aside>

        <section className="studio-board">
          <div className="board-toolbar">
            <div><small>02 · FORMATION</small><strong>{formation}</strong></div>
            <div className="formation-pills">{formations.map((item) => <button type="button" className={formation === item ? 'active' : ''} key={item} onClick={() => applyFormation(item)}>{item}</button>)}</div>
            <div className="board-actions"><button className="reset-layout" type="button" disabled={layoutHistory.length === 0} onClick={undoLayout}>↶ 실행 취소</button><button className="reset-layout" type="button" onClick={() => applyFormation(formation)}>배치 원위치</button></div>
          </div>
          <div className="opponent-toolbar">
            <label><input type="checkbox" checked={opponentVisible} onChange={(event) => setOpponentVisible(event.target.checked)} /><span>상대팀 표시</span></label>
            <div><small>{context.opponentName || '상대 팀'} 대형</small>{formations.map((item) => <button type="button" className={opponentFormation === item ? 'active' : ''} key={item} onClick={() => applyOpponentFormation(item)}>{item}</button>)}</div>
            <p>붉은 선수를 직접 움직여 상대 압박 구조를 재현하세요.</p>
          </div>
          <div className="pitch studio-pitch" ref={pitchRef}>
            <div className="pitch-lines"><i className="halfway" /><i className="circle" /><i className="box top" /><i className="box bottom" /></div>
            <div className="attack-label">↑ {context.teamName || '우리 팀'} 공격 방향</div>
            {!hasDragged && <div className="drag-coachmark"><span>↕</span><strong>선수를 끌어서 위치를 바꿔보세요</strong><small>클릭하면 이름과 역할을 수정할 수 있습니다</small></div>}
            {opponentVisible && opponents.map((player) => (
              <button
                className={`player-token opponent-token ${draggingId === player.id ? 'dragging' : ''}`}
                type="button"
                key={player.id}
                style={{ left: `${player.x}%`, top: `${player.y}%` }}
                onPointerDown={(event) => { pushLayoutHistory(); event.currentTarget.setPointerCapture(event.pointerId); setDraggingId(player.id) }}
                onPointerMove={(event) => handleOpponentPointerMove(event, player.id)}
                onPointerUp={handlePointerUp}
                aria-label={`${context.opponentName} ${player.shortName}`}
              ><span>{player.number}</span><strong>{player.shortName}</strong></button>
            ))}
            {onPitch.map((player) => (
              <button
                className={`player-token ${selectedId === player.id ? 'selected' : ''} ${draggingId === player.id ? 'dragging' : ''}`}
                type="button"
                key={player.id}
                style={{ left: `${player.x}%`, top: `${player.y}%` }}
                onPointerDown={(event) => handlePointerDown(event, player.id)}
                onPointerMove={(event) => handlePointerMove(event, player.id)}
                onPointerUp={handlePointerUp}
                aria-label={`${player.name}, ${player.role}`}
              ><span>{player.number}</span><strong>{player.shortName}</strong></button>
            ))}
          </div>
          <div className="bench panel studio-bench">
            <div className="bench-heading"><div><small>BENCH</small><strong>필드 선수를 선택한 뒤 교체 선수를 누르세요</strong></div><span>{bench.length}명</span></div>
            <div className="bench-list">{bench.map((player) => <button type="button" key={player.id} onClick={() => swapWithBench(player.id)}><span>{player.number}</span><div><strong>{player.shortName}</strong><small>{player.name}</small></div>{selectedPlayer?.onPitch && <em>교체</em>}</button>)}</div>
          </div>
        </section>

        <aside className="studio-right">
          <section className="panel studio-panel player-editor">
            <div className="panel-title"><span>02</span><div><small>SELECTED PLAYER</small><h2>선수 정보</h2></div></div>
            {selectedPlayer && <>
              <div className="selected-summary"><b>{selectedPlayer.number}</b><div><strong>{selectedPlayer.name}</strong><small>{selectedPlayer.onPitch ? '필드 선수' : '벤치 선수'}</small></div></div>
              <div className="form-grid player-fields">
                <label>표시 이름<input maxLength={10} value={selectedPlayer.shortName} onChange={(event) => updatePlayer({ shortName: event.target.value, name: event.target.value })} /></label>
                <label>등번호<input type="number" min="1" max="99" value={selectedPlayer.number} onChange={(event) => updatePlayer({ number: Number(event.target.value) })} /></label>
              </div>
              <label className="full-field">역할<select value={selectedPlayer.role} onChange={(event) => updatePlayer({ role: event.target.value })}>{roleOptions[selectedPlayer.position].map((role) => <option key={role}>{role}</option>)}</select></label>
            </>}
          </section>

          <section className="panel studio-panel diagnostics-panel">
            <div className="panel-title"><span>04</span><div><small>STRUCTURE CHECK</small><h2>실시간 전술 점검</h2></div></div>
            <div className="diagnostic-summary"><span><small>팀 폭</small><b>{diagnostics.teamWidth}</b></span><span><small>수비 라인</small><b>{diagnostics.defensiveLine}</b></span><span><small>전환 위험</small><b>{diagnostics.transitionRisk}</b></span></div>
            <ul className="diagnostic-list">{diagnostics.checks.map((check) => <li className={check.tone} key={check.title}><i>{check.tone === 'good' ? '✓' : '!'}</i><div><strong>{check.title}</strong><span>{check.detail}</span></div></li>)}</ul>
            <div className="studio-recommendation"><small>COACHING NOTE</small><strong>{context.objective}</strong><p>{getStudioNote(context, tactics, diagnostics.transitionRisk)}</p></div>
          </section>

          <section className="panel studio-panel session-summary">
            <small>현재 전술안</small>
            <h3>{context.teamName} vs {context.opponentName}</h3>
            <p>{context.phase} {context.minute > 0 ? `${context.minute}분` : ''} · {context.ourScore}:{context.theirScore} · {formation}</p>
            <div className="plan-slot-buttons"><button type="button" onClick={() => savePlan('A')}>A안에 저장</button><button type="button" onClick={() => savePlan('B')}>B안에 저장</button></div>
            <div className="plan-slots">
              <PlanSlot label="A" snapshot={planSlots.A} onLoad={loadPlan} />
              <PlanSlot label="B" snapshot={planSlots.B} onLoad={loadPlan} />
            </div>
            {planSlots.A && planSlots.B && <div className="ab-difference"><small>A/B 차이</small><span>전환 위험 <b>{signed(planSlots.B.transitionRisk - planSlots.A.transitionRisk)}</b></span><span>팀 폭 <b>{signed(planSlots.B.teamWidth - planSlots.A.teamWidth)}</b></span></div>}
            <div className="export-actions"><button className="secondary-button" type="button" onClick={() => window.print()}>PDF</button><button className="primary-button" type="button" onClick={exportPng}>PNG 저장 <span>↓</span></button></div>
            {exportStatus === 'done' && <p className="export-feedback success">PNG 저장이 완료됐습니다.</p>}
            {exportStatus === 'error' && <p className="export-feedback error">이미지 생성에 실패했습니다. 다시 시도해주세요.</p>}
          </section>
        </aside>
      </div>

      <div className="export-card" ref={exportRef} aria-hidden="true">
        <header><div><span>R:</span><strong>RE:TACTIC</strong></div><small>TACTICAL PLAN · {new Date().toISOString().slice(0, 10)}</small></header>
        <div className="export-card-body">
          <section className="export-pitch">
            <div className="export-pitch-lines"><i /><i /><i /></div>
            {opponentVisible && opponents.map((player) => <span className="export-player opponent" key={`export-${player.id}`} style={{ left: `${player.x}%`, top: `${player.y}%` }}>{player.shortName}</span>)}
            {onPitch.map((player) => <span className="export-player ours" key={`export-${player.id}`} style={{ left: `${player.x}%`, top: `${player.y}%` }}>{player.shortName}</span>)}
          </section>
          <aside className="export-summary">
            <p>MATCH PLAN</p>
            <h2>{context.teamName} <b>{context.ourScore}:{context.theirScore}</b> {context.opponentName}</h2>
            <span>{context.phase} {context.minute ? `${context.minute}′` : ''} · 목표: {context.objective}</span>
            <div className="export-formations"><i><small>우리 대형</small><b>{formation}</b></i><i><small>상대 대형</small><b>{opponentFormation}</b></i></div>
            <div className="export-metrics"><i><small>팀 폭</small><b>{diagnostics.teamWidth}</b></i><i><small>수비 라인</small><b>{diagnostics.defensiveLine}</b></i><i><small>전환 위험</small><b>{diagnostics.transitionRisk}</b></i></div>
            <div className="export-note"><small>COACHING NOTE</small><strong>{context.objective}</strong><p>{getStudioNote(context, tactics, diagnostics.transitionRisk)}</p></div>
          </aside>
        </div>
        <footer><span>Designed with RE:TACTIC Universal Tactics Studio</span><b>jinwon25.github.io/Dacon</b></footer>
      </div>
    </main>
  )
}

function PlanSlot({ label, snapshot, onLoad }: { label: 'A' | 'B'; snapshot?: PlanSnapshot; onLoad: (snapshot: PlanSnapshot) => void }) {
  return <div className={`plan-slot ${snapshot ? 'saved' : ''}`}><b>{label}</b>{snapshot ? <><span>{snapshot.formation}</span><small>위험 {snapshot.transitionRisk} · {snapshot.savedAt}</small><button type="button" onClick={() => onLoad(snapshot)}>불러오기</button></> : <><span>비어 있음</span><small>위 버튼으로 현재 안 저장</small></>}</div>
}

const signed = (value: number) => value > 0 ? `+${value}` : `${value}`

function RangeControl({ label, low, high, value, onChange }: { label: string; low: string; high: string; value: number; onChange: (value: number) => void }) {
  return <label className="tactic-slider"><span><strong>{label}</strong><b>{value}</b></span><input type="range" min="0" max="100" value={value} onChange={(event) => onChange(Number(event.target.value))} style={{ '--range': `${value}%` } as React.CSSProperties} /><small><i>{low}</i><i>{high}</i></small></label>
}

function getStudioNote(context: MatchContext, tactics: Tactics, risk: number) {
  if (risk > 65) return '공을 잃은 순간 중앙을 지킬 선수를 한 명 지정하세요. 공격과 압박을 동시에 높이면 복귀 거리가 길어집니다.'
  if (context.objective === '득점 필요' && tactics.tempo < 55) return '득점이 필요하지만 템포가 낮습니다. 공격 전환 속도를 한 단계 높이는 안을 비교해보세요.'
  if (context.objective === '리드 보호' && tactics.risk > 55) return '리드 보호 목표에 비해 위험 감수가 높습니다. 풀백 한 명의 전진을 제한하면 균형이 좋아집니다.'
  if (context.objective === '압박 탈출') return '첫 번째 패스 옵션과 반대편 전환 선수를 멀리 두어 상대 압박 폭을 늘리세요.'
  return '현재 구조는 큰 불균형이 없습니다. 선수 역할이 실제 움직임과 일치하는지 마지막으로 확인하세요.'
}
