import { useMemo, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'
import UniversalStudio from './components/UniversalStudio'
import SpatialEvidence from './components/SpatialEvidence'
import TacticalSequence from './components/TacticalSequence'
import PassNetwork from './components/PassNetworkPro'
import BallFlow from './components/BallFlow'
import MatchStats from './components/MatchStats'
import CountryFlag from './components/CountryFlag'
import OfficialReportEvidence from './components/OfficialReportEvidence'
import { evidenceMethod } from './data/evidence'
import { formationGroups, formationGuidance, formationPositions, formations, roleOptions } from './data/match'
import { cloneScenarioSquad, defaultScenarioId, guidedScenarios, type GuidedScenario, type GuidedScenarioId } from './data/scenarios'
import type { FormationKey, Metrics, Player, Stage, Tactics } from './types'

const stageOrder: Stage[] = ['intro', 'briefing', 'tactics', 'result']
const stageLabels: Record<Stage, string> = { intro: '시작', briefing: '진단', tactics: '설계', result: '검토' }
const mobileStageActions: Record<Stage, string> = { intro: '브리핑', briefing: '전술 보드', tactics: '비교 실행', result: '다시 설계' }
type AppMode = 'home' | 'guided' | 'studio'
type GuidedSubstitution = {
  id: string
  outgoingId: string
  outgoingName: string
  incomingId: string
  incomingName: string
  squadBefore: Player[]
  selectedIdBefore: string
}

const tacticPresets: Array<{ id: string; label: string; detail: string; values: Tactics }> = [
  { id: 'control', label: '균형 유지', detail: '간격과 점유 우선', values: { pressing: 52, width: 55, tempo: 50, risk: 42 } },
  { id: 'possession', label: '점유 회복', detail: '좁은 지원과 재압박', values: { pressing: 64, width: 48, tempo: 45, risk: 36 } },
  { id: 'transition', label: '전환 공격', detail: '넓고 빠른 전진', values: { pressing: 56, width: 70, tempo: 78, risk: 62 } },
  { id: 'chase', label: '종료 압박', detail: '득점 우선 고위험', values: { pressing: 82, width: 74, tempo: 86, risk: 84 } },
]

const getInitialMode = (): AppMode => {
  if (window.location.hash === '#studio') return 'studio'
  if (stageOrder.some((stage) => window.location.hash === `#${stage}`)) return 'guided'
  return 'home'
}

const getInitialStage = (): Stage => {
  const hashStage = window.location.hash.replace('#', '') as Stage
  return stageOrder.includes(hashStage) ? hashStage : 'intro'
}

const clamp = (value: number, min = 0, max = 100) => Math.min(max, Math.max(min, value))

function revertSubstitutionPlan(squad: Player[], records: GuidedSubstitution[]) {
  return [...records].reverse().reduce((current, record) => {
    const incomingOnPitch = current.find((player) => player.id === record.incomingId)
    const outgoingBefore = record.squadBefore.find((player) => player.id === record.outgoingId)
    const incomingBefore = record.squadBefore.find((player) => player.id === record.incomingId)
    if (!incomingOnPitch || !outgoingBefore || !incomingBefore) return current
    return current.map((player) => {
      if (player.id === record.outgoingId) {
        return { ...outgoingBefore, onPitch: true, slot: incomingOnPitch.slot, x: incomingOnPitch.x, y: incomingOnPitch.y }
      }
      if (player.id === record.incomingId) {
        return { ...incomingBefore, onPitch: false, slot: null, x: 0, y: 0 }
      }
      return player
    })
  }, squad.map((player) => ({ ...player })))
}

function calculateMetrics(
  squad: Player[],
  tactics: Tactics,
  formation: FormationKey,
  scenario: GuidedScenario,
): Metrics {
  const impactPlayerOnPitch = squad.some((player) => player.onPitch && player.id === scenario.impactPlayerId)
  const attackBoost = formation === '3-4-3' ? 8 : formation === '4-2-3-1' ? 4 : formation === '5-3-2' ? -4 : 2
  const safetyBoost = formation === '5-3-2' ? 10 : formation === '3-4-3' ? -6 : 2
  const observedPassGap = scenario.evidence.opponent.passCompletion - scenario.evidence.ours.passCompletion
  const observedPressureLoad = scenario.officialReport ? scenario.evidence.ours.pressures / 9 : scenario.evidence.ours.pressures
  const outfield = squad.filter((player) => player.onPitch && player.position !== 'GK')
  const averageY = outfield.reduce((sum, player) => sum + player.y, 0) / Math.max(outfield.length, 1)
  const widthSpread = Math.max(...outfield.map((player) => player.x), 50) - Math.min(...outfield.map((player) => player.x), 50)
  const forwardShift = clamp((53 - averageY) * .45, -8, 8)
  const widthFit = clamp(8 - Math.abs(widthSpread - tactics.width) * .15, -5, 8)
  const attackingRoles = outfield.filter((player) => /공격|라인 브레이커|인사이드|포처|공간 침투/.test(player.role)).length
  const roleBoost = clamp((attackingRoles - 3) * 1.4, -4, 6)

  if (scenario.id === 'argentina-netherlands-83') {
    return {
      threat: clamp(Math.round(34 + tactics.tempo * .2 + tactics.risk * .18 + attackBoost + forwardShift + roleBoost + (impactPlayerOnPitch ? -2 : 3))),
      control: clamp(Math.round(58 - observedPassGap * .45 + (100 - tactics.risk) * .16 + tactics.width * .08 + widthFit * .45 + (impactPlayerOnPitch ? 5 : 0))),
      exposure: clamp(Math.round(62 + tactics.risk * .2 + tactics.pressing * .08 + forwardShift * .7 - widthFit * .35 - safetyBoost - (impactPlayerOnPitch ? 9 : 0))),
      fatigue: clamp(Math.round(34 + observedPressureLoad * .35 + tactics.pressing * .25 + (impactPlayerOnPitch ? -4 : 2))),
    }
  }

  return {
    threat: clamp(Math.round(24 + tactics.tempo * 0.25 + tactics.risk * 0.28 + attackBoost + forwardShift + roleBoost + (impactPlayerOnPitch ? 7 : 0))),
    control: clamp(Math.round(52 - observedPassGap * 0.6 + tactics.width * 0.12 + (100 - tactics.risk) * 0.08 + widthFit * .5)),
    exposure: clamp(Math.round(12 + tactics.pressing * 0.14 + tactics.risk * 0.35 + forwardShift * .7 - widthFit * .3 - safetyBoost)),
    fatigue: clamp(Math.round(12 + observedPressureLoad * 0.45 + tactics.pressing * 0.28 + tactics.tempo * 0.16)),
  }
}

function App() {
  const [scenarioId, setScenarioId] = useState<GuidedScenarioId>(defaultScenarioId)
  const scenario = guidedScenarios[scenarioId]
  const [mode, setMode] = useState<AppMode>(getInitialMode)
  const [stage, setStage] = useState<Stage>(getInitialStage)
  const [formation, setFormation] = useState<FormationKey>(scenario.defaultFormation)
  const [squad, setSquad] = useState<Player[]>(() => cloneScenarioSquad(scenario))
  const [selectedId, setSelectedId] = useState<string>(scenario.selectedPlayerId)
  const [draggingId, setDraggingId] = useState<string | null>(null)
  const [substitutions, setSubstitutions] = useState<GuidedSubstitution[]>([])
  const [tactics, setTactics] = useState<Tactics>({ ...scenario.defaultTactics })
  const pitchRef = useRef<HTMLDivElement>(null)

  const selectedPlayer = squad.find((player) => player.id === selectedId) ?? null
  const metrics = useMemo(() => calculateMetrics(squad, tactics, formation, scenario), [squad, tactics, formation, scenario])

  const goToStage = (nextStage: Stage) => {
    setStage(nextStage)
    window.history.replaceState(null, '', `#${nextStage}`)
  }

  const selectScenario = (nextId: GuidedScenarioId) => {
    const next = guidedScenarios[nextId]
    setScenarioId(nextId)
    setStage('intro')
    setFormation(next.defaultFormation)
    setSquad(cloneScenarioSquad(next))
    setSelectedId(next.selectedPlayerId)
    setDraggingId(null)
    setSubstitutions([])
    setTactics({ ...next.defaultTactics })
    window.history.replaceState(null, '', '#intro')
  }

  const setFormationPreset = (nextFormation: FormationKey) => {
    setFormation(nextFormation)
    setSquad((current) => current.map((player) => {
      if (!player.onPitch || player.slot === null) return player
      return { ...player, ...formationPositions[nextFormation][player.slot] }
    }))
  }

  const updateTactic = (key: keyof Tactics, value: number) => {
    setTactics((current) => ({ ...current, [key]: value }))
  }

  const applyTacticPreset = (values: Tactics) => {
    setTactics({ ...values })
  }

  const handlePointerDown = (event: ReactPointerEvent<HTMLButtonElement>, playerId: string) => {
    event.currentTarget.setPointerCapture(event.pointerId)
    setDraggingId(playerId)
    setSelectedId(playerId)
  }

  const handlePointerMove = (event: ReactPointerEvent<HTMLButtonElement>, playerId: string) => {
    if (draggingId !== playerId || !pitchRef.current) return
    const rect = pitchRef.current.getBoundingClientRect()
    const x = clamp(((event.clientX - rect.left) / rect.width) * 100, 7, 93)
    const y = clamp(((event.clientY - rect.top) / rect.height) * 100, 6, 94)
    setSquad((current) => current.map((player) => player.id === playerId ? { ...player, x, y } : player))
  }

  const handlePointerUp = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    setDraggingId(null)
  }

  const movePlayerWithKeyboard = (event: ReactKeyboardEvent<HTMLButtonElement>, playerId: string) => {
    const offsets: Partial<Record<string, [number, number]>> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
    }
    const offset = offsets[event.key]
    if (!offset) return
    event.preventDefault()
    const step = event.shiftKey ? 5 : 2
    setSelectedId(playerId)
    setSquad((current) => current.map((player) => player.id === playerId
      ? { ...player, x: clamp(player.x + offset[0] * step, 7, 93), y: clamp(player.y + offset[1] * step, 6, 94) }
      : player))
  }

  const changeRole = (role: string) => {
    if (!selectedPlayer) return
    setSquad((current) => current.map((player) => player.id === selectedPlayer.id ? { ...player, role } : player))
  }

  const substitute = (incomingId: string) => {
    const incomingPlayer = squad.find((player) => player.id === incomingId)
    const cannotReturn = substitutions.some((item) => item.outgoingId === incomingId)
    if (!selectedPlayer?.onPitch || selectedPlayer.slot === null || !incomingPlayer || incomingPlayer.onPitch || cannotReturn || substitutions.length >= 5) return
    const outgoingSlot = selectedPlayer.slot
    const outgoingCoordinate = { x: selectedPlayer.x, y: selectedPlayer.y }
    const record: GuidedSubstitution = {
      id: `sub-${substitutions.length + 1}-${selectedPlayer.id}-${incomingId}`,
      outgoingId: selectedPlayer.id,
      outgoingName: selectedPlayer.shortName,
      incomingId,
      incomingName: incomingPlayer.shortName,
      squadBefore: squad.map((player) => ({ ...player })),
      selectedIdBefore: selectedId,
    }
    setSquad((current) => current.map((player) => {
      if (player.id === selectedPlayer.id) {
        return { ...player, onPitch: false, slot: null, x: 0, y: 0 }
      }
      if (player.id === incomingId) {
        return { ...player, onPitch: true, slot: outgoingSlot, ...outgoingCoordinate }
      }
      return player
    }))
    setSelectedId(incomingId)
    setSubstitutions((current) => [...current, record])
  }

  const undoLastSubstitution = () => {
    const previous = substitutions[substitutions.length - 1]
    if (!previous) return
    setSquad((current) => revertSubstitutionPlan(current, [previous]))
    setSelectedId(previous.selectedIdBefore)
    setSubstitutions((current) => current.slice(0, -1))
  }

  const resetSubstitutions = () => {
    const first = substitutions[0]
    if (!first) return
    if (!window.confirm('계획한 교체를 모두 취소하고 교체 전 명단으로 돌아갈까요?')) return
    setSquad((current) => revertSubstitutionPlan(current, substitutions))
    setSelectedId(first.selectedIdBefore)
    setSubstitutions([])
  }

  const goHome = () => {
    setMode('home')
    setStage('intro')
    setFormation(scenario.defaultFormation)
    setSquad(cloneScenarioSquad(scenario))
    setSelectedId(scenario.selectedPlayerId)
    setSubstitutions([])
    setTactics({ ...scenario.defaultTactics })
    window.history.replaceState(null, '', window.location.pathname)
  }

  const switchMode = (nextMode: AppMode) => {
    setMode(nextMode)
    if (nextMode === 'guided' && mode === 'home') setStage('intro')
    window.history.replaceState(null, '', nextMode === 'home' ? window.location.pathname : nextMode === 'studio' ? '#studio' : `#${mode === 'home' ? 'intro' : stage}`)
  }

  const advanceMobileStage = () => {
    const currentIndex = stageOrder.indexOf(stage)
    goToStage(stage === 'result' ? 'tactics' : stageOrder[Math.min(currentIndex + 1, stageOrder.length - 1)])
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">본문 바로가기</a>
      <header className="topbar">
        <button className="brand" type="button" onClick={goHome} aria-label="서비스 홈으로">
          <span className="brand-mark">R:</span>
          <span>RE:TACTIC</span>
        </button>
        <div className="topbar-center">
          <nav className="mode-switch" aria-label="서비스 모드">
            <button type="button" aria-pressed={mode === 'guided'} className={mode === 'guided' ? 'active' : ''} onClick={() => switchMode('guided')}><span className="mode-long">실제 경기 분석</span><span className="mode-short">경기 분석</span></button>
            <button type="button" aria-pressed={mode === 'studio'} className={mode === 'studio' ? 'active' : ''} onClick={() => switchMode('studio')}><span className="mode-long">범용 전술 스튜디오</span><span className="mode-short">전술판</span></button>
          </nav>
          {mode === 'guided' && <div className="match-chip">
            <span><CountryFlag code={scenario.ours.short} label={scenario.ours.name} /> {scenario.ours.short}</span><strong>{scenario.score[0]} : {scenario.score[1]}</strong><span>{scenario.opponent.short} <CountryFlag code={scenario.opponent.short} label={scenario.opponent.name} /></span><em>{scenario.minute}′</em>
          </div>}
        </div>
        {mode === 'guided' ? <nav className="stage-nav" aria-label="진행 단계">
          {stageOrder.map((item, index) => {
            const currentIndex = stageOrder.indexOf(stage)
            const state = index === currentIndex ? 'current' : index < currentIndex ? 'completed' : ''
            return <button type="button" key={item} className={state} aria-current={index === currentIndex ? 'step' : undefined} disabled={index > currentIndex} onClick={() => goToStage(item)}><i>{index < currentIndex ? '✓' : index + 1}</i><b>{stageLabels[item]}</b></button>
          })}
        </nav> : mode === 'studio' ? <span className="studio-top-status"><i /> 작업 내용 자동 저장</span> : <span className="home-top-status">데이터 · 판단 · 전술</span>}
      </header>
      {mode !== 'home' && <div className="mobile-navigation">
        <nav className="mobile-mode-nav" aria-label="모바일 서비스 모드">
          <button type="button" aria-pressed={mode === 'guided'} className={mode === 'guided' ? 'active' : ''} onClick={() => switchMode('guided')}>실제 경기 분석</button>
          <button type="button" aria-pressed={mode === 'studio'} className={mode === 'studio' ? 'active' : ''} onClick={() => switchMode('studio')}>범용 전술판</button>
        </nav>
        {mode === 'guided' && <div className="mobile-stage-status" role="region" aria-label={`현재 단계 ${stageLabels[stage]}, ${stageOrder.indexOf(stage) + 1}/${stageOrder.length}`}>
          <span><b>{stageLabels[stage]}</b><small>{stageOrder.indexOf(stage) + 1} / {stageOrder.length}</small></span>
          <i aria-hidden="true"><b style={{ width: `${((stageOrder.indexOf(stage) + 1) / stageOrder.length) * 100}%` }} /></i>
          <button className="mobile-stage-action" type="button" onClick={advanceMobileStage}>{mobileStageActions[stage]} <span aria-hidden="true">→</span></button>
        </div>}
      </div>}

      <div id="main-content" tabIndex={-1}>
      {mode === 'home' ? <HomeScreen onGuided={() => switchMode('guided')} onStudio={() => switchMode('studio')} /> : mode === 'studio' ? <UniversalStudio /> : <>
      {stage === 'intro' && <IntroScreen scenario={scenario} scenarioId={scenarioId} onScenario={selectScenario} onStart={() => goToStage('briefing')} onStudio={() => switchMode('studio')} />}
      {stage === 'briefing' && <BriefingScreen scenario={scenario} onBack={() => goToStage('intro')} onNext={() => goToStage('tactics')} />}
      {stage === 'tactics' && (
        <TacticsScreen
          scenario={scenario}
          formation={formation}
          squad={squad}
          selectedPlayer={selectedPlayer}
          draggingId={draggingId}
          substitutions={substitutions}
          tactics={tactics}
          metrics={metrics}
          pitchRef={pitchRef}
          onBack={() => goToStage('briefing')}
          onFormation={setFormationPreset}
          onTactic={updateTactic}
          onTacticPreset={applyTacticPreset}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onKeyboardMove={movePlayerWithKeyboard}
          onSelect={setSelectedId}
          onRole={changeRole}
          onSubstitute={substitute}
          onUndoSubstitution={undoLastSubstitution}
          onResetSubstitutions={resetSubstitutions}
          onSubmit={() => goToStage('result')}
        />
      )}
      {stage === 'result' && <ResultScreen scenario={scenario} metrics={metrics} squad={squad} tactics={tactics} formation={formation} substitutions={substitutions} onRetry={() => goToStage('tactics')} />}
      </>}
      </div>
    </div>
  )
}

function HomeScreen({ onGuided, onStudio }: { onGuided: () => void; onStudio: () => void }) {
  return (
    <main className="home-screen">
      <section className="home-hero">
        <p className="eyebrow">월드컵 전술 의사결정 연구실</p>
        <h1><span>경기를 읽고,</span><strong>전술로 답하다.</strong></h1>
        <p>실제 월드컵 장면을 데이터로 진단하고,<br />당신의 배치와 지시가 만든 변화를 비교하세요.</p>
        <div className="home-actions">
          <button className="primary-button large" type="button" onClick={onGuided}>실제 경기에서 시작 <span>→</span></button>
          <button className="secondary-button large" type="button" onClick={onStudio}>빈 전술판 열기</button>
        </div>
        <div className="home-proof" aria-label="서비스 특징">
          <span><b>02</b><small>실제 경기 미션</small></span>
          <span><b>키 불필요</b><small>설치·로그인 없음</small></span>
          <span><b>전술 비교</b><small>기준 전술과 비교</small></span>
        </div>
      </section>

      <section className="home-visual" aria-label="RE:TACTIC 전술 분석 미리보기">
        <div className="home-visual-head"><span>실시간 전술 작업</span><b>65′</b></div>
        <div className="home-pitch">
          <i className="home-halfway" /><i className="home-circle" />
          {[[50,87],[25,69],[42,73],[59,73],[77,67],[35,51],[61,48],[20,31],[50,25],[80,33],[61,16]].map(([x,y], index) => <span key={`${x}-${y}`} className={index === 8 ? 'focus' : ''} style={{ left: `${x}%`, top: `${y}%` }}>{index + 1}</span>)}
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d="M 61 48 Q 70 34 80 33" /><path d="M 80 33 Q 73 16 61 16" /></svg>
        </div>
        <div className="home-readout"><span><small>공격 지역 진입</small><b>4 <i>vs</i> 16</b></span><span><small>내 전술 변화</small><b className="positive">+8</b></span></div>
      </section>

      <section className="home-path" aria-label="서비스 이용 흐름">
        <article><span>01</span><div><small>진단</small><strong>실제 데이터로 문제를 읽고</strong></div></article>
        <article><span>02</span><div><small>결정</small><strong>감독처럼 직접 개입하고</strong></div></article>
        <article><span>03</span><div><small>비교</small><strong>기준 전술과 변화를 비교합니다</strong></div></article>
      </section>
    </main>
  )
}

function IntroScreen({ scenario, scenarioId, onScenario, onStart, onStudio }: { scenario: GuidedScenario; scenarioId: GuidedScenarioId; onScenario: (id: GuidedScenarioId) => void; onStart: () => void; onStudio: () => void }) {
  const missionTitle = scenario.id === 'korea-south-africa-64' ? '경기를 되돌려라.' : scenario.missionType === '득점 필요' ? '결승골을 만들어라.' : '리드를 지켜라.'
  return (
    <main className="intro-screen">
      <div className="stadium-glow" />
      <section className="mission-picker-panel">
        <div><p className="eyebrow">경기 선택</p><h2>어떤 순간에 개입하시겠습니까?</h2></div>
        <div className="scenario-picker" role="group" aria-label="실제 경기 미션 선택">
          {Object.values(guidedScenarios).map((item) => <button type="button" key={item.id} className={scenarioId === item.id ? 'active' : ''} onClick={() => onScenario(item.id)} aria-pressed={scenarioId === item.id}><span>{item.order}</span><CountryFlag code={item.ours.short} label={item.ours.name} /><div><small>{item.tournament}</small><strong>{item.ours.name}–{item.opponent.name}</strong><em>{item.minute}′ · {item.missionType}</em></div><b>{item.difficulty}</b></button>)}
        </div>
      </section>
      <section className="intro-copy">
        <p className="eyebrow">{scenario.intro.eyebrow}</p>
        <div className="mission-heading"><span>{scenario.intro.title}</span><h1><strong>{scenario.intro.accent}</strong><b>{missionTitle}</b></h1></div>
        <p className="intro-lead">{scenario.intro.lead}</p>
        <div className="objective-card">
          <span className="objective-icon">◎</span>
          <div>
            <small>경기 목표</small>
            <strong>{scenario.objective}</strong>
          </div>
        </div>
        <div className="intro-actions">
          <button className="primary-button large" type="button" onClick={onStart}>데이터 브리핑 시작 <span>→</span></button>
          <button className="secondary-button large" type="button" onClick={onStudio}>빈 전술판에서 시작</button>
        </div>
        <p className="no-login"><b>공개 원본의 범위와 분석 가정을 화면에 구분합니다.</b> 로그인 없이 약 3분 · 마우스와 터치 지원</p>
      </section>
      <section className="intro-scoreboard" aria-label="경기 상황">
        <div className="time-ring"><strong>{scenario.minute}</strong><span>분</span></div>
        <div className="teams-row">
          <div><span className="flag-orb korea"><CountryFlag code={scenario.ours.short} label={scenario.ours.name} /></span><strong>{scenario.ours.name}</strong><small>{scenario.ours.status}</small></div>
          <b>{scenario.score[0]}</b><i>:</i><b>{scenario.score[1]}</b>
          <div><span className="flag-orb portugal"><CountryFlag code={scenario.opponent.short} label={scenario.opponent.name} /></span><strong>{scenario.opponent.name}</strong><small>{scenario.opponent.status}</small></div>
        </div>
        <div className="timeline-mini">
          {scenario.id === 'korea-portugal-65'
            ? <><span style={{ left: '5%' }}>5′ <b>0–1</b></span><span style={{ left: '31%' }}>27′ <b>1–1</b></span><em style={{ left: '72%' }}>현재 시점</em></>
            : scenario.id === 'korea-south-africa-64'
              ? <><span style={{ left: '69%' }}>63′ <b>0–1</b></span><em style={{ left: '72%' }}>재구성 시점</em></>
              : <><span style={{ left: '30%' }}>35′ <b>1–0</b></span><span style={{ left: '67%' }}>73′ <b>2–0</b></span><span style={{ left: '78%' }}>82′ <b>2–1</b></span><em style={{ left: '88%' }}>현재 시점</em></>}
        </div>
      </section>
    </main>
  )
}

function BriefingScreen({ scenario, onBack, onNext }: { scenario: GuidedScenario; onBack: () => void; onNext: () => void }) {
  const [evidenceOpen, setEvidenceOpen] = useState(() => window.matchMedia('(min-width: 721px)').matches)
  const evidenceCount = scenario.officialReport ? 2 : 3

  return (
    <main className="briefing-screen page-wrap">
      <div className="page-heading">
        <div>
          <p className="eyebrow">경기 개입 브리핑 · {scenario.minute}′</p>
          <h1>{scenario.briefing.title}</h1>
          <p>{scenario.briefing.description}</p>
        </div>
        <span className="live-pill"><i /> 검증된 경기 데이터</span>
      </div>

      <section className="decision-brief" aria-label="경기 개입 핵심 요약">
        <span><small>관측 구간</small><b>{scenario.windowLabel}</b><em>{scenario.briefing.diagnosisTitle}</em></span>
        <span><small>필요한 결과</small><b>{scenario.briefing.contextNumber}</b><em>{scenario.briefing.contextLabel}</em></span>
        <span><small>우선 검토</small><b>{scenario.briefing.optionPlayer}</b><em>{scenario.briefing.optionRole}</em></span>
      </section>

      <section className="briefing-grid">
        <MatchStats scenario={scenario} />

        <article className="analysis-card">
          <div className="card-heading"><span>READ</span><div><small>경기 상황</small><h2>우리에게 필요한 결과</h2></div></div>
          <div className="context-number"><strong>{scenario.briefing.contextNumber}</strong><span>{scenario.briefing.contextLabel}</span></div>
          <ul className="signal-list">
            <li><span className="signal good">↗</span><div><strong>{scenario.briefing.successTitle}</strong><small>{scenario.briefing.successDetail}</small></div></li>
            <li><span className="signal warn">—</span><div><strong>{scenario.briefing.failureTitle}</strong><small>{scenario.briefing.failureDetail}</small></div></li>
          </ul>
        </article>

        <article className="analysis-card">
          <div className="card-heading"><span>ACT</span><div><small>개입 선택지</small><h2>{scenario.briefing.optionTitle}</h2></div></div>
          <div className="player-spotlight">
            <div className="shirt-number">{scenario.briefing.optionNumber}</div>
            <div><strong>{scenario.briefing.optionPlayer}</strong><small>{scenario.briefing.optionPosition} · {scenario.briefing.optionRole}</small></div>
            <b>{scenario.briefing.optionAvailability}</b>
          </div>
          <div className="trait-row">{scenario.briefing.optionTraits.map((trait) => <span key={trait}>{trait}</span>)}</div>
          <p className="coach-quote accent">“{scenario.briefing.optionQuote}”</p>
        </article>
      </section>

      <details className="evidence-stack" open={evidenceOpen} onToggle={(event) => setEvidenceOpen(event.currentTarget.open)}>
        <summary><span><small>상세 분석 근거</small><strong>{scenario.officialReport ? '공식 보고서와 패스 구조' : '공간·패스·볼 흐름'}</strong></span><b>{evidenceCount}개 보기 <i aria-hidden="true">⌄</i></b></summary>
        <div className="evidence-stack-content">
          {scenario.officialReport
            ? <><OfficialReportEvidence scenario={scenario} /><PassNetwork scenario={scenario} /></>
            : <><SpatialEvidence scenario={scenario} /><PassNetwork scenario={scenario} /><BallFlow scenario={scenario} /></>}
        </div>
      </details>

      <div className="source-strip">
        {scenario.officialReport ? <b className="source-wordmark">FIFA<br />TRAINING CENTRE</b> : <img src="/statsbomb-logo.png" alt="StatsBomb" width="5885" height="943" loading="lazy" />}
        <p><strong>데이터 근거</strong> {scenario.sourceNote ?? `Match ${scenario.matchId} · ${scenario.windowLabel} 이벤트 직접 집계 · 추출일 ${scenario.extractedAt}`}</p>
        <a href={scenario.sourceUrl} target="_blank" rel="noreferrer">원본 자료 ↗</a>
      </div>

      <div className="page-actions">
        <button className="text-button" type="button" onClick={onBack}>← 이전</button>
        <button className="primary-button" type="button" onClick={onNext}>전술 보드로 이동 <span>→</span></button>
      </div>
    </main>
  )
}

interface TacticsScreenProps {
  scenario: GuidedScenario
  formation: FormationKey
  squad: Player[]
  selectedPlayer: Player | null
  draggingId: string | null
  substitutions: GuidedSubstitution[]
  tactics: Tactics
  metrics: Metrics
  pitchRef: React.RefObject<HTMLDivElement>
  onBack: () => void
  onFormation: (formation: FormationKey) => void
  onTactic: (key: keyof Tactics, value: number) => void
  onTacticPreset: (values: Tactics) => void
  onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>, id: string) => void
  onPointerMove: (event: ReactPointerEvent<HTMLButtonElement>, id: string) => void
  onPointerUp: (event: ReactPointerEvent<HTMLButtonElement>) => void
  onKeyboardMove: (event: ReactKeyboardEvent<HTMLButtonElement>, id: string) => void
  onSelect: (id: string) => void
  onRole: (role: string) => void
  onSubstitute: (id: string) => void
  onUndoSubstitution: () => void
  onResetSubstitutions: () => void
  onSubmit: () => void
}

function TacticsScreen(props: TacticsScreenProps) {
  const { scenario, formation, squad, selectedPlayer, draggingId, substitutions, tactics, metrics, pitchRef } = props
  const onPitch = squad.filter((player) => player.onPitch)
  const bench = squad.filter((player) => !player.onPitch)
  const selectedEvidence = selectedPlayer ? scenario.playerEvidence[selectedPlayer.id] : null
  const substitutionsRemaining = 5 - substitutions.length
  const substitutedOffIds = new Set(substitutions.map((item) => item.outgoingId))
  const activePreset = tacticPresets.find((preset) => (Object.keys(tactics) as Array<keyof Tactics>).every((key) => tactics[key] === preset.values[key]))?.id

  return (
    <main className="tactics-screen page-wrap wide">
      <div className="tactics-heading">
        <div><p className="eyebrow">경기 개입 전술판 · {scenario.minute}′</p><h1>{scenario.objective}</h1></div>
        <div className="decision-clock"><span>분석 기준</span><strong>{scenario.minute}′</strong><small>{scenario.windowLabel} 관측 + 시나리오 비교</small></div>
      </div>

      <div className="guided-instructions" aria-label="전술 설계 순서">
        <span><b>1</b><div><strong>포메이션 선택</strong><small>기본 대형을 정합니다</small></div></span>
        <span><b>2</b><div><strong>선수 배치·교체</strong><small>선수를 끌거나 벤치를 누릅니다</small></div></span>
        <span><b>3</b><div><strong>팀 지시 조정</strong><small>압박·폭·템포를 바꿉니다</small></div></span>
        <span><b>4</b><div><strong>전술안 분석</strong><small>이점과 리스크를 확인합니다</small></div></span>
      </div>

      <div className="tactics-layout">
        <aside className="control-panel panel">
          <div className="formation-control-block">
            <div className="panel-title"><span>01</span><div><small>팀 대형</small><h2>포메이션</h2></div></div>
            <label className="formation-select">
              <span>비소유 기본 대형</span>
              <select name="guided-formation" aria-label="포메이션 선택" value={formation} onChange={(event) => props.onFormation(event.target.value as FormationKey)}>
                {formationGroups.map((group) => <optgroup key={group.label} label={`${group.label} 포메이션`}>{group.items.map((item) => <option value={item} key={item}>{item} · {formationGuidance[item]}</option>)}</optgroup>)}
              </select>
            </label>
            <div className="formation-summary">
              <FormationGlyph formation={formation} />
              <strong>{formation}</strong>
              <span>{formationGuidance[formation]}</span>
              <i>{formation === scenario.defaultFormation ? '실제 기준 대형' : '변경 전술 대형'} · {formations.length}개 선택지</i>
            </div>
            <p className="helper">포메이션은 비소유 기본 위치입니다. 역할과 직접 배치로 볼 소유 시 움직임을 설계합니다.</p>
          </div>

          <div className="section-divider" />
          <div className="tactics-control-block">
            <div className="panel-title"><span>02</span><div><small>팀 전술 지시</small><h2>팀 지시</h2></div></div>
            <div className="tactic-presets" role="group" aria-label="전술 프리셋">
              {tacticPresets.map((preset) => <button type="button" key={preset.id} className={activePreset === preset.id ? 'active' : ''} aria-pressed={activePreset === preset.id} onClick={() => props.onTacticPreset(preset.values)}><strong>{preset.label}</strong><small>{preset.detail}</small></button>)}
            </div>
            <Slider label="압박 강도" low="기다리기" high="즉시 압박" value={tactics.pressing} onChange={(value) => props.onTactic('pressing', value)} />
            <Slider label="공격 폭" low="좁게" high="넓게" value={tactics.width} onChange={(value) => props.onTactic('width', value)} />
            <Slider label="공격 템포" low="차분하게" high="빠르게" value={tactics.tempo} onChange={(value) => props.onTactic('tempo', value)} />
            <Slider label="위험 감수" low="안전하게" high="과감하게" value={tactics.risk} onChange={(value) => props.onTactic('risk', value)} />
          </div>
        </aside>

        <section className="board-column">
          <p className="sr-only" id="guided-pitch-instructions">선수를 선택하고 방향키로 2퍼센트씩, Shift와 방향키로 5퍼센트씩 이동할 수 있습니다.</p>
          <div className="pitch" ref={pitchRef} role="group" aria-label="선수 위치 조정 전술 보드" aria-describedby="guided-pitch-instructions">
            <div className="pitch-lines"><i className="halfway" /><i className="circle" /><i className="box top" /><i className="box bottom" /></div>
            <div className="attack-label">↑ ATTACK</div>
            {onPitch.map((player) => (
              <button
                className={`player-token ${selectedPlayer?.id === player.id ? 'selected' : ''} ${draggingId === player.id ? 'dragging' : ''}`}
                type="button"
                key={player.id}
                style={{ left: `${player.x}%`, top: `${player.y}%` }}
                onPointerDown={(event) => props.onPointerDown(event, player.id)}
                onPointerMove={(event) => props.onPointerMove(event, player.id)}
                onPointerUp={props.onPointerUp}
                onPointerCancel={props.onPointerUp}
                onKeyDown={(event) => props.onKeyboardMove(event, player.id)}
                aria-label={`${player.name}, ${player.role}, 가로 위치 ${Math.round(player.x)}, 세로 위치 ${Math.round(player.y)}`}
              >
                <span>{player.number}</span><strong>{player.shortName}</strong>
              </button>
            ))}
          </div>

          <div className="bench panel">
            <div className="bench-heading">
              <div><small>교체 계획</small><strong>{selectedPlayer?.onPitch ? `${selectedPlayer.shortName} 대신 투입할 선수를 선택하세요` : '먼저 필드 선수를 선택하세요'}</strong></div>
              <div className="substitution-actions"><span>{substitutions.length} / 5명 · 1회 교체 창</span><button type="button" disabled={substitutions.length === 0} onClick={props.onUndoSubstitution}>↶ 직전 취소</button><button type="button" disabled={substitutions.length === 0} onClick={props.onResetSubstitutions}>전체 취소</button></div>
            </div>
            {substitutions.length > 0 && <ol className="substitution-log" aria-label="계획한 교체 순서">{substitutions.map((item, index) => <li key={item.id}><b>{index + 1}</b><span><s>{item.outgoingName}</s><i aria-hidden="true">→</i><strong>{item.incomingName}</strong></span></li>)}</ol>}
            <div className="bench-list">
              {bench.map((player) => {
                const cannotReturn = substitutedOffIds.has(player.id)
                const canEnter = Boolean(selectedPlayer?.onPitch) && substitutionsRemaining > 0 && !cannotReturn
                return (
                <button type="button" key={player.id} disabled={cannotReturn} onClick={() => canEnter ? props.onSubstitute(player.id) : props.onSelect(player.id)}>
                  <span>{player.number}</span><div><strong>{player.shortName}</strong><small>{player.position} · {player.role}</small></div>
                  {cannotReturn ? <em className="unavailable">재투입 불가</em> : canEnter ? <em>투입</em> : null}
                </button>
              )})}
            </div>
            <p className="substitution-rule">최대 5명까지 한 번의 교체 창에 묶어 설계합니다. 교체 아웃된 선수는 재투입할 수 없으며, 비교 실행 전에는 언제든 직전 교체를 취소할 수 있습니다.</p>
          </div>
        </section>

        <aside className="insight-column">
          <section className="panel selected-player">
            <div className="panel-title"><span>03</span><div><small>선수 역할</small><h2>개인 역할</h2></div></div>
            {selectedPlayer ? (
              <>
                <div className="selected-summary"><b>{selectedPlayer.number}</b><div><strong>{selectedPlayer.name}</strong><small>{selectedPlayer.position} · {selectedPlayer.onPitch ? 'ON PITCH' : 'BENCH'}</small></div></div>
                {selectedEvidence ? (
                  <div className="attribute-row evidence"><span>패스 <b>{selectedEvidence.passesCompleted}/{selectedEvidence.passesAttempted}</b></span><span>압박 <b>{selectedEvidence.pressures}</b></span><span>슈팅 <b>{selectedEvidence.shots}</b></span></div>
                ) : <p className="no-evidence">65분 이전 미출전 · 경기 내 관측값 없음</p>}
                {selectedEvidence?.note && <p className="evidence-note">{selectedEvidence.note}</p>}
                <label className="role-select">역할<select name="guided-player-role" value={selectedPlayer.role} onChange={(event) => props.onRole(event.target.value)} disabled={!selectedPlayer.onPitch}>{roleOptions[selectedPlayer.position].map((role) => <option key={role}>{role}</option>)}</select></label>
              </>
            ) : <p>선수를 선택하세요.</p>}
          </section>

          <section className="panel metrics-panel">
            <div className="panel-title"><span>04</span><div><small>전술 변화 비교</small><h2>전술 리스크 검토</h2></div></div>
            <Metric label={scenario.metricLabels[0]} value={metrics.threat} good />
            <Metric label={scenario.metricLabels[1]} value={metrics.control} good />
            <Metric label={scenario.metricLabels[2]} value={metrics.exposure} />
            <Metric label={scenario.metricLabels[3]} value={metrics.fatigue} />
            <div className="projection-note"><span>✦</span><p><strong>분석 코멘트</strong>{getCoachNote(metrics, squad, scenario)}</p></div>
            <small className="estimate-label">* {scenario.windowLabel} 관측값에 대형·배치 높이·팀 폭·역할·교체·팀 지시의 설명 가능한 규칙을 적용합니다. {evidenceMethod.caution}</small>
          </section>

          <button className="primary-button submit-tactic" type="button" onClick={props.onSubmit}>전술 변화 비교 실행 <span>→</span></button>
          <button className="text-button center" type="button" onClick={props.onBack}>브리핑 다시 보기</button>
        </aside>
      </div>
    </main>
  )
}

function FormationGlyph({ formation }: { formation: FormationKey }) {
  return (
    <span className="formation-glyph" aria-hidden="true">
      {formationPositions[formation].map((position, index) => <i key={index} style={{ left: `${position.x}%`, top: `${position.y}%` }} />)}
    </span>
  )
}

function Slider({ label, low, high, value, onChange }: { label: string; low: string; high: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="tactic-slider">
      <span><strong>{label}</strong><b>{value}</b></span>
      <input name={`guided-tactic-${label}`} type="range" min="0" max="100" value={value} onChange={(event) => onChange(Number(event.target.value))} style={{ '--range': `${value}%` } as React.CSSProperties} />
      <small><i>{low}</i><i>{high}</i></small>
    </label>
  )
}

function Metric({ label, value, good = false }: { label: string; value: number; good?: boolean }) {
  const displayValue = value >= 67 ? '높음' : value >= 42 ? '보통' : '낮음'
  return (
    <div className={`metric ${good ? 'good' : 'risk'}`}>
      <span><strong>{label}</strong><b>{displayValue}</b></span>
      <div><i style={{ width: `${value}%` }} /></div>
      <small>{value}/100</small>
    </div>
  )
}

function getCoachNote(metrics: Metrics, squad: Player[], scenario: GuidedScenario) {
  const impactOn = squad.some((player) => player.id === scenario.impactPlayerId && player.onPitch)
  if (scenario.id === 'argentina-netherlands-83') {
    if (metrics.exposure > 65) return '박스 노출이 높습니다. 최종 라인만 내리지 말고 크로스 시작점을 먼저 압박해야 합니다.'
    if (impactOn && metrics.control > 58) return '몬티엘이 측면을 닫고 있습니다. 메시나 라우타로 한 명은 역습 출구로 남겨두세요.'
    if (metrics.threat < 45) return '모든 선수가 내려오면 세컨드볼을 따내도 전진할 수 없습니다. 전방 출구를 한 명 유지하세요.'
    return '블록 간격은 안정적입니다. 박스 앞 세컨드볼 담당과 크로스 압박 선수를 명확히 지정하세요.'
  }
  if (metrics.exposure > 65) return '공격 숫자는 충분하지만 공을 잃은 직후 중앙 공간이 위험합니다.'
  if (impactOn && metrics.threat > 60) return '황희찬의 속도가 손흥민의 전진 패스와 연결될 가능성이 높습니다.'
  if (metrics.threat < 52) return '탈락을 피하려면 더 빠른 템포나 뒷공간을 노릴 선수가 필요합니다.'
  return '균형은 안정적입니다. 이제 승부를 바꿀 한 가지 과감한 선택이 필요합니다.'
}

function ResultScreen({ scenario, metrics, squad, tactics, formation, substitutions, onRetry }: { scenario: GuidedScenario; metrics: Metrics; squad: Player[]; tactics: Tactics; formation: FormationKey; substitutions: GuidedSubstitution[]; onRetry: () => void }) {
  const impactOn = squad.some((player) => player.id === scenario.impactPlayerId && player.onPitch)
  const baselineMetrics = calculateMetrics(cloneScenarioSquad(scenario), scenario.defaultTactics, scenario.defaultFormation, scenario)
  const performance = Math.round(scenario.id === 'argentina-netherlands-83'
    ? metrics.threat * .15 + metrics.control * .3 + (100 - metrics.exposure) * .4 + (100 - metrics.fatigue) * .15 + (impactOn ? 5 : 0)
    : metrics.threat * .32 + metrics.control * .32 + (100 - metrics.exposure) * .25 + (100 - metrics.fatigue) * .11 + (impactOn ? 6 : 0))
  const recommendation = performance >= 70 ? '채택 권고' : performance < 50 ? '재설계 필요' : '조건부 채택'
  const tone = performance >= 70 ? 'win' : performance < 50 ? 'lose' : 'draw'
  const planLabel = impactOn ? scenario.result.planOn : tactics.pressing > 68 ? '고강도 압박 유지안' : scenario.result.planOff
  const operatingCondition = metrics.exposure > 60
    ? scenario.result.operatingRisk
    : scenario.result.operatingSafe
  const impactPlayer = squad.find((player) => player.id === scenario.impactPlayerId)?.shortName ?? scenario.briefing.optionPlayer
  const whatIfMetrics = ([
    ['threat', scenario.metricLabels[0], true],
    ['control', scenario.metricLabels[1], true],
    ['exposure', scenario.metricLabels[2], false],
    ['fatigue', scenario.metricLabels[3], false],
  ] as const).map(([key, label, higherIsBetter]) => {
    const before = baselineMetrics[key]
    const after = metrics[key]
    const delta = after - before
    return { key, label, before, after, delta, improved: higherIsBetter ? delta > 0 : delta < 0 }
  })

  return (
    <main className="result-screen page-wrap">
      <div className={`result-hero ${tone}`}>
        <p className="eyebrow">전술 변화 비교 · 경기 결과 예측이 아닌 코칭 검토</p>
        <div className="decision-status"><span>제안 전술</span><strong>{formation}</strong><em>{recommendation}</em></div>
        <h1>{planLabel}</h1>
        <p>{scenario.result.baseline}을 기준선으로 사용자의 전술안을 비교한 코칭 검토 결과입니다.</p>
      </div>

      <section className="what-if-panel panel" aria-label="실제 기준 전술과 사용자 전술 변화 비교">
        <header><div><small>기준 전술 → 내 전술</small><h2>내 개입으로 달라진 전술 상태</h2></div><span>{scenario.defaultFormation} 기준</span></header>
        <div>
          {whatIfMetrics.map((item) => (
            <article key={item.key} className={item.delta === 0 ? 'neutral' : item.improved ? 'improved' : 'declined'}>
              <small>{item.label}</small>
              <p><span>{item.before}</span><i>→</i><strong>{item.after}</strong></p>
              <b>{item.delta === 0 ? '변화 없음' : `${item.delta > 0 ? '+' : ''}${item.delta}점`}</b>
            </article>
          ))}
        </div>
        <p>실제 관측 구간을 기준선으로 대형, 선수 배치 높이와 폭, 역할, 교체, 팀 지시만 바꿔 다시 계산했습니다. 경기 결과나 득점 확률을 예측하지 않습니다.</p>
      </section>

      {scenario.id === 'korea-portugal-65' ? <TacticalSequence hwangOn={impactOn} formation={formation} tactics={tactics} /> : <section className="result-evidence-summary panel" aria-label="브리핑 패스 구조 근거 요약">
        <div><small>브리핑 근거</small><h2>패스 구조는 판단 근거로만 다시 확인합니다</h2><p>{scenario.networkCopy.title} 결과 화면에서는 같은 네트워크를 반복하지 않고 전술 변화와 운영 조건에 집중합니다.</p></div>
        <dl>
          <div><dt>{scenario.ours.name}</dt><dd>{scenario.networks.ours.completedPasses}회 완료</dd></div>
          <div><dt>{scenario.opponent.name}</dt><dd>{scenario.networks.opponent.completedPasses}회 완료</dd></div>
          <div><dt>위치 기준</dt><dd>{scenario.networkPositionBasis ? '참조 배치' : '평균 위치'}</dd></div>
        </dl>
      </section>}

      <section className="result-grid">
        <article className="report-card">
          <div className="panel-title"><span>MEMO</span><div><small>의사결정 메모</small><h2>이점과 리스크를 검토합니다</h2></div></div>
          <ul className="report-list">
            <li className={impactOn ? 'positive' : 'neutral'}><b>{impactOn ? '✓' : '!'}</b><div><strong>{scenario.id === 'korea-portugal-65' ? '전진 수단' : scenario.id === 'korea-south-africa-64' ? '후방 균형' : '측면 대응'}</strong><span>{impactOn ? scenario.result.impactOn : scenario.result.impactOff}</span></div></li>
            <li className={substitutions.length > 0 ? 'positive' : 'neutral'}><b>{substitutions.length > 0 ? '↗' : '—'}</b><div><strong>교체 계획 · {substitutions.length}/5명</strong><span>{substitutions.length > 0 ? substitutions.map((item) => `${item.outgoingName} → ${item.incomingName}`).join(' · ') : '선수 교체 없이 배치와 팀 지시만 변경했습니다.'}</span></div></li>
            <li className={metrics.exposure < 60 ? 'positive' : 'negative'}><b>{metrics.exposure < 60 ? '✓' : '!'}</b><div><strong>전환 수비</strong><span>{metrics.exposure < 60 ? '공격적 개입 속에서도 후방 숫자를 관리할 수 있는 범위입니다.' : '압박과 위험 감수가 함께 높아 공을 잃은 뒤 중앙 보호 조건이 필요합니다.'}</span></div></li>
            <li className={metrics.control > 50 ? 'positive' : 'neutral'}><b>{metrics.control > 50 ? '✓' : '!'}</b><div><strong>{scenario.metricLabels[1]}</strong><span>{scenario.windowLabel} 패스 성공률 {scenario.evidence.ours.passCompletion}%를 기준으로 한 비교 점수는 {metrics.control}점입니다.</span></div></li>
          </ul>
          <div className="actual-choice"><span>실제 경기와 비교</span><p>{scenario.result.actualChoice} {scenario.result.actualOutcome} 이 사실은 사후 비교 정보이며 시나리오 점수 계산에는 정답값으로 사용하지 않습니다.</p></div>
        </article>

        <article className="manager-card" aria-label="공유용 전술 카드">
          <div className="card-top"><span>RE:TACTIC</span><small>공유용 전술 카드 · {scenario.minute}′</small></div>
          <div className="manager-badge">R:</div>
          <p>제안 전술</p>
          <h2>{planLabel}</h2>
          <div className="manager-traits"><span>{formation}</span><span>위험 {tactics.risk}</span><span>템포 {tactics.tempo}</span><span>{substitutions.length > 0 ? `교체 ${substitutions.length}명` : impactOn ? `${impactPlayer} 투입` : '기존 인원 유지'}</span></div>
          <small>운영 조건 · {operatingCondition}</small>
        </article>
      </section>

      <div className="result-actions">
        <button className="secondary-button" type="button" onClick={() => window.print()}>전술안 인쇄·저장</button>
        <button className="primary-button" type="button" onClick={onRetry}>전술안 다시 설계 <span>↻</span></button>
      </div>
      <footer className="data-source">
        {scenario.officialReport ? <b className="source-wordmark">FIFA<br />TRAINING CENTRE</b> : <img src="/statsbomb-logo.png" alt="StatsBomb" width="5885" height="943" loading="lazy" />}
        <p>{scenario.officialReport
          ? '데이터: FIFA Training Centre 전체 경기 보고서. 64분 개입안은 RE:TACTIC의 사후 전술 재구성이며 실제 경기 예측이나 FIFA의 공식 권고가 아닙니다.'
          : `경기 이벤트 데이터: StatsBomb Open Data · Match ${scenario.matchId}. 파생 지표는 RE:TACTIC이 계산했습니다. 시나리오 점수는 실제 경기 결과 예측이나 승률이 아닙니다.`}</p>
      </footer>
    </main>
  )
}

export default App
