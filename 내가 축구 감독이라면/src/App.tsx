import { useMemo, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import UniversalStudio from './components/UniversalStudio'
import SpatialEvidence from './components/SpatialEvidence'
import TacticalSequence from './components/TacticalSequence'
import PassNetwork from './components/PassNetwork'
import BallFlow from './components/BallFlow'
import MatchStats from './components/MatchStats'
import { evidenceMethod } from './data/evidence'
import { formationPositions, roleOptions } from './data/match'
import { cloneScenarioSquad, defaultScenarioId, guidedScenarios, type GuidedScenario, type GuidedScenarioId } from './data/scenarios'
import type { FormationKey, Metrics, Player, Stage, Tactics } from './types'

const formations: FormationKey[] = ['4-3-3', '4-2-3-1', '4-4-2', '4-1-4-1', '4-3-1-2', '3-4-3', '5-3-2']
const formationGuidance: Record<FormationKey, string> = {
  '4-3-3': '측면 폭과 전방 압박의 균형',
  '4-2-3-1': '중앙 보호와 2선 연결 강화',
  '4-4-2': '두 줄 수비와 빠른 전환',
  '4-1-4-1': '중앙 간격을 좁히는 안정형',
  '4-3-1-2': '중앙 수적 우위와 투톱 침투',
  '3-4-3': '높은 폭과 공격 숫자 확보',
  '5-3-2': '박스 보호와 역습 출구 유지',
}
const stageOrder: Stage[] = ['intro', 'briefing', 'tactics', 'result']
const stageLabels: Record<Stage, string> = { intro: '시작', briefing: '진단', tactics: '설계', result: '검토' }
type AppMode = 'guided' | 'studio'

const getInitialMode = (): AppMode => window.location.hash === '#studio' ? 'studio' : 'guided'

const getInitialStage = (): Stage => {
  const hashStage = window.location.hash.replace('#', '') as Stage
  return stageOrder.includes(hashStage) ? hashStage : 'intro'
}

const clamp = (value: number, min = 0, max = 100) => Math.min(max, Math.max(min, value))

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
  const observedPressureLoad = scenario.evidence.ours.pressures
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
  const [substitutionUsed, setSubstitutionUsed] = useState(false)
  const [tactics, setTactics] = useState<Tactics>({ ...scenario.defaultTactics })
  const pitchRef = useRef<HTMLDivElement>(null)

  const selectedPlayer = squad.find((player) => player.id === selectedId) ?? null
  const metrics = useMemo(() => calculateMetrics(squad, tactics, formation, scenario), [squad, tactics, formation, scenario])

  const selectScenario = (nextId: GuidedScenarioId) => {
    const next = guidedScenarios[nextId]
    setScenarioId(nextId)
    setStage('intro')
    setFormation(next.defaultFormation)
    setSquad(cloneScenarioSquad(next))
    setSelectedId(next.selectedPlayerId)
    setDraggingId(null)
    setSubstitutionUsed(false)
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

  const changeRole = (role: string) => {
    if (!selectedPlayer) return
    setSquad((current) => current.map((player) => player.id === selectedPlayer.id ? { ...player, role } : player))
  }

  const substitute = (incomingId: string) => {
    if (!selectedPlayer?.onPitch || selectedPlayer.slot === null || substitutionUsed) return
    const outgoingSlot = selectedPlayer.slot
    const outgoingCoordinate = { x: selectedPlayer.x, y: selectedPlayer.y }
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
    setSubstitutionUsed(true)
  }

  const resetGame = () => {
    setMode('guided')
    setStage('intro')
    setFormation(scenario.defaultFormation)
    setSquad(cloneScenarioSquad(scenario))
    setSelectedId(scenario.selectedPlayerId)
    setSubstitutionUsed(false)
    setTactics({ ...scenario.defaultTactics })
    window.history.replaceState(null, '', '#intro')
  }

  const switchMode = (nextMode: AppMode) => {
    setMode(nextMode)
    window.history.replaceState(null, '', nextMode === 'studio' ? '#studio' : `#${stage}`)
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" type="button" onClick={resetGame} aria-label="처음으로">
          <span className="brand-mark">R:</span>
          <span>RE:TACTIC</span>
        </button>
        <div className="topbar-center">
          <nav className="mode-switch" aria-label="서비스 모드">
            <button type="button" className={mode === 'guided' ? 'active' : ''} onClick={() => switchMode('guided')}><span className="mode-long">실제 경기 분석</span><span className="mode-short">경기 분석</span></button>
            <button type="button" className={mode === 'studio' ? 'active' : ''} onClick={() => switchMode('studio')}><span className="mode-long">범용 전술 스튜디오</span><span className="mode-short">전술판</span></button>
          </nav>
          {mode === 'guided' && <div className="match-chip">
            <span>{scenario.ours.flag} {scenario.ours.short}</span><strong>{scenario.score[0]} : {scenario.score[1]}</strong><span>{scenario.opponent.short} {scenario.opponent.flag}</span><em>{scenario.minute}′</em>
          </div>}
        </div>
        {mode === 'guided' ? <nav className="stage-nav" aria-label="진행 단계">
          {stageOrder.map((item, index) => {
            const currentIndex = stageOrder.indexOf(stage)
            return <button type="button" key={item} className={currentIndex >= index ? 'active' : ''} disabled={index > currentIndex} onClick={() => setStage(item)}><i>{index + 1}</i><b>{stageLabels[item]}</b></button>
          })}
        </nav> : <span className="studio-top-status"><i /> 작업 내용 자동 저장</span>}
      </header>
      <nav className="mobile-mode-nav" aria-label="모바일 서비스 모드">
        <button type="button" className={mode === 'guided' ? 'active' : ''} onClick={() => switchMode('guided')}>실제 경기 분석</button>
        <button type="button" className={mode === 'studio' ? 'active' : ''} onClick={() => switchMode('studio')}>범용 전술판</button>
      </nav>

      {mode === 'studio' ? <UniversalStudio /> : <>
      {stage === 'intro' && <IntroScreen scenario={scenario} scenarioId={scenarioId} onScenario={selectScenario} onStart={() => setStage('briefing')} onStudio={() => switchMode('studio')} />}
      {stage === 'briefing' && <BriefingScreen scenario={scenario} onBack={() => setStage('intro')} onNext={() => setStage('tactics')} />}
      {stage === 'tactics' && (
        <TacticsScreen
          scenario={scenario}
          formation={formation}
          squad={squad}
          selectedPlayer={selectedPlayer}
          draggingId={draggingId}
          substitutionUsed={substitutionUsed}
          tactics={tactics}
          metrics={metrics}
          pitchRef={pitchRef}
          onBack={() => setStage('briefing')}
          onFormation={setFormationPreset}
          onTactic={updateTactic}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onSelect={setSelectedId}
          onRole={changeRole}
          onSubstitute={substitute}
          onSubmit={() => setStage('result')}
        />
      )}
      {stage === 'result' && <ResultScreen scenario={scenario} metrics={metrics} squad={squad} tactics={tactics} formation={formation} onRetry={() => setStage('tactics')} />}
      </>}
    </div>
  )
}

function IntroScreen({ scenario, scenarioId, onScenario, onStart, onStudio }: { scenario: GuidedScenario; scenarioId: GuidedScenarioId; onScenario: (id: GuidedScenarioId) => void; onStart: () => void; onStudio: () => void }) {
  return (
    <main className="intro-screen">
      <div className="stadium-glow" />
      <section className="intro-copy">
        <p className="eyebrow">{scenario.intro.eyebrow}</p>
        <h1>{scenario.intro.title} <span>{scenario.intro.accent}</span><br />{scenario.missionType === '득점 필요' ? '한 골이 필요합니다.' : '한 골을 지켜야 합니다.'}</h1>
        <p className="intro-lead">{scenario.intro.lead.split('\n').map((line) => <span key={line}>{line}<br /></span>)}</p>
        <div className="scenario-picker" role="group" aria-label="실제 경기 미션 선택">
          {Object.values(guidedScenarios).map((item) => <button type="button" key={item.id} className={scenarioId === item.id ? 'active' : ''} onClick={() => onScenario(item.id)} aria-pressed={scenarioId === item.id}><span>{item.order}</span><div><small>{item.tournament}</small><strong>{item.ours.name}–{item.opponent.name}</strong><em>{item.minute}′ · {item.missionType}</em></div><b>{item.difficulty}</b></button>)}
        </div>
        <div className="objective-card">
          <span className="objective-icon">◎</span>
          <div>
            <small>MATCH OBJECTIVE</small>
            <strong>{scenario.objective}</strong>
          </div>
        </div>
        <div className="intro-actions">
          <button className="primary-button large" type="button" onClick={onStart}>데이터 브리핑 시작 <span>→</span></button>
          <button className="secondary-button large" type="button" onClick={onStudio}>빈 전술판에서 시작</button>
        </div>
        <p className="no-login"><b>두 미션 모두 실제 StatsBomb 이벤트를 사용합니다.</b> 로그인 없이 약 3분 · 마우스와 터치 지원</p>
      </section>
      <section className="intro-scoreboard" aria-label="경기 상황">
        <div className="time-ring"><strong>{scenario.minute}</strong><span>MIN</span></div>
        <div className="teams-row">
          <div><span className="flag-orb korea">{scenario.ours.flag}</span><strong>{scenario.ours.name}</strong><small>{scenario.ours.status}</small></div>
          <b>{scenario.score[0]}</b><i>:</i><b>{scenario.score[1]}</b>
          <div><span className="flag-orb portugal">{scenario.opponent.flag}</span><strong>{scenario.opponent.name}</strong><small>{scenario.opponent.status}</small></div>
        </div>
        <div className="timeline-mini">
          {scenario.id === 'korea-portugal-65' ? <><span style={{ left: '5%' }}>5′ <b>0–1</b></span><span style={{ left: '31%' }}>27′ <b>1–1</b></span><em style={{ left: '72%' }}>YOU ARE HERE</em></> : <><span style={{ left: '30%' }}>35′ <b>1–0</b></span><span style={{ left: '67%' }}>73′ <b>2–0</b></span><span style={{ left: '78%' }}>82′ <b>2–1</b></span><em style={{ left: '88%' }}>YOU ARE HERE</em></>}
        </div>
      </section>
    </main>
  )
}

function BriefingScreen({ scenario, onBack, onNext }: { scenario: GuidedScenario; onBack: () => void; onNext: () => void }) {
  return (
    <main className="briefing-screen page-wrap">
      <div className="page-heading">
        <div>
          <p className="eyebrow">MATCH INTERVENTION BRIEF · {scenario.minute}′</p>
          <h1>{scenario.briefing.title}</h1>
          <p>{scenario.briefing.description}</p>
        </div>
        <span className="live-pill"><i /> VERIFIED MATCH DATA</span>
      </div>

      <section className="briefing-grid">
        <MatchStats scenario={scenario} />

        <article className="analysis-card">
          <div className="card-heading"><span>02</span><div><small>MATCH CONTEXT</small><h2>우리에게 필요한 결과</h2></div></div>
          <div className="context-number"><strong>{scenario.briefing.contextNumber}</strong><span>{scenario.briefing.contextLabel}</span></div>
          <ul className="signal-list">
            <li><span className="signal good">↗</span><div><strong>{scenario.briefing.successTitle}</strong><small>{scenario.briefing.successDetail}</small></div></li>
            <li><span className="signal warn">—</span><div><strong>{scenario.briefing.failureTitle}</strong><small>{scenario.briefing.failureDetail}</small></div></li>
          </ul>
        </article>

        <article className="analysis-card">
          <div className="card-heading"><span>03</span><div><small>INTERVENTION OPTION</small><h2>{scenario.briefing.optionTitle}</h2></div></div>
          <div className="player-spotlight">
            <div className="shirt-number">{scenario.briefing.optionNumber}</div>
            <div><strong>{scenario.briefing.optionPlayer}</strong><small>{scenario.briefing.optionPosition} · {scenario.briefing.optionRole}</small></div>
            <b>{scenario.briefing.optionAvailability}</b>
          </div>
          <div className="trait-row">{scenario.briefing.optionTraits.map((trait) => <span key={trait}>{trait}</span>)}</div>
          <p className="coach-quote accent">“{scenario.briefing.optionQuote}”</p>
        </article>
      </section>

      <SpatialEvidence scenario={scenario} />
      <PassNetwork scenario={scenario} />
      <BallFlow scenario={scenario} />

      <div className="source-strip">
        <img src="/statsbomb-logo.png" alt="StatsBomb" />
        <p><strong>데이터 근거</strong> Match {scenario.matchId} · {scenario.windowLabel} 이벤트 직접 집계 · 추출일 {scenario.extractedAt}</p>
        <a href={scenario.sourceUrl} target="_blank" rel="noreferrer">원본 저장소 ↗</a>
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
  substitutionUsed: boolean
  tactics: Tactics
  metrics: Metrics
  pitchRef: React.RefObject<HTMLDivElement>
  onBack: () => void
  onFormation: (formation: FormationKey) => void
  onTactic: (key: keyof Tactics, value: number) => void
  onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>, id: string) => void
  onPointerMove: (event: ReactPointerEvent<HTMLButtonElement>, id: string) => void
  onPointerUp: (event: ReactPointerEvent<HTMLButtonElement>) => void
  onSelect: (id: string) => void
  onRole: (role: string) => void
  onSubstitute: (id: string) => void
  onSubmit: () => void
}

function TacticsScreen(props: TacticsScreenProps) {
  const { scenario, formation, squad, selectedPlayer, draggingId, substitutionUsed, tactics, metrics, pitchRef } = props
  const onPitch = squad.filter((player) => player.onPitch)
  const bench = squad.filter((player) => !player.onPitch)
  const selectedEvidence = selectedPlayer ? scenario.playerEvidence[selectedPlayer.id] : null

  return (
    <main className="tactics-screen page-wrap wide">
      <div className="tactics-heading">
        <div><p className="eyebrow">MATCH INTERVENTION WORKSPACE · {scenario.minute}′</p><h1>{scenario.objective}</h1></div>
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
          <div className="panel-title"><span>01</span><div><small>TEAM SHAPE</small><h2>포메이션</h2></div></div>
          <label className="formation-select">
            <span>기본 대형</span>
            <select aria-label="포메이션 선택" value={formation} onChange={(event) => props.onFormation(event.target.value as FormationKey)}>
              {formations.map((item) => <option value={item} key={item}>{item} · {formationGuidance[item]}</option>)}
            </select>
          </label>
          <div className="formation-summary"><strong>{formation}</strong><span>{formationGuidance[formation]}</span><i>{formation === scenario.defaultFormation ? '실제 기준 대형' : 'WHAT-IF 대형'}</i></div>
          <p className="helper">대형을 적용한 뒤 선수 위치를 직접 조정할 수 있습니다.</p>

          <div className="section-divider" />
          <div className="panel-title"><span>02</span><div><small>TEAM INSTRUCTIONS</small><h2>팀 지시</h2></div></div>
          <Slider label="압박 강도" low="기다리기" high="즉시 압박" value={tactics.pressing} onChange={(value) => props.onTactic('pressing', value)} />
          <Slider label="공격 폭" low="좁게" high="넓게" value={tactics.width} onChange={(value) => props.onTactic('width', value)} />
          <Slider label="공격 템포" low="차분하게" high="빠르게" value={tactics.tempo} onChange={(value) => props.onTactic('tempo', value)} />
          <Slider label="위험 감수" low="안전하게" high="과감하게" value={tactics.risk} onChange={(value) => props.onTactic('risk', value)} />
        </aside>

        <section className="board-column">
          <div className="pitch" ref={pitchRef}>
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
                aria-label={`${player.name}, ${player.role}`}
              >
                <span>{player.number}</span><strong>{player.shortName}</strong>
              </button>
            ))}
          </div>

          <div className="bench panel">
            <div className="bench-heading"><div><small>BENCH</small><strong>{substitutionUsed ? '교체 완료' : '먼저 나갈 선수를 선택하세요'}</strong></div><span>{substitutionUsed ? '1 / 1' : '0 / 1'} 교체</span></div>
            <div className="bench-list">
              {bench.map((player) => (
                <button type="button" key={player.id} onClick={() => selectedPlayer?.onPitch && !substitutionUsed ? props.onSubstitute(player.id) : props.onSelect(player.id)}>
                  <span>{player.number}</span><div><strong>{player.shortName}</strong><small>{player.position} · {player.role}</small></div>
                  {selectedPlayer?.onPitch && !substitutionUsed && <em>투입</em>}
                </button>
              ))}
            </div>
          </div>
        </section>

        <aside className="insight-column">
          <section className="panel selected-player">
            <div className="panel-title"><span>03</span><div><small>PLAYER ROLE</small><h2>개인 역할</h2></div></div>
            {selectedPlayer ? (
              <>
                <div className="selected-summary"><b>{selectedPlayer.number}</b><div><strong>{selectedPlayer.name}</strong><small>{selectedPlayer.position} · {selectedPlayer.onPitch ? 'ON PITCH' : 'BENCH'}</small></div></div>
                {selectedEvidence ? (
                  <div className="attribute-row evidence"><span>패스 <b>{selectedEvidence.passesCompleted}/{selectedEvidence.passesAttempted}</b></span><span>압박 <b>{selectedEvidence.pressures}</b></span><span>슈팅 <b>{selectedEvidence.shots}</b></span></div>
                ) : <p className="no-evidence">65분 이전 미출전 · 경기 내 관측값 없음</p>}
                {selectedEvidence?.note && <p className="evidence-note">{selectedEvidence.note}</p>}
                <label className="role-select">역할<select value={selectedPlayer.role} onChange={(event) => props.onRole(event.target.value)} disabled={!selectedPlayer.onPitch}>{roleOptions[selectedPlayer.position].map((role) => <option key={role}>{role}</option>)}</select></label>
              </>
            ) : <p>선수를 선택하세요.</p>}
          </section>

          <section className="panel metrics-panel">
            <div className="panel-title"><span>04</span><div><small>SCENARIO COMPARISON</small><h2>전술 리스크 검토</h2></div></div>
            <Metric label={scenario.metricLabels[0]} value={metrics.threat} good />
            <Metric label={scenario.metricLabels[1]} value={metrics.control} good />
            <Metric label={scenario.metricLabels[2]} value={metrics.exposure} />
            <Metric label={scenario.metricLabels[3]} value={metrics.fatigue} />
            <div className="projection-note"><span>✦</span><p><strong>분석 코멘트</strong>{getCoachNote(metrics, squad, scenario)}</p></div>
            <small className="estimate-label">* {scenario.windowLabel} 관측값에 대형·배치 높이·팀 폭·역할·교체·팀 지시의 설명 가능한 규칙을 적용합니다. {evidenceMethod.caution}</small>
          </section>

          <button className="primary-button submit-tactic" type="button" onClick={props.onSubmit}>WHAT-IF 비교 실행 <span>→</span></button>
          <button className="text-button center" type="button" onClick={props.onBack}>브리핑 다시 보기</button>
        </aside>
      </div>
    </main>
  )
}

function Slider({ label, low, high, value, onChange }: { label: string; low: string; high: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="tactic-slider">
      <span><strong>{label}</strong><b>{value}</b></span>
      <input type="range" min="0" max="100" value={value} onChange={(event) => onChange(Number(event.target.value))} style={{ '--range': `${value}%` } as React.CSSProperties} />
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

function ResultScreen({ scenario, metrics, squad, tactics, formation, onRetry }: { scenario: GuidedScenario; metrics: Metrics; squad: Player[]; tactics: Tactics; formation: FormationKey; onRetry: () => void }) {
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
        <p className="eyebrow">TACTICAL WHAT-IF · COACHING MODEL, NOT A MATCH PREDICTION</p>
        <div className="decision-status"><span>TACTICAL PROPOSAL</span><strong>{formation}</strong><em>{recommendation}</em></div>
        <h1>{planLabel}</h1>
        <p>{scenario.result.baseline}을 기준선으로 사용자의 전술안을 비교한 코칭 검토 결과입니다.</p>
      </div>

      <section className="what-if-panel panel" aria-label="실제 기준 전술과 사용자 전술 변화 비교">
        <header><div><small>BASELINE → MY PLAN</small><h2>내 개입으로 달라진 전술 상태</h2></div><span>{scenario.defaultFormation} 기준</span></header>
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

      {scenario.id === 'korea-portugal-65' ? <TacticalSequence hwangOn={impactOn} formation={formation} tactics={tactics} /> : <PassNetwork scenario={scenario} />}

      <section className="result-grid">
        <article className="manager-card">
          <div className="card-top"><span>RE:TACTIC</span><small>INTERVENTION NOTE · {scenario.minute}′</small></div>
          <div className="manager-badge">R:</div>
          <p>PROPOSED INTERVENTION</p>
          <h2>{planLabel}</h2>
          <div className="manager-traits"><span>{formation}</span><span>위험 {tactics.risk}</span><span>템포 {tactics.tempo}</span><span>{impactOn ? `${impactPlayer} 투입` : '기존 인원 유지'}</span></div>
          <small>운영 조건 · {operatingCondition}</small>
        </article>

        <article className="report-card">
          <div className="panel-title"><span>05</span><div><small>DECISION MEMO</small><h2>이점과 리스크를 검토합니다</h2></div></div>
          <ul className="report-list">
            <li className={impactOn ? 'positive' : 'neutral'}><b>{impactOn ? '✓' : '!'}</b><div><strong>{scenario.id === 'korea-portugal-65' ? '전진 수단' : '측면 대응'}</strong><span>{impactOn ? scenario.result.impactOn : scenario.result.impactOff}</span></div></li>
            <li className={metrics.exposure < 60 ? 'positive' : 'negative'}><b>{metrics.exposure < 60 ? '✓' : '!'}</b><div><strong>전환 수비</strong><span>{metrics.exposure < 60 ? '공격적 개입 속에서도 후방 숫자를 관리할 수 있는 범위입니다.' : '압박과 위험 감수가 함께 높아 공을 잃은 뒤 중앙 보호 조건이 필요합니다.'}</span></div></li>
            <li className={metrics.control > 50 ? 'positive' : 'neutral'}><b>{metrics.control > 50 ? '✓' : '!'}</b><div><strong>{scenario.metricLabels[1]}</strong><span>{scenario.windowLabel} 패스 성공률 {scenario.evidence.ours.passCompletion}%를 기준으로 한 비교 점수는 {metrics.control}점입니다.</span></div></li>
          </ul>
          <div className="actual-choice"><span>실제 경기와 비교</span><p>{scenario.result.actualChoice} {scenario.result.actualOutcome} 이 사실은 사후 비교 정보이며 시나리오 점수 계산에는 정답값으로 사용하지 않습니다.</p></div>
        </article>
      </section>

      <div className="result-actions">
        <button className="secondary-button" type="button" onClick={() => window.print()}>전술안 인쇄·저장</button>
        <button className="primary-button" type="button" onClick={onRetry}>전술안 다시 설계 <span>↻</span></button>
      </div>
      <footer className="data-source">
        <img src="/statsbomb-logo.png" alt="StatsBomb" />
        <p>Match event data: StatsBomb Open Data · Match {scenario.matchId}. 파생 지표는 RE:TACTIC이 계산했습니다. 시나리오 점수는 실제 경기 결과 예측이나 승률이 아닙니다.</p>
      </footer>
    </main>
  )
}

export default App
