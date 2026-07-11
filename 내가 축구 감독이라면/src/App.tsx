import { useMemo, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import UniversalStudio from './components/UniversalStudio'
import SpatialEvidence from './components/SpatialEvidence'
import TacticalSequence from './components/TacticalSequence'
import { evidenceMethod, matchEvidence, playerEvidence } from './data/evidence'
import { formationPositions, initialSquad, roleOptions } from './data/match'
import type { FormationKey, Metrics, Player, Stage, Tactics } from './types'

const formations: FormationKey[] = ['4-3-3', '4-2-3-1', '3-4-3', '5-3-2']
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
): Metrics {
  const hwangHeechanOnPitch = squad.some((player) => player.onPitch && player.id === 'hwang-heechan')
  const attackBoost = formation === '3-4-3' ? 8 : formation === '4-2-3-1' ? 4 : formation === '5-3-2' ? -4 : 2
  const safetyBoost = formation === '5-3-2' ? 10 : formation === '3-4-3' ? -6 : 2
  const observedPassGap = matchEvidence.portugal.passCompletion - matchEvidence.southKorea.passCompletion
  const observedPressureLoad = matchEvidence.southKorea.pressures

  return {
    threat: clamp(Math.round(24 + tactics.tempo * 0.25 + tactics.risk * 0.28 + attackBoost + (hwangHeechanOnPitch ? 7 : 0))),
    control: clamp(Math.round(52 - observedPassGap * 0.6 + tactics.width * 0.12 + (100 - tactics.risk) * 0.08)),
    exposure: clamp(Math.round(12 + tactics.pressing * 0.14 + tactics.risk * 0.35 - safetyBoost)),
    fatigue: clamp(Math.round(12 + observedPressureLoad * 0.45 + tactics.pressing * 0.28 + tactics.tempo * 0.16)),
  }
}

function App() {
  const [mode, setMode] = useState<AppMode>(getInitialMode)
  const [stage, setStage] = useState<Stage>(getInitialStage)
  const [formation, setFormation] = useState<FormationKey>('4-3-3')
  const [squad, setSquad] = useState<Player[]>(initialSquad)
  const [selectedId, setSelectedId] = useState<string>('son-heungmin')
  const [draggingId, setDraggingId] = useState<string | null>(null)
  const [substitutionUsed, setSubstitutionUsed] = useState(false)
  const [tactics, setTactics] = useState<Tactics>({ pressing: 58, width: 62, tempo: 64, risk: 56 })
  const pitchRef = useRef<HTMLDivElement>(null)

  const selectedPlayer = squad.find((player) => player.id === selectedId) ?? null
  const metrics = useMemo(() => calculateMetrics(squad, tactics, formation), [squad, tactics, formation])

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
    setFormation('4-3-3')
    setSquad(initialSquad)
    setSelectedId('son-heungmin')
    setSubstitutionUsed(false)
    setTactics({ pressing: 58, width: 62, tempo: 64, risk: 56 })
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
            <span>🇰🇷 KOR</span><strong>1 : 1</strong><span>POR 🇵🇹</span><em>65′</em>
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
      {stage === 'intro' && <IntroScreen onStart={() => setStage('briefing')} onStudio={() => switchMode('studio')} />}
      {stage === 'briefing' && <BriefingScreen onBack={() => setStage('intro')} onNext={() => setStage('tactics')} />}
      {stage === 'tactics' && (
        <TacticsScreen
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
      {stage === 'result' && <ResultScreen metrics={metrics} squad={squad} tactics={tactics} formation={formation} onRetry={() => setStage('tactics')} />}
      </>}
    </div>
  )
}

function IntroScreen({ onStart, onStudio }: { onStart: () => void; onStudio: () => void }) {
  return (
    <main className="intro-screen">
      <div className="stadium-glow" />
      <section className="intro-copy">
        <p className="eyebrow">MATCH INTERVENTION LAB · 2022 WORLD CUP</p>
        <h1>남은 시간 <span>25분.</span><br />한 골이 필요합니다.</h1>
        <p className="intro-lead">
          포르투갈과 1–1. 이대로라면 탈락입니다.<br />실제 경기 데이터를 확인하고 65분 개입안을 설계하세요.
        </p>
        <div className="objective-card">
          <span className="objective-icon">◎</span>
          <div>
            <small>MATCH OBJECTIVE</small>
            <strong>균형을 잃지 않고 결승골을 만들어라</strong>
          </div>
        </div>
        <div className="intro-actions">
          <button className="primary-button large" type="button" onClick={onStart}>실제 경기로 익히기 <span>→</span></button>
          <button className="secondary-button large" type="button" onClick={onStudio}>빈 전술판에서 시작</button>
        </div>
        <p className="no-login"><b>처음이라면 실제 경기 분석을 추천합니다.</b> 로그인 없이 약 3분 · 마우스와 터치 지원</p>
      </section>
      <section className="intro-scoreboard" aria-label="경기 상황">
        <div className="time-ring"><strong>65</strong><span>MIN</span></div>
        <div className="teams-row">
          <div><span className="flag-orb korea">🇰🇷</span><strong>대한민국</strong><small>승리 필요</small></div>
          <b>1</b><i>:</i><b>1</b>
          <div><span className="flag-orb portugal">🇵🇹</span><strong>포르투갈</strong><small>조 1위 확정권</small></div>
        </div>
        <div className="timeline-mini">
          <span style={{ left: '5%' }}>5′ <b>0–1</b></span>
          <span style={{ left: '31%' }}>27′ <b>1–1</b></span>
          <em style={{ left: '72%' }}>YOU ARE HERE</em>
        </div>
      </section>
    </main>
  )
}

function BriefingScreen({ onBack, onNext }: { onBack: () => void; onNext: () => void }) {
  const korea = matchEvidence.southKorea
  const portugal = matchEvidence.portugal

  return (
    <main className="briefing-screen page-wrap">
      <div className="page-heading">
        <div>
          <p className="eyebrow">MATCH INTERVENTION BRIEF · 65′</p>
          <h1>직전 20분의 문제를 먼저 정의합니다.</h1>
          <p>StatsBomb 실제 이벤트 45–64분을 기준으로 만든 코칭 스태프용 경기 스냅샷입니다.</p>
        </div>
        <span className="live-pill"><i /> VERIFIED MATCH DATA</span>
      </div>

      <section className="briefing-grid">
        <article className="analysis-card pitch-analysis">
          <div className="card-heading"><span>01</span><div><small>LAST 20 MINUTES</small><h2>전진보다 압박에 에너지를 썼습니다</h2></div></div>
          <div className="evidence-comparison">
            <div className="comparison-head"><span>45′–64′</span><b>대한민국</b><b>포르투갈</b></div>
            <ComparisonRow label="패스 성공률" korea={`${korea.passCompletion}%`} portugal={`${portugal.passCompletion}%`} koreaValue={korea.passCompletion} portugalValue={portugal.passCompletion} />
            <ComparisonRow label="공격 지역 진입" korea={`${korea.finalThirdEntries}회`} portugal={`${portugal.finalThirdEntries}회`} koreaValue={korea.finalThirdEntries} portugalValue={portugal.finalThirdEntries} max={16} />
            <ComparisonRow label="박스 진입" korea={`${korea.boxEntries}회`} portugal={`${portugal.boxEntries}회`} koreaValue={korea.boxEntries} portugalValue={portugal.boxEntries} max={4} />
            <ComparisonRow label="압박" korea={`${korea.pressures}회`} portugal={`${portugal.pressures}회`} koreaValue={korea.pressures} portugalValue={portugal.pressures} max={30} invert />
          </div>
          <p className="coach-quote">“압박은 더 많았지만 전진 진입은 4 대 16입니다. 탈취 이후 첫 두 번의 패스를 개선해야 합니다.”</p>
        </article>

        <article className="analysis-card">
          <div className="card-heading"><span>02</span><div><small>MATCH CONTEXT</small><h2>우리에게 필요한 결과</h2></div></div>
          <div className="context-number"><strong>1</strong><span>GOAL<br />NEEDED</span></div>
          <ul className="signal-list">
            <li><span className="signal good">↗</span><div><strong>승리 시</strong><small>다른 경기 결과에 따라 16강 진출</small></div></li>
            <li><span className="signal warn">—</span><div><strong>무승부 시</strong><small>조별리그 탈락</small></div></li>
          </ul>
        </article>

        <article className="analysis-card">
          <div className="card-heading"><span>03</span><div><small>INTERVENTION OPTION</small><h2>전환 속도를 높일 교체안</h2></div></div>
          <div className="player-spotlight">
            <div className="shirt-number">11</div>
            <div><strong>황희찬</strong><small>FW · 라인 브레이커</small></div>
            <b>AVAILABLE 65′</b>
          </div>
          <div className="trait-row"><span>이재성 OUT</span><span>황희찬 IN</span><span>전환 공격</span></div>
          <p className="coach-quote accent">“실제 경기에서도 65분에 실행된 교체입니다. 여기서는 결과를 복제하지 않고 전술적 이점과 리스크를 비교합니다.”</p>
        </article>
      </section>

      <SpatialEvidence />

      <div className="source-strip">
        <img src="/statsbomb-logo.png" alt="StatsBomb" />
        <p><strong>데이터 근거</strong> Match 3857262 · 45′ 이상 65′ 미만 이벤트 직접 집계 · 추출일 2026-07-11</p>
        <a href={matchEvidence.sourceUrl} target="_blank" rel="noreferrer">원본 저장소 ↗</a>
      </div>

      <div className="page-actions">
        <button className="text-button" type="button" onClick={onBack}>← 이전</button>
        <button className="primary-button" type="button" onClick={onNext}>전술 보드로 이동 <span>→</span></button>
      </div>
    </main>
  )
}

function ComparisonRow({ label, korea, portugal, koreaValue, portugalValue, max = 100, invert = false }: { label: string; korea: string; portugal: string; koreaValue: number; portugalValue: number; max?: number; invert?: boolean }) {
  const koreaWidth = clamp(koreaValue / max * 100)
  const portugalWidth = clamp(portugalValue / max * 100)
  return (
    <div className={`comparison-row ${invert ? 'invert' : ''}`}>
      <span>{label}</span><strong>{korea}</strong><strong>{portugal}</strong>
      <div className="comparison-bars"><i style={{ width: `${koreaWidth}%` }} /><i style={{ width: `${portugalWidth}%` }} /></div>
    </div>
  )
}

interface TacticsScreenProps {
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
  const { formation, squad, selectedPlayer, draggingId, substitutionUsed, tactics, metrics, pitchRef } = props
  const onPitch = squad.filter((player) => player.onPitch)
  const bench = squad.filter((player) => !player.onPitch)
  const selectedEvidence = selectedPlayer ? playerEvidence[selectedPlayer.id] : null

  return (
    <main className="tactics-screen page-wrap wide">
      <div className="tactics-heading">
        <div><p className="eyebrow">MATCH INTERVENTION WORKSPACE · 65′</p><h1>개입 전술안을 설계하세요.</h1></div>
        <div className="decision-clock"><span>분석 기준</span><strong>65′</strong><small>관측값 + 시나리오 비교</small></div>
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
          <div className="formation-grid">
            {formations.map((item) => <button className={formation === item ? 'selected' : ''} type="button" key={item} onClick={() => props.onFormation(item)}>{item}</button>)}
          </div>
          <p className="helper">프리셋 적용 후 선수를 자유롭게 움직일 수 있습니다.</p>

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
            <Metric label="침투 가능성" value={metrics.threat} good />
            <Metric label="볼 순환 안정성" value={metrics.control} good />
            <Metric label="전환 노출" value={metrics.exposure} />
            <Metric label="고강도 부담" value={metrics.fatigue} />
            <div className="projection-note"><span>✦</span><p><strong>분석 코멘트</strong>{getCoachNote(metrics, squad)}</p></div>
            <small className="estimate-label">* {evidenceMethod.modeled}. {evidenceMethod.caution}</small>
          </section>

          <button className="primary-button submit-tactic" type="button" onClick={props.onSubmit}>전술안 분석하기 <span>→</span></button>
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

function getCoachNote(metrics: Metrics, squad: Player[]) {
  const hwangOn = squad.some((player) => player.id === 'hwang-heechan' && player.onPitch)
  if (metrics.exposure > 65) return '공격 숫자는 충분하지만 공을 잃은 직후 중앙 공간이 위험합니다.'
  if (hwangOn && metrics.threat > 60) return '황희찬의 속도가 손흥민의 전진 패스와 연결될 가능성이 높습니다.'
  if (metrics.threat < 52) return '탈락을 피하려면 더 빠른 템포나 뒷공간을 노릴 선수가 필요합니다.'
  return '균형은 안정적입니다. 이제 승부를 바꿀 한 가지 과감한 선택이 필요합니다.'
}

function ResultScreen({ metrics, squad, tactics, formation, onRetry }: { metrics: Metrics; squad: Player[]; tactics: Tactics; formation: FormationKey; onRetry: () => void }) {
  const hwangOn = squad.some((player) => player.id === 'hwang-heechan' && player.onPitch)
  const performance = metrics.threat + metrics.control - metrics.exposure * 0.65 - metrics.fatigue * 0.15 + (hwangOn ? 8 : 0)
  const recommendation = performance >= 90 ? '채택 권고' : performance < 58 ? '재설계 필요' : '조건부 채택'
  const tone = performance >= 90 ? 'win' : performance < 58 ? 'lose' : 'draw'
  const planLabel = hwangOn ? '전환 속도 강화안' : tactics.pressing > 68 ? '고강도 압박 유지안' : '점유 안정화안'
  const operatingCondition = metrics.exposure > 60
    ? '공을 잃은 직후 중앙 미드필더 한 명은 반드시 잔류'
    : '첫 전진 패스 실패 시 즉시 수비 블록 복귀'

  return (
    <main className="result-screen page-wrap">
      <div className={`result-hero ${tone}`}>
        <p className="eyebrow">INTERVENTION REVIEW · NOT A MATCH PREDICTION</p>
        <div className="decision-status"><span>TACTICAL PROPOSAL</span><strong>{formation}</strong><em>{recommendation}</em></div>
        <h1>{planLabel}</h1>
        <p>45–64분 실제 관측값을 기준선으로 사용자의 전술안을 비교한 코칭 검토 결과입니다.</p>
      </div>

      <TacticalSequence hwangOn={hwangOn} formation={formation} tactics={tactics} />

      <section className="result-grid">
        <article className="manager-card">
          <div className="card-top"><span>RE:TACTIC</span><small>INTERVENTION NOTE · 65′</small></div>
          <div className="manager-badge">R:</div>
          <p>PROPOSED INTERVENTION</p>
          <h2>{planLabel}</h2>
          <div className="manager-traits"><span>{formation}</span><span>위험 {tactics.risk}</span><span>템포 {tactics.tempo}</span><span>{hwangOn ? '황희찬 투입' : '기존 인원 유지'}</span></div>
          <small>운영 조건 · {operatingCondition}</small>
        </article>

        <article className="report-card">
          <div className="panel-title"><span>05</span><div><small>DECISION MEMO</small><h2>이점과 리스크를 검토합니다</h2></div></div>
          <ul className="report-list">
            <li className={hwangOn ? 'positive' : 'neutral'}><b>{hwangOn ? '✓' : '!'}</b><div><strong>전진 수단</strong><span>{hwangOn ? '황희찬 투입으로 탈취 후 측면 뒷공간을 바로 공격할 선택지가 생깁니다.' : '기존 인원 유지 시 패스 성공률 격차를 줄일 별도 전진 패턴이 필요합니다.'}</span></div></li>
            <li className={metrics.exposure < 60 ? 'positive' : 'negative'}><b>{metrics.exposure < 60 ? '✓' : '!'}</b><div><strong>전환 수비</strong><span>{metrics.exposure < 60 ? '공격적 개입 속에서도 후방 숫자를 관리할 수 있는 범위입니다.' : '압박과 위험 감수가 함께 높아 공을 잃은 뒤 중앙 보호 조건이 필요합니다.'}</span></div></li>
            <li className={metrics.control > 50 ? 'positive' : 'neutral'}><b>{metrics.control > 50 ? '✓' : '!'}</b><div><strong>볼 순환</strong><span>직전 20분 패스 성공률 65.6%를 기준으로 한 안정성 비교 점수는 {metrics.control}점입니다.</span></div></li>
          </ul>
          <div className="actual-choice"><span>실제 경기와 비교</span><p>실제 경기에서는 65분 황희찬 투입 후 추가시간 손흥민의 전진 패스를 받아 결승골을 기록했습니다. 이 사실은 사후 비교 정보이며 시나리오 점수 계산에는 정답값으로 사용하지 않습니다.</p></div>
        </article>
      </section>

      <div className="result-actions">
        <button className="secondary-button" type="button" onClick={() => window.print()}>전술안 인쇄·저장</button>
        <button className="primary-button" type="button" onClick={onRetry}>전술안 다시 설계 <span>↻</span></button>
      </div>
      <footer className="data-source">
        <img src="/statsbomb-logo.png" alt="StatsBomb" />
        <p>Match event data: StatsBomb Open Data · Match 3857262. 파생 지표는 RE:TACTIC이 계산했습니다. 시나리오 점수는 실제 경기 결과 예측이나 승률이 아닙니다.</p>
      </footer>
    </main>
  )
}

export default App
