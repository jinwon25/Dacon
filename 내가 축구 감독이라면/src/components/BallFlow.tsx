import { useEffect, useState } from 'react'
import type { GuidedScenario } from '../data/scenarios'

const actionLabel = { pass: '패스', carry: '운반', shot: '슈팅' } as const

export default function BallFlow({ scenario }: { scenario: GuidedScenario }) {
  const actions = scenario.flow.actions
  const [step, setStep] = useState(-1)
  const [playing, setPlaying] = useState(false)
  const current = step >= 0 ? actions[step] : null
  const ball = current?.end ?? actions[0].start

  useEffect(() => {
    setStep(-1)
    setPlaying(false)
  }, [scenario.id])

  useEffect(() => {
    if (!playing) return
    if (step >= actions.length - 1) {
      setPlaying(false)
      return
    }
    const timer = window.setTimeout(() => setStep((value) => value + 1), 850)
    return () => window.clearTimeout(timer)
  }, [actions.length, playing, step])

  const togglePlayback = () => {
    if (playing) {
      setPlaying(false)
      return
    }
    if (step >= actions.length - 1) setStep(-1)
    setPlaying(true)
  }

  return (
    <section className="ball-flow panel">
      <header className="ball-flow-heading">
        <div className="panel-title"><span>06</span><div><small>실제 볼 흐름 · 이벤트 순서</small><h2>{scenario.flow.title}</h2></div></div>
        <div className="flow-controls">
          <button type="button" onClick={() => { setPlaying(false); setStep(-1) }}>처음으로</button>
          <button className="flow-play" type="button" onClick={togglePlayback}>{playing ? '■ 일시정지' : '▶ 실제 흐름 재생'}</button>
        </div>
      </header>

      <div className="ball-flow-body">
        <div className="flow-pitch" aria-label={`${scenario.flow.title} 이벤트 좌표 재생`}>
          <svg viewBox="0 0 120 80" preserveAspectRatio="xMidYMid meet" role="img">
            <defs>
              <marker id={`flow-arrow-${scenario.id}`} markerUnits="userSpaceOnUse" markerWidth="3.6" markerHeight="3.6" refX="3.2" refY="1.8" viewBox="0 0 3.6 3.6" orient="auto"><path d="M0,0 L3.6,1.8 L0,3.6 Z" /></marker>
            </defs>
            <rect x="1" y="1" width="118" height="78" className="flow-pitch-line" />
            <line x1="60" y1="1" x2="60" y2="79" className="flow-pitch-line" />
            <circle cx="60" cy="40" r="9.15" className="flow-pitch-line" />
            <rect x="1" y="18" width="18" height="44" className="flow-pitch-line" />
            <rect x="101" y="18" width="18" height="44" className="flow-pitch-line" />
            {actions.map((action, index) => <line key={action.id} x1={action.start[0]} y1={action.start[1]} x2={action.end[0]} y2={action.end[1]} className={`flow-path ${action.type} ${index < step ? 'complete' : ''} ${index === step ? 'active' : ''}`} markerEnd={index === step ? `url(#flow-arrow-${scenario.id})` : undefined}><title>{action.time} {action.player} · {actionLabel[action.type]}</title></line>)}
            {actions.map((action, index) => <g className={`flow-event-node ${index <= step ? 'revealed' : ''} ${index === step ? 'active' : ''}`} key={`node-${action.id}`}><circle cx={action.start[0]} cy={action.start[1]} r="1.45" /><text x={action.start[0]} y={action.start[1] - 2.6}>{index + 1}</text></g>)}
            <circle className="flow-ball" cx={ball[0]} cy={ball[1]} r="1.65" />
          </svg>
          <span className="network-direction">공격 방향 →</span>
        </div>

        <aside className="flow-insight" aria-live="polite">
          <span className={`flow-type ${current?.type ?? 'ready'}`}>{current ? actionLabel[current.type] : 'READY'}</span>
          <strong>{current ? `${current.time} · ${current.player}` : '재생하면 실제 볼 이동을 순서대로 표시합니다'}</strong>
          <p>{current ? current.type === 'pass' ? `${current.recipient}에게 연결한 패스입니다.` : current.type === 'carry' ? '선수가 공을 직접 운반한 구간입니다.' : `${current.outcome ?? '슈팅'}으로 시퀀스가 끝났습니다.` : scenario.flow.summary}</p>
          <div className="flow-progress"><i style={{ width: `${step < 0 ? 0 : (step + 1) / actions.length * 100}%` }} /></div>
          <small>{Math.max(0, step + 1)} / {actions.length} EVENTS</small>
          <div className="flow-method"><b>표현 범위</b><span>StatsBomb 이벤트의 시작·종료 좌표를 시간순으로 연결했습니다.</span><span>아이콘 사이 이동은 설명용 보간이며 연속 볼·선수 트래킹이 아닙니다.</span></div>
        </aside>
      </div>

      <div className="flow-timeline" aria-label="이벤트 타임라인">
        {actions.map((action, index) => <button type="button" key={action.id} className={`${action.type} ${index === step ? 'active' : ''} ${index < step ? 'complete' : ''}`} aria-current={index === step ? 'step' : undefined} onClick={() => { setPlaying(false); setStep(index) }}><i>{String(index + 1).padStart(2, '0')}</i><span><b>{action.time}</b><small>{action.player} · {actionLabel[action.type]}</small></span></button>)}
      </div>
    </section>
  )
}
