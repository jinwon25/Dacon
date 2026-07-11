import { useEffect, useState } from 'react'
import type { FormationKey, Tactics } from '../types'

type SequenceView = 'actual' | 'proposal'

interface SequencePlayer {
  id: string
  label: string
  team: 'korea' | 'portugal'
  from: [number, number]
  to: [number, number]
}

const actualPlayers: SequencePlayer[] = [
  { id: 'cho', label: '조', team: 'korea', from: [60.7, 41.2], to: [72, 45] },
  { id: 'son', label: '손', team: 'korea', from: [75, 66], to: [87.1, 68.4] },
  { id: 'lee', label: '이', team: 'korea', from: [61, 24], to: [71, 30] },
  { id: 'hwang', label: '황', team: 'korea', from: [52, 52], to: [66, 54] },
  { id: 'por-1', label: 'P', team: 'portugal', from: [73, 33], to: [79, 42] },
  { id: 'por-2', label: 'P', team: 'portugal', from: [80, 58], to: [85, 61] },
  { id: 'por-3', label: 'P', team: 'portugal', from: [69, 76], to: [77, 72] },
]

function proposalPlayers(hwangOn: boolean, formation: FormationKey): SequencePlayer[] {
  const wideStart = formation === '3-4-3' ? 24 : 30
  return [
    { id: 'pivot', label: '황', team: 'korea', from: [52, 48], to: [64, 45] },
    { id: 'son-proposal', label: '손', team: 'korea', from: [64, 66], to: [80, 57] },
    { id: 'runner', label: hwangOn ? '희' : '이', team: 'korea', from: [wideStart, 20], to: [84, 19] },
    { id: 'nine', label: '조', team: 'korea', from: [72, 46], to: [88, 42] },
    { id: 'cover', label: '정', team: 'korea', from: [48, 61], to: [58, 62] },
    { id: 'por-a', label: 'P', team: 'portugal', from: [71, 31], to: [79, 38] },
    { id: 'por-b', label: 'P', team: 'portugal', from: [78, 55], to: [84, 50] },
    { id: 'por-c', label: 'P', team: 'portugal', from: [67, 75], to: [75, 68] },
  ]
}

export default function TacticalSequence({ hwangOn, formation, tactics }: { hwangOn: boolean; formation: FormationKey; tactics: Tactics }) {
  const [view, setView] = useState<SequenceView>('actual')
  const [phase, setPhase] = useState(false)
  const players = view === 'actual' ? actualPlayers : proposalPlayers(hwangOn, formation)
  const tempoSeconds = Math.max(1.2, 2.8 - tactics.tempo / 60)

  useEffect(() => {
    setPhase(false)
  }, [view])

  const play = () => {
    setPhase(false)
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => setPhase(true)))
  }

  return (
    <section className="tactical-sequence panel">
      <header className="sequence-heading">
        <div className="panel-title"><span>05</span><div><small>2D TACTICAL SEQUENCE</small><h2>변경 전후의 움직임을 비교합니다</h2></div></div>
        <div className="sequence-controls">
          <div role="tablist" aria-label="전술 시퀀스 선택">
            <button type="button" className={view === 'actual' ? 'active' : ''} onClick={() => setView('actual')}>실제 55분</button>
            <button type="button" className={view === 'proposal' ? 'active' : ''} onClick={() => setView('proposal')}>내 개입안</button>
          </div>
          <button className="play-sequence" type="button" onClick={play}>▶ 재생</button>
        </div>
      </header>

      <div className="sequence-body">
        <div className="sequence-pitch">
          <div className="sequence-pitch-lines"><i /><i /><i /></div>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            {players.filter((player) => player.team === 'korea').map((player) => <line key={player.id} x1={player.from[0]} y1={player.from[1]} x2={player.to[0]} y2={player.to[1]} className={`sequence-route ${view}`} />)}
          </svg>
          {players.map((player) => {
            const position = phase ? player.to : player.from
            return <span key={player.id} className={`sequence-player ${player.team}`} style={{ left: `${position[0]}%`, top: `${position[1]}%`, transitionDuration: `${tempoSeconds}s` }}>{player.label}</span>
          })}
          <span className={`sequence-ball ${phase ? 'moving' : ''} ${view}`} style={{ transitionDuration: `${tempoSeconds}s` }} />
          <em>{view === 'actual' ? '실제 이벤트 좌표 기반' : '사용자 입력 기반 휴리스틱 동선'}</em>
        </div>

        <aside className="sequence-analysis">
          <span className={`sequence-type ${view}`}>{view === 'actual' ? 'OBSERVED' : 'PROPOSED'}</span>
          {view === 'actual' ? <>
            <h3>조규성의 연결 후 손흥민 슈팅</h3>
            <p>55분 공격 지역 진입은 슈팅으로 이어졌지만, 손흥민 주변의 지원 숫자가 적어 수비가 슈팅 경로를 빠르게 차단했습니다.</p>
            <ul><li><b>패스</b> 조규성 → 손흥민</li><li><b>결과</b> Blocked · xG 0.024</li><li><b>근거</b> StatsBomb 이벤트 좌표</li></ul>
          </> : <>
            <h3>{hwangOn ? '반대편 러너를 추가한 전환안' : '기존 인원으로 폭을 넓힌 전환안'}</h3>
            <p>{formation} 대형에서 전방 폭을 넓혀 중앙 수비를 분산하고, 손흥민의 첫 패스 이후 두 번째 침투 선수를 만듭니다.</p>
            <ul><li><b>템포</b> {tactics.tempo}/100</li><li><b>위험 감수</b> {tactics.risk}/100</li><li><b>주의</b> 예측 결과가 아닌 전술 동선 비교</li></ul>
          </>}
        </aside>
      </div>
      <p className="sequence-disclaimer">공과 패스 좌표는 실제 이벤트를 사용합니다. 공이 없는 선수의 위치는 360 추적 데이터가 없어 설명용으로 구성했습니다.</p>
    </section>
  )
}
