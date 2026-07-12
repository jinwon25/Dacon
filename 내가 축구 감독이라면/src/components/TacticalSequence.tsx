import { useEffect, useState } from 'react'
import { koreaPortugal360 } from '../data/freezeFrameEvidence'
import type { FormationKey, Tactics } from '../types'

type SequenceView = 'actual' | 'proposal'

interface SequencePlayer {
  id: string
  label: string
  team: 'korea' | 'portugal'
  from: [number, number]
  to: [number, number]
}

const actualFrame = (shot: boolean): SequencePlayer[] => (shot ? koreaPortugal360.shot.players : koreaPortugal360.pass.players).map((player) => ({
  id: player.id,
  label: player.label,
  team: player.team,
  from: [player.x, player.y],
  to: [player.x, player.y],
}))

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
  const players = view === 'actual' ? actualFrame(phase) : proposalPlayers(hwangOn, formation)
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
        <div className="panel-title"><span>05</span><div><small>2차원 전술 움직임</small><h2>변경 전후의 움직임을 비교합니다</h2></div></div>
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
            {view === 'actual'
              ? <line x1="60.7" y1="41.2" x2="87.1" y2="68.4" className="sequence-route actual" />
              : players.filter((player) => player.team === 'korea').map((player) => <line key={player.id} x1={player.from[0]} y1={player.from[1]} x2={player.to[0]} y2={player.to[1]} className="sequence-route proposal" />)}
          </svg>
          {players.map((player) => {
            const position = phase ? player.to : player.from
            return <span key={player.id} className={`sequence-player ${player.team}`} style={{ left: `${position[0]}%`, top: `${position[1]}%`, transitionDuration: `${tempoSeconds}s` }}>{player.label}</span>
          })}
          <span className={`sequence-ball ${phase ? 'moving' : ''} ${view}`} style={{ transitionDuration: `${tempoSeconds}s` }} />
          <em>{view === 'actual' ? (phase ? koreaPortugal360.shot.label : koreaPortugal360.pass.label) : '사용자 입력 기반 휴리스틱 동선'}</em>
        </div>

        <aside className="sequence-analysis">
          <span className={`sequence-type ${view}`}>{view === 'actual' ? 'OBSERVED' : 'PROPOSED'}</span>
          {view === 'actual' ? <>
            <h3>조규성의 연결 후 손흥민 슈팅</h3>
            <p>패스 순간 13명, 슈팅 순간 8명의 가시 선수를 실제 360 프리즈프레임으로 보여줍니다. 손흥민과 골키퍼 사이에 수비가 밀집해 슈팅 경로가 제한됐습니다.</p>
            <ul><li><b>패스</b> 조규성 → 손흥민</li><li><b>결과</b> Blocked · xG 0.024</li><li><b>근거</b> StatsBomb 360 두 이벤트 스냅샷</li></ul>
          </> : <>
            <h3>{hwangOn ? '반대편 러너를 추가한 전환안' : '기존 인원으로 폭을 넓힌 전환안'}</h3>
            <p>{formation} 대형에서 전방 폭을 넓혀 중앙 수비를 분산하고, 손흥민의 첫 패스 이후 두 번째 침투 선수를 만듭니다.</p>
            <ul><li><b>템포</b> {tactics.tempo}/100</li><li><b>위험 감수</b> {tactics.risk}/100</li><li><b>주의</b> 예측 결과가 아닌 전술 동선 비교</li></ul>
          </>}
        </aside>
      </div>
      <p className="sequence-disclaimer">실제 장면은 StatsBomb 360의 두 이벤트 프리즈프레임입니다. 이벤트 사이의 연속 트래킹이 아니며, 제안 장면의 움직임만 설명용 휴리스틱입니다.</p>
    </section>
  )
}
