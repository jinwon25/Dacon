import { useState } from 'react'
import type { GuidedScenario } from '../data/scenarios'

type View = 'entries' | 'shots'
const pitchX = (value: number) => value * 1.2
const pitchY = (value: number) => value * .8

export default function SpatialEvidence({ scenario }: { scenario: GuidedScenario }) {
  const [view, setView] = useState<View>('entries')
  const korea = scenario.spatial.ours
  const portugal = scenario.spatial.opponent

  return (
    <section className="spatial-evidence panel">
      <header className="spatial-heading">
        <div className="panel-title"><span>04</span><div><small>공간 근거 · {scenario.windowLabel}</small><h2>숫자가 만들어진 위치를 확인하세요</h2></div></div>
        <div className="spatial-tabs" role="tablist" aria-label="공간 데이터 종류">
          <button type="button" role="tab" aria-selected={view === 'entries'} className={view === 'entries' ? 'active' : ''} onClick={() => setView('entries')}>공격 지역 진입</button>
          <button type="button" role="tab" aria-selected={view === 'shots'} className={view === 'shots' ? 'active' : ''} onClick={() => setView('shots')}>슈팅 위치</button>
        </div>
      </header>

      <div className="spatial-body">
        <div className="event-pitch" aria-label={view === 'entries' ? '팀별 공격 지역 진입 패스맵' : '팀별 슈팅 위치'}>
          <svg viewBox="0 0 120 80" preserveAspectRatio="xMidYMid meet" role="img">
            <defs>
              <marker id="arrow-korea" markerUnits="userSpaceOnUse" markerWidth="4" markerHeight="4" refX="3.5" refY="2" viewBox="0 0 4 4" orient="auto"><path d="M0,0 L4,2 L0,4 Z" className="marker-korea" /></marker>
              <marker id="arrow-portugal" markerUnits="userSpaceOnUse" markerWidth="3.2" markerHeight="3.2" refX="2.8" refY="1.6" viewBox="0 0 3.2 3.2" orient="auto"><path d="M0,0 L3.2,1.6 L0,3.2 Z" className="marker-portugal" /></marker>
            </defs>
            <rect x="80" y="1" width="39" height="78" className="final-third-zone" />
            <rect x="1" y="1" width="118" height="78" className="event-pitch-outline" />
            <line x1="60" y1="1" x2="60" y2="79" className="event-pitch-line" />
            <circle cx="60" cy="40" r="9.15" className="event-pitch-line" />
            <circle cx="60" cy="40" r=".8" className="event-pitch-line event-center-dot" />
            <rect x="1" y="18" width="18" height="44" className="event-pitch-line" />
            <rect x="101" y="18" width="18" height="44" className="event-pitch-line" />
            <rect x="1" y="30" width="6" height="20" className="event-pitch-line" />
            <rect x="113" y="30" width="6" height="20" className="event-pitch-line" />
            <line x1="80" y1="1" x2="80" y2="79" className="final-third-line" />

            {view === 'entries' && <>
              {portugal.finalThirdEntries.map((pass, index) => <g key={`por-${index}`}><title>{pass.minute}′ {pass.player} → {pass.recipient}</title><line x1={pitchX(pass.start[0])} y1={pitchY(pass.start[1])} x2={pitchX(pass.end[0])} y2={pitchY(pass.end[1])} className="event-pass portugal" markerEnd="url(#arrow-portugal)" /><circle cx={pitchX(pass.start[0])} cy={pitchY(pass.start[1])} r=".8" className="event-node portugal" /></g>)}
              {korea.finalThirdEntries.map((pass, index) => <g key={`kor-${index}`}><title>{pass.minute}′ {pass.player} → {pass.recipient}</title><line x1={pitchX(pass.start[0])} y1={pitchY(pass.start[1])} x2={pitchX(pass.end[0])} y2={pitchY(pass.end[1])} className="event-pass korea" markerEnd="url(#arrow-korea)" /><circle cx={pitchX(pass.start[0])} cy={pitchY(pass.start[1])} r="1.35" className="event-node korea" /></g>)}
            </>}
            {view === 'shots' && <>
              {korea.shots.map((shot, index) => <g key={`kor-shot-${index}`}><title>{shot.minute}′ {shot.player} · xG {shot.xg}</title><circle cx={pitchX(shot.location[0])} cy={pitchY(shot.location[1])} r={3.4 + shot.xg * 20} className="event-shot korea" /><text x={pitchX(shot.location[0])} y={pitchY(shot.location[1]) + 1} className="shot-label">{shot.minute}′</text></g>)}
              {portugal.shots.map((shot, index) => <g key={`por-shot-${index}`}><title>{shot.minute}′ {shot.player} · xG {shot.xg}</title><circle cx={pitchX(shot.location[0])} cy={pitchY(shot.location[1])} r={3.4 + shot.xg * 20} className="event-shot portugal" /><text x={pitchX(shot.location[0])} y={pitchY(shot.location[1]) + 1} className="shot-label">{shot.minute}′</text></g>)}
            </>}
          </svg>
          <span className="pitch-direction">공격 방향 →</span>
        </div>

        <aside className="spatial-insight">
          <div className="spatial-legend"><span><i className="korea" /> {scenario.ours.name}</span><span><i className="portugal" /> {scenario.opponent.name}</span></div>
          {view === 'entries' ? <>
            <strong>{scenario.spatialCopy.entriesTitle}</strong>
            <p>{scenario.spatialCopy.entriesBody}</p>
            <div className="insight-kpis"><span><small>{scenario.ours.name}</small><b>{scenario.spatialCopy.entriesOurs}</b></span><span><small>{scenario.opponent.name}</small><b>{scenario.spatialCopy.entriesOpponent}</b></span></div>
          </> : <>
            <strong>{scenario.spatialCopy.shotsTitle}</strong>
            <p>{scenario.spatialCopy.shotsBody}</p>
            <div className="insight-kpis"><span><small>{scenario.ours.name} xG</small><b>{scenario.spatialCopy.shotsOurs}</b></span><span><small>{scenario.opponent.name} xG</small><b>{scenario.spatialCopy.shotsOpponent}</b></span></div>
          </>}
          <small className="source-note">StatsBomb event coordinates · Match {scenario.matchId}</small>
        </aside>
      </div>
    </section>
  )
}
