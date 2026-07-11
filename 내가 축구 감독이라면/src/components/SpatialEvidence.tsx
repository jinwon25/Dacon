import { useState } from 'react'
import { spatialEvidence } from '../data/spatialEvidence'

type View = 'entries' | 'shots'

export default function SpatialEvidence() {
  const [view, setView] = useState<View>('entries')
  const korea = spatialEvidence['South Korea']
  const portugal = spatialEvidence.Portugal

  return (
    <section className="spatial-evidence panel">
      <header className="spatial-heading">
        <div className="panel-title"><span>04</span><div><small>SPATIAL EVIDENCE · 45′–64′</small><h2>숫자가 만들어진 위치를 확인하세요</h2></div></div>
        <div className="spatial-tabs" role="tablist" aria-label="공간 데이터 종류">
          <button type="button" className={view === 'entries' ? 'active' : ''} onClick={() => setView('entries')}>공격 지역 진입</button>
          <button type="button" className={view === 'shots' ? 'active' : ''} onClick={() => setView('shots')}>슈팅 위치</button>
        </div>
      </header>

      <div className="spatial-body">
        <div className="event-pitch" aria-label={view === 'entries' ? '팀별 공격 지역 진입 패스맵' : '팀별 슈팅 위치'}>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img">
            <defs>
              <marker id="arrow-korea" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" className="marker-korea" /></marker>
              <marker id="arrow-portugal" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" className="marker-portugal" /></marker>
            </defs>
            <rect x="1" y="1" width="98" height="98" className="event-pitch-outline" />
            <line x1="50" y1="1" x2="50" y2="99" className="event-pitch-line" />
            <ellipse cx="50" cy="50" rx="10" ry="15" className="event-pitch-line" />
            <rect x="1" y="22" width="15" height="56" className="event-pitch-line" />
            <rect x="84" y="22" width="15" height="56" className="event-pitch-line" />
            <line x1="66.7" y1="1" x2="66.7" y2="99" className="final-third-line" />

            {view === 'entries' && <>
              {korea.finalThirdEntries.map((pass, index) => <g key={`kor-${index}`}><title>{pass.minute}′ {pass.player} → {pass.recipient}</title><line x1={pass.start[0]} y1={pass.start[1]} x2={pass.end[0]} y2={pass.end[1]} className="event-pass korea" markerEnd="url(#arrow-korea)" /><circle cx={pass.start[0]} cy={pass.start[1]} r="1.2" className="event-node korea" /></g>)}
              {portugal.finalThirdEntries.map((pass, index) => <g key={`por-${index}`}><title>{pass.minute}′ {pass.player} → {pass.recipient}</title><line x1={pass.start[0]} y1={pass.start[1]} x2={pass.end[0]} y2={pass.end[1]} className="event-pass portugal" markerEnd="url(#arrow-portugal)" /><circle cx={pass.start[0]} cy={pass.start[1]} r=".8" className="event-node portugal" /></g>)}
            </>}
            {view === 'shots' && <>
              {korea.shots.map((shot, index) => <g key={`kor-shot-${index}`}><title>{shot.minute}′ {shot.player} · xG {shot.xg}</title><circle cx={shot.location[0]} cy={shot.location[1]} r={2.5 + shot.xg * 30} className="event-shot korea" /><text x={shot.location[0]} y={shot.location[1] + .7} className="shot-label">{shot.minute}′</text></g>)}
              {portugal.shots.map((shot, index) => <g key={`por-shot-${index}`}><title>{shot.minute}′ {shot.player} · xG {shot.xg}</title><circle cx={shot.location[0]} cy={shot.location[1]} r={2.5 + shot.xg * 30} className="event-shot portugal" /><text x={shot.location[0]} y={shot.location[1] + .7} className="shot-label">{shot.minute}′</text></g>)}
            </>}
          </svg>
          <span className="pitch-direction">공격 방향 →</span>
        </div>

        <aside className="spatial-insight">
          <div className="spatial-legend"><span><i className="korea" /> 대한민국</span><span><i className="portugal" /> 포르투갈</span></div>
          {view === 'entries' ? <>
            <strong>한국의 진입은 적고 직접적이었습니다</strong>
            <p>4회 중 2회가 김승규에서 조규성으로 향한 긴 패스였습니다. 반면 포르투갈은 칸셀루·비티냐를 거쳐 양 측면으로 16회 진입했습니다.</p>
            <div className="insight-kpis"><span><small>한국</small><b>4</b></span><span><small>포르투갈</small><b>16</b></span></div>
          </> : <>
            <strong>양 팀 모두 양질의 슈팅을 만들지 못했습니다</strong>
            <p>20분 동안 한국 2회, 포르투갈 1회의 슈팅이 나왔지만 양 팀의 합산 xG는 0.098에 그쳤습니다.</p>
            <div className="insight-kpis"><span><small>한국 xG</small><b>0.045</b></span><span><small>포르투갈 xG</small><b>0.053</b></span></div>
          </>}
          <small className="source-note">StatsBomb event coordinates · Match 3857262</small>
        </aside>
      </div>
    </section>
  )
}
