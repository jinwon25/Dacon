import { useMemo, useState } from 'react'
import type { GuidedScenario, PassNetworkData } from '../data/scenarios'

type TeamView = 'ours' | 'opponent'

const pitchX = (value: number) => value * 1.2
const pitchY = (value: number) => value * .8
const nodeRadius = (involvements: number) => Math.min(5, 2.1 + involvements * .075)

function strongestConnection(data: PassNetworkData) {
  return data.edges.reduce<PassNetworkData['edges'][number] | null>((best, edge) => !best || edge.count > best.count ? edge : best, null)
}

export default function PassNetwork({ scenario }: { scenario: GuidedScenario }) {
  const [teamView, setTeamView] = useState<TeamView>('ours')
  const data = scenario.networks[teamView]
  const team = scenario[teamView]
  const nodeById = useMemo(() => new Map(data.nodes.map((node) => [node.id, node])), [data.nodes])
  const hub = data.nodes.reduce((best, node) => node.involvements > best.involvements ? node : best, data.nodes[0])
  const strongest = strongestConnection(data)
  const strongestLabels = strongest
    ? `${nodeById.get(strongest.from)?.label ?? strongest.from} → ${nodeById.get(strongest.to)?.label ?? strongest.to}`
    : '반복 연결 없음'

  return (
    <section className="pass-network panel">
      <header className="pass-network-heading">
        <div className="panel-title"><span>05</span><div><small>PASS NETWORK · {scenario.windowLabel}</small><h2>패스 연결 구조를 읽어보세요</h2></div></div>
        <div className="network-tabs" role="tablist" aria-label="패스맵 팀 선택">
          <button type="button" role="tab" aria-selected={teamView === 'ours'} className={teamView === 'ours' ? 'active ours' : ''} onClick={() => setTeamView('ours')}>{scenario.ours.name}</button>
          <button type="button" role="tab" aria-selected={teamView === 'opponent'} className={teamView === 'opponent' ? 'active opponent' : ''} onClick={() => setTeamView('opponent')}>{scenario.opponent.name}</button>
        </div>
      </header>

      <div className="pass-network-body">
        <div className={`network-pitch ${teamView}`} aria-label={`${team.name} 완료 패스 네트워크`}>
          <svg viewBox="0 0 120 80" preserveAspectRatio="xMidYMid meet" role="img">
            <rect x="1" y="1" width="118" height="78" className="network-pitch-line" />
            <line x1="60" y1="1" x2="60" y2="79" className="network-pitch-line" />
            <circle cx="60" cy="40" r="9.15" className="network-pitch-line" />
            <rect x="1" y="18" width="18" height="44" className="network-pitch-line" />
            <rect x="101" y="18" width="18" height="44" className="network-pitch-line" />
            {data.edges.map((edge) => {
              const from = nodeById.get(edge.from)
              const to = nodeById.get(edge.to)
              if (!from || !to) return null
              return <line key={`${edge.from}-${edge.to}`} x1={pitchX(from.x)} y1={pitchY(from.y)} x2={pitchX(to.x)} y2={pitchY(to.y)} className="network-edge" style={{ '--edge-width': `${Math.min(5, .7 + edge.count * .45)}px` } as React.CSSProperties}><title>{from.label} → {to.label} · {edge.count}회</title></line>
            })}
            {data.nodes.map((node) => {
              const radius = nodeRadius(node.involvements)
              return <g className="network-node" key={node.id}><title>{node.label} · 패스 관여 {node.involvements}회</title><circle cx={pitchX(node.x)} cy={pitchY(node.y)} r={radius} /><text x={pitchX(node.x)} y={pitchY(node.y) + radius + 3.2}>{node.label}</text></g>
            })}
          </svg>
          <span className="network-direction">공격 방향 →</span>
        </div>

        <aside className="network-insight">
          <div className="network-legend"><span><i className="node-sample" /> 노드 크기: 관여</span><span><i className="edge-sample" /> 선 굵기: 완료 횟수</span></div>
          <strong>{scenario.networkCopy.title}</strong>
          <p>{scenario.networkCopy.body}</p>
          <div className="network-kpis">
            <span><small>완료 패스</small><b>{data.completedPasses}</b></span>
            <span><small>연결 허브</small><b>{hub?.label ?? '—'}</b></span>
            <span><small>최다 연결</small><b>{strongestLabels}</b></span>
            <span><small>표시 기준</small><b>{data.minimumEdgeCount}회+</b></span>
          </div>
          <small className="source-note">평균 위치는 완료 패스의 시작·도착 좌표 · 교체 전후 선수 포함</small>
        </aside>
      </div>
    </section>
  )
}
