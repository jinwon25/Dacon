import { useMemo, useState } from 'react'
import type { GuidedScenario, PassNetworkData } from '../data/scenarios'

type TeamView = 'ours' | 'opponent'

const pitchX = (value: number) => value * 1.2
const pitchY = (value: number) => value * .8
const nodeRadius = (involvements: number, maximum: number) => 2.25 + Math.sqrt(involvements / Math.max(maximum, 1)) * 3.15

function strongestConnection(data: PassNetworkData) {
  return data.edges.reduce<PassNetworkData['edges'][number] | null>((best, edge) => !best || edge.count > best.count ? edge : best, null)
}

export default function PassNetwork({ scenario }: { scenario: GuidedScenario }) {
  const [teamView, setTeamView] = useState<TeamView>('ours')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const data = scenario.networks[teamView]
  const nodeMetricLabel = data.nodeMetricLabel ?? '패스 관여'
  const team = scenario[teamView]
  const nodeById = useMemo(() => new Map(data.nodes.map((node) => [node.id, node])), [data.nodes])
  const maximumNodeValue = Math.max(...data.nodes.map((node) => node.involvements), 1)
  const maximumEdgeValue = Math.max(...data.edges.map((edge) => edge.count), 1)
  const hub = data.nodes.reduce((best, node) => node.involvements > best.involvements ? node : best, data.nodes[0])
  const strongest = strongestConnection(data)
  const strongestLabels = strongest
    ? `${nodeById.get(strongest.from)?.label ?? strongest.from} → ${nodeById.get(strongest.to)?.label ?? strongest.to}`
    : '반복 연결 없음'
  const selectedNode = selectedId ? nodeById.get(selectedId) ?? null : null
  const selectedConnections = selectedId ? data.edges.filter((edge) => edge.from === selectedId || edge.to === selectedId) : []
  const selectedPartners = [...new Set(selectedConnections.map((edge) => nodeById.get(edge.from === selectedId ? edge.to : edge.from)?.label).filter(Boolean))]
  const selectedStrongest = selectedConnections.reduce<PassNetworkData['edges'][number] | null>((best, edge) => !best || edge.count > best.count ? edge : best, null)
  const selectedStrongestPartner = selectedStrongest
    ? nodeById.get(selectedStrongest.from === selectedId ? selectedStrongest.to : selectedStrongest.from)?.label ?? '—'
    : '반복 연결 없음'

  const changeTeam = (next: TeamView) => {
    setTeamView(next)
    setSelectedId(null)
  }

  return (
    <section className="pass-network panel">
      <header className="pass-network-heading">
        <div className="panel-title"><span>05</span><div><small>패스 연결망 · {scenario.windowLabel}</small><h2>패스 연결 구조를 읽어보세요</h2></div></div>
        <div className="network-tabs" role="tablist" aria-label="패스맵 팀 선택">
          <button type="button" role="tab" aria-selected={teamView === 'ours'} className={teamView === 'ours' ? 'active ours' : ''} onClick={() => changeTeam('ours')}>{scenario.ours.name}</button>
          <button type="button" role="tab" aria-selected={teamView === 'opponent'} className={teamView === 'opponent' ? 'active opponent' : ''} onClick={() => changeTeam('opponent')}>{scenario.opponent.name}</button>
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
              const connected = !selectedId || edge.from === selectedId || edge.to === selectedId
              return <line key={`${edge.from}-${edge.to}`} x1={pitchX(from.x)} y1={pitchY(from.y)} x2={pitchX(to.x)} y2={pitchY(to.y)} className={`network-edge ${selectedId ? connected ? 'highlighted' : 'dimmed' : ''}`} style={{ '--edge-width': `${.8 + edge.count / maximumEdgeValue * 4}px` } as React.CSSProperties}><title>{from.label} → {to.label} · {edge.count}회</title></line>
            })}
            {data.nodes.map((node) => {
              const radius = nodeRadius(node.involvements, maximumNodeValue)
              const connected = !selectedId || node.id === selectedId || selectedConnections.some((edge) => edge.from === node.id || edge.to === node.id)
              return <g role="button" tabIndex={0} aria-label={`${node.label} 패스 연결 보기`} className={`network-node ${node.id === selectedId ? 'selected' : ''} ${selectedId && !connected ? 'dimmed' : ''}`} key={node.id} onClick={() => setSelectedId((current) => current === node.id ? null : node.id)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setSelectedId((current) => current === node.id ? null : node.id) } }}><title>{node.label} · {nodeMetricLabel} {node.involvements}회</title><circle cx={pitchX(node.x)} cy={pitchY(node.y)} r={radius} /><text x={pitchX(node.x)} y={pitchY(node.y) + radius + 3.2}>{node.label}</text></g>
            })}
          </svg>
          <span className="network-direction">공격 방향 →</span>
        </div>

        <aside className="network-insight">
          <div className="network-legend"><span><i className="node-sample" /> 노드 크기: {nodeMetricLabel}</span><span><i className="edge-sample" /> 선 굵기: 완료 횟수</span></div>
          <strong>{selectedNode ? `${selectedNode.label} 연결 상세` : scenario.networkCopy.title}</strong>
          <p>{selectedNode ? `선택한 선수와 직접 연결된 패스만 강조했습니다. 노드를 다시 누르면 전체 연결망으로 돌아갑니다.` : scenario.networkCopy.body}</p>
          <div className="network-kpis" aria-live="polite">
            {selectedNode ? <>
            <span><small>{nodeMetricLabel}</small><b>{selectedNode.involvements}회</b></span>
            <span><small>직접 연결 선수</small><b>{selectedPartners.length}명</b></span>
            <span><small>가장 강한 연결</small><b>{selectedStrongestPartner}</b></span>
            <span><small>{scenario.networkPositionBasis ? '참조 위치' : '평균 위치'}</small><b>{Math.round(selectedNode.x)}, {Math.round(selectedNode.y)}</b></span>
            </> : <>
            <span><small>완료 패스</small><b>{data.completedPasses}</b></span>
            <span><small>연결 허브</small><b>{hub?.label ?? '—'}</b></span>
            <span><small>최다 연결</small><b>{strongestLabels}</b></span>
            <span><small>표시 기준</small><b>{data.minimumEdgeCount}회+</b></span>
            </>}
          </div>
          <small className="source-note">선수를 선택해 연결을 탐색하세요 · {scenario.networkPositionBasis ?? '평균 위치는 완료 패스 좌표 · 교체 전후 선수 포함'}</small>
        </aside>
      </div>
    </section>
  )
}
