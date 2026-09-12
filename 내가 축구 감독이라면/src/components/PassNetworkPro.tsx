import { useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent } from 'react'
import type { GuidedScenario, PassNetworkData, PassNetworkNode } from '../data/scenarios'

type TeamView = 'ours' | 'opponent'

const pitchX = (value: number) => value * 1.2
const pitchY = (value: number) => value * .8
const nodeRadius = (involvements: number, maximum: number) => 1.35 + Math.sqrt(involvements / Math.max(maximum, 1)) * 1.65
const normalizeName = (value: string) => value.replace(/[.\s·-]/g, '').toLocaleLowerCase()

function strongestConnection(data: PassNetworkData) {
  return data.edges.reduce<PassNetworkData['edges'][number] | null>((best, edge) => !best || edge.count > best.count ? edge : best, null)
}

function nodeCode(node: PassNetworkNode, scenario: GuidedScenario, teamView: TeamView) {
  if (teamView === 'ours') {
    const normalizedLabel = normalizeName(node.label)
    const player = scenario.squad.find((item) => {
      const candidates = [item.name, item.shortName].map(normalizeName)
      return candidates.some((candidate) => candidate === normalizedLabel || candidate.includes(normalizedLabel) || normalizedLabel.includes(candidate))
    })
    if (player) return { code: String(player.number), reserve: !player.onPitch }
  }

  const compact = node.label.replace(/[.\s·-]/g, '')
  return { code: compact.slice(0, compact.length > 3 ? 2 : 3).toUpperCase(), reserve: false }
}

function edgePath(from: PassNetworkNode, to: PassNetworkNode, fromRadius: number, toRadius: number, reverse: boolean) {
  const startX = pitchX(from.x)
  const startY = pitchY(from.y)
  const endX = pitchX(to.x)
  const endY = pitchY(to.y)
  const dx = endX - startX
  const dy = endY - startY
  const length = Math.max(Math.hypot(dx, dy), 1)
  const unitX = dx / length
  const unitY = dy / length
  const x1 = startX + unitX * (fromRadius + .8)
  const y1 = startY + unitY * (fromRadius + .8)
  const x2 = endX - unitX * (toRadius + 1.25)
  const y2 = endY - unitY * (toRadius + 1.25)

  if (!reverse) return `M ${x1.toFixed(2)} ${y1.toFixed(2)} L ${x2.toFixed(2)} ${y2.toFixed(2)}`

  const direction = from.id.localeCompare(to.id) > 0 ? 1 : -1
  const bend = Math.min(4.2, Math.max(2.5, length * .08)) * direction
  const controlX = (x1 + x2) / 2 - unitY * bend
  const controlY = (y1 + y2) / 2 + unitX * bend
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} Q ${controlX.toFixed(2)} ${controlY.toFixed(2)} ${x2.toFixed(2)} ${y2.toFixed(2)}`
}

interface NetworkPitchProps {
  scenario: GuidedScenario
  teamView: TeamView
  active: boolean
  selectedId: string | null
  onActivate: (teamView: TeamView) => void
  onSelect: (teamView: TeamView, nodeId: string) => void
}

function NetworkPitch({ scenario, teamView, active, selectedId, onActivate, onSelect }: NetworkPitchProps) {
  const data = scenario.networks[teamView]
  const team = scenario[teamView]
  const nodeMetricLabel = data.nodeMetricLabel ?? '패스 관여'
  const nodeById = useMemo(() => new Map(data.nodes.map((node) => [node.id, node])), [data.nodes])
  const maximumNodeValue = Math.max(...data.nodes.map((node) => node.involvements), 1)
  const maximumEdgeValue = Math.max(...data.edges.map((edge) => edge.count), 1)
  const radii = useMemo(() => new Map(data.nodes.map((node) => [node.id, nodeRadius(node.involvements, maximumNodeValue)])), [data.nodes, maximumNodeValue])
  const edgeKeys = useMemo(() => new Set(data.edges.map((edge) => `${edge.from}→${edge.to}`)), [data.edges])
  const [focusedId, setFocusedId] = useState(data.nodes[0]?.id ?? '')
  const nodeRefs = useRef(new Map<string, SVGGElement>())
  const activeSelectedId = active ? selectedId : null
  const selectedConnections = activeSelectedId ? data.edges.filter((edge) => edge.from === activeSelectedId || edge.to === activeSelectedId) : []
  const connectedIds = new Set(selectedConnections.flatMap((edge) => [edge.from, edge.to]))

  useEffect(() => {
    setFocusedId(data.nodes[0]?.id ?? '')
  }, [data.nodes, scenario.id])

  const focusNode = (nextIndex: number) => {
    const next = data.nodes[(nextIndex + data.nodes.length) % data.nodes.length]
    if (!next) return
    setFocusedId(next.id)
    window.requestAnimationFrame(() => nodeRefs.current.get(next.id)?.focus())
  }

  const handleNodeKeyDown = (event: ReactKeyboardEvent<SVGGElement>, nodeId: string) => {
    const currentIndex = data.nodes.findIndex((node) => node.id === nodeId)
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      onSelect(teamView, nodeId)
      return
    }
    if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault()
      focusNode(event.key === 'Home' ? 0 : data.nodes.length - 1)
      return
    }
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp' || event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      event.preventDefault()
      focusNode(currentIndex + (event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1))
    }
  }

  const titleId = `network-title-${scenario.id}-${teamView}`
  const descriptionId = `network-description-${scenario.id}-${teamView}`

  return (
    <article className={`network-pitch-panel ${teamView} ${active ? 'active mobile-active' : ''}`}>
      <header>
        <button type="button" className="network-team-select" aria-pressed={active} onClick={() => onActivate(teamView)}><span>{team.short}</span><strong>{team.name}</strong></button>
        <small>{data.completedPasses}회 완료</small>
      </header>
      <div className={`network-pitch ${teamView} ${scenario.networkPositionBasis ? 'reference-position' : 'average-position'}`}>
        <svg viewBox="0 0 120 80" preserveAspectRatio="xMidYMid meet" role="group" aria-labelledby={`${titleId} ${descriptionId}`}>
          <title id={titleId}>{team.name} 완료 패스 연결 구조</title>
          <desc id={descriptionId}>{scenario.windowLabel}, {nodeMetricLabel}에 따라 노드 크기가 달라지며 같은 방향 완료 패스가 {data.minimumEdgeCount}회 이상인 연결을 표시합니다. 방향키로 선수를 탐색할 수 있습니다.</desc>
          <defs>
            <marker id={`network-arrow-${scenario.id}-${teamView}`} markerUnits="userSpaceOnUse" markerWidth="2.7" markerHeight="2.7" refX="2.45" refY="1.35" viewBox="0 0 2.7 2.7" orient="auto">
              <path d="M0,0 L2.7,1.35 L0,2.7 Z" />
            </marker>
          </defs>
          <rect x="1" y="1" width="118" height="78" className="network-pitch-line" />
          <line x1="60" y1="1" x2="60" y2="79" className="network-pitch-line" />
          <circle cx="60" cy="40" r="9.15" className="network-pitch-line" />
          <circle cx="60" cy="40" r=".7" className="network-pitch-line network-center-dot" />
          <rect x="1" y="18" width="18" height="44" className="network-pitch-line" />
          <rect x="101" y="18" width="18" height="44" className="network-pitch-line" />
          {data.edges.map((edge) => {
            const from = nodeById.get(edge.from)
            const to = nodeById.get(edge.to)
            if (!from || !to) return null
            const connected = !activeSelectedId || edge.from === activeSelectedId || edge.to === activeSelectedId
            const reverse = edgeKeys.has(`${edge.to}→${edge.from}`)
            const width = .55 + edge.count / maximumEdgeValue * 2.05
            const opacity = .26 + edge.count / maximumEdgeValue * .54
            return <path key={`${edge.from}-${edge.to}`} d={edgePath(from, to, radii.get(from.id) ?? 2, radii.get(to.id) ?? 2, reverse)} markerEnd={`url(#network-arrow-${scenario.id}-${teamView})`} className={`network-edge ${activeSelectedId ? connected ? 'highlighted' : 'dimmed' : ''}`} style={{ '--edge-width': `${width.toFixed(2)}px`, '--edge-opacity': opacity.toFixed(2) } as CSSProperties}><title>{from.label} → {to.label} · {edge.count}회</title></path>
          })}
          {data.nodes.map((node) => {
            const radius = radii.get(node.id) ?? 2
            const connected = !activeSelectedId || connectedIds.has(node.id)
            const selected = activeSelectedId === node.id
            const marker = nodeCode(node, scenario, teamView)
            return <g
              ref={(element) => { if (element) nodeRefs.current.set(node.id, element); else nodeRefs.current.delete(node.id) }}
              role="button"
              tabIndex={focusedId === node.id ? 0 : -1}
              aria-pressed={selected}
              aria-label={`${node.label}, ${nodeMetricLabel} ${node.involvements}회${marker.reserve ? ', 교체 출전 선수' : ''}`}
              className={`network-node ${selected ? 'selected' : ''} ${activeSelectedId && !connected ? 'dimmed' : ''} ${marker.reserve ? 'reserve' : ''}`}
              key={node.id}
              onFocus={() => { setFocusedId(node.id); onActivate(teamView) }}
              onClick={() => onSelect(teamView, node.id)}
              onKeyDown={(event) => handleNodeKeyDown(event, node.id)}
            >
              <title>{node.label} · {nodeMetricLabel} {node.involvements}회</title>
              <circle className="network-node-hit" cx={pitchX(node.x)} cy={pitchY(node.y)} r={radius} />
              <circle className="network-node-visual" cx={pitchX(node.x)} cy={pitchY(node.y)} r={radius} />
              <text className="network-node-code" x={pitchX(node.x)} y={pitchY(node.y) + .85}>{marker.code}</text>
              {selected && <text className="network-node-name" x={pitchX(node.x)} y={pitchY(node.y) + radius + 3.4}>{node.label}</text>}
            </g>
          })}
        </svg>
        <span className="network-direction">공격 방향 →</span>
      </div>
    </article>
  )
}

export default function PassNetworkPro({ scenario }: { scenario: GuidedScenario }) {
  const [teamView, setTeamView] = useState<TeamView>('ours')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const data = scenario.networks[teamView]
  const nodeMetricLabel = data.nodeMetricLabel ?? '패스 관여'
  const nodeById = useMemo(() => new Map(data.nodes.map((node) => [node.id, node])), [data.nodes])
  const hub = data.nodes.reduce((best, node) => node.involvements > best.involvements ? node : best, data.nodes[0])
  const strongest = strongestConnection(data)
  const strongestLabels = strongest
    ? `${nodeById.get(strongest.from)?.label ?? strongest.from} → ${nodeById.get(strongest.to)?.label ?? strongest.to}`
    : '반복 연결 없음'
  const selectedNode = selectedId ? nodeById.get(selectedId) ?? null : null
  const selectedConnections = selectedId ? data.edges.filter((edge) => edge.from === selectedId || edge.to === selectedId) : []
  const selectedPartners = new Set(selectedConnections.map((edge) => edge.from === selectedId ? edge.to : edge.from))
  const selectedInbound = selectedConnections.reduce((total, edge) => total + (edge.to === selectedId ? edge.count : 0), 0)
  const selectedOutbound = selectedConnections.reduce((total, edge) => total + (edge.from === selectedId ? edge.count : 0), 0)

  useEffect(() => {
    setTeamView('ours')
    setSelectedId(null)
  }, [scenario.id])

  const changeTeam = (next: TeamView) => {
    setTeamView(next)
    setSelectedId(null)
  }

  const activateTeam = (next: TeamView) => {
    if (next === teamView) return
    setTeamView(next)
    setSelectedId(null)
  }

  const selectNode = (nextTeam: TeamView, nodeId: string) => {
    setTeamView(nextTeam)
    setSelectedId((current) => nextTeam === teamView && current === nodeId ? null : nodeId)
  }

  return (
    <section className="pass-network panel" id="pass-network">
      <header className="pass-network-heading">
        <div className="panel-title"><span>PASS</span><div><small>패스 연결망 · {scenario.windowLabel}</small><h2>양 팀의 연결 구조를 비교하세요</h2></div></div>
        <div className="network-tabs" role="group" aria-label="모바일 패스맵 팀 선택">
          <button type="button" aria-pressed={teamView === 'ours'} className={teamView === 'ours' ? 'active ours' : ''} onClick={() => changeTeam('ours')}>{scenario.ours.name}</button>
          <button type="button" aria-pressed={teamView === 'opponent'} className={teamView === 'opponent' ? 'active opponent' : ''} onClick={() => changeTeam('opponent')}>{scenario.opponent.name}</button>
        </div>
      </header>

      <div className="network-context-strip" aria-label="패스 네트워크 분석 기준">
        <span><small>구간</small><b>{scenario.windowLabel}</b></span>
        <span className={scenario.networkPositionBasis ? 'caution' : ''}><small>위치</small><b>{scenario.networkPositionBasis ? '참조 배치' : '평균 위치'}</b></span>
        <span><small>노드</small><b>{nodeMetricLabel}</b></span>
        <span><small>연결</small><b>같은 방향 완료</b></span>
        <span><small>표시 {scenario.ours.short}/{scenario.opponent.short}</small><b>{scenario.networks.ours.minimumEdgeCount}+ / {scenario.networks.opponent.minimumEdgeCount}+</b></span>
        <span><small>범위</small><b>교체 선수 포함</b></span>
      </div>

      <div className="pass-network-body">
        <div className="network-comparison" aria-label="양 팀 패스 네트워크 비교">
          <NetworkPitch scenario={scenario} teamView="ours" active={teamView === 'ours'} selectedId={selectedId} onActivate={activateTeam} onSelect={selectNode} />
          <NetworkPitch scenario={scenario} teamView="opponent" active={teamView === 'opponent'} selectedId={selectedId} onActivate={activateTeam} onSelect={selectNode} />
        </div>

        <aside className="network-insight">
          <div className="network-legend"><span><i className="node-sample" /> 크기: {nodeMetricLabel}</span><span><i className="edge-sample" /> 굵기·명도: 완료 횟수</span><span><i className="reserve-sample" /> 점선: 교체 출전</span></div>
          <strong>{selectedNode ? `${selectedNode.label} 연결 상세` : scenario.networkCopy.title}</strong>
          <p>{selectedNode ? '선택한 선수의 송신·수신 연결만 강조했습니다. 방향키로 다른 선수를 탐색할 수 있습니다.' : scenario.networkCopy.body}</p>
          <div className="network-kpis" aria-live="polite">
            {selectedNode ? <>
              <span><small>{nodeMetricLabel}</small><b>{selectedNode.involvements}회</b></span>
              <span><small>연결 선수</small><b>{selectedPartners.size}명</b></span>
              <span><small>보낸 패스</small><b>{selectedOutbound}회</b></span>
              <span><small>받은 패스</small><b>{selectedInbound}회</b></span>
            </> : <>
              <span><small>완료 패스</small><b>{data.completedPasses}</b></span>
              <span><small>연결 허브</small><b>{hub?.label ?? '—'}</b></span>
              <span><small>최다 방향</small><b>{strongestLabels}</b></span>
              <span><small>표시 기준</small><b>{data.minimumEdgeCount}회+</b></span>
            </>}
          </div>
          <small className="source-note">{scenario.networkPositionBasis ?? '평균 위치는 완료 패스 좌표를 사용합니다.'} · 상위 연결만 표시하므로 선이 없는 선수도 패스 기록이 있을 수 있습니다.</small>
        </aside>
      </div>
    </section>
  )
}
