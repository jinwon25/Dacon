import { useEffect, useMemo, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import { toPng } from 'html-to-image'
import { createGenericSquad, createOpponentSquad } from '../data/generic'
import { formationPositions, roleOptions } from '../data/match'
import type { FormationKey, Player, Tactics } from '../types'

const formations: FormationKey[] = ['4-3-3', '4-2-3-1', '4-4-2', '4-1-4-1', '4-3-1-2', '3-4-3', '5-3-2']
const storageKey = 'retactic-universal-studio-v1'

interface MatchContext {
  teamName: string
  opponentName: string
  minute: number
  ourScore: number
  theirScore: number
  phase: '경기 전' | '전반전' | '하프타임' | '후반전' | '연장전'
  objective: '균형 유지' | '득점 필요' | '리드 보호' | '압박 탈출' | '상대 역습 차단'
}

interface SavedStudio {
  context: MatchContext
  formation: FormationKey
  tactics: Tactics
  squad: Player[]
  opponentFormation?: FormationKey
  opponentVisible?: boolean
  opponents?: Player[]
  planSlots?: PlanSlots
  routes?: TacticalRoute[]
  scenes?: TacticalScene[]
  activeSceneId?: string
}

interface TacticalRoute {
  id: string
  type: 'pass' | 'run'
  actorId: string
  from: { x: number; y: number }
  to: { x: number; y: number }
}

interface TacticalScene {
  id: string
  name: string
  squad: Player[]
  opponents: Player[]
  formation: FormationKey
  opponentFormation: FormationKey
  routes: TacticalRoute[]
}

interface StudioTemplate {
  id: string
  title: string
  note: string
  context: MatchContext
  formation: FormationKey
  opponentFormation: FormationKey
  tactics: Tactics
  positions?: Array<[number, number]>
  routes: Omit<TacticalRoute, 'id'>[]
}

interface CoachPreview {
  playerId: string
  x: number
  y: number
  title: string
}

interface PlanSnapshot {
  savedAt: string
  context: MatchContext
  formation: FormationKey
  tactics: Tactics
  squad: Player[]
  opponentFormation: FormationKey
  opponents: Player[]
  routes?: TacticalRoute[]
  teamWidth: number
  defensiveLine: number
  transitionRisk: number
}

type PlanSlots = { A?: PlanSnapshot; B?: PlanSnapshot }
type LayoutSnapshot = { squad: Player[]; opponents: Player[]; routes: TacticalRoute[]; formation: FormationKey; opponentFormation: FormationKey }
type BoardTool = 'move' | 'pass' | 'run'
type RightPanelTab = 'player' | 'analysis' | 'plan'

const defaultContext: MatchContext = {
  teamName: '우리 팀',
  opponentName: '상대 팀',
  minute: 0,
  ourScore: 0,
  theirScore: 0,
  phase: '경기 전',
  objective: '균형 유지',
}

const defaultTactics: Tactics = { pressing: 50, width: 55, tempo: 50, risk: 45 }
const clamp = (value: number, min = 0, max = 100) => Math.min(max, Math.max(min, value))
const distance = (a: { x: number; y: number }, b: { x: number; y: number }) => Math.hypot(a.x - b.x, a.y - b.y)
const arrowPoints = (route: TacticalRoute) => {
  const angle = Math.atan2(route.to.y - route.from.y, route.to.x - route.from.x)
  const size = 2.2
  const wing = 1.2
  const baseX = route.to.x - Math.cos(angle) * size
  const baseY = route.to.y - Math.sin(angle) * size
  const perpendicularX = Math.cos(angle + Math.PI / 2) * wing
  const perpendicularY = Math.sin(angle + Math.PI / 2) * wing
  return `${route.to.x},${route.to.y} ${baseX + perpendicularX},${baseY + perpendicularY} ${baseX - perpendicularX},${baseY - perpendicularY}`
}
const clonePlayers = (players: Player[]) => players.map((player) => ({ ...player }))
const cloneRoutes = (routes: TacticalRoute[]) => routes.map((route) => ({ ...route, from: { ...route.from }, to: { ...route.to } }))
const sceneNames = ['초기 배치', '빌드업', '공격 전개', '마무리', '전환 대비']

const studioTemplates: StudioTemplate[] = [
  {
    id: 'korea-4231',
    title: '대한민국 4-2-3-1',
    note: '두 명의 6번과 넓은 2선',
    context: { ...defaultContext, teamName: '대한민국', opponentName: '상대 팀', objective: '균형 유지' },
    formation: '4-2-3-1', opponentFormation: '4-3-3',
    tactics: { pressing: 58, width: 64, tempo: 61, risk: 52 },
    routes: [
      { type: 'pass', actorId: 'generic-lcm', from: { x: 40, y: 54 }, to: { x: 18, y: 31 } },
      { type: 'run', actorId: 'generic-lw', from: { x: 20, y: 35 }, to: { x: 18, y: 14 } },
    ],
  },
  {
    id: 'argentina-433',
    title: '아르헨티나 4-3-3',
    note: '중앙 과부하와 하프스페이스',
    context: { ...defaultContext, teamName: '아르헨티나', opponentName: '상대 팀', objective: '압박 탈출' },
    formation: '4-3-3', opponentFormation: '4-2-3-1',
    tactics: { pressing: 55, width: 58, tempo: 54, risk: 48 },
    routes: [
      { type: 'pass', actorId: 'generic-dm', from: { x: 50, y: 62 }, to: { x: 37, y: 43 } },
      { type: 'run', actorId: 'generic-rw', from: { x: 80, y: 31 }, to: { x: 65, y: 16 } },
    ],
  },
  {
    id: 'france-counter',
    title: '프랑스 역습 전술',
    note: '낮은 블록 뒤 빠른 측면 전환',
    context: { ...defaultContext, teamName: '프랑스', opponentName: '상대 팀', phase: '후반전', minute: 70, objective: '상대 역습 차단' },
    formation: '4-2-3-1', opponentFormation: '4-3-3',
    tactics: { pressing: 38, width: 68, tempo: 82, risk: 57 },
    routes: [
      { type: 'pass', actorId: 'generic-dm', from: { x: 50, y: 59 }, to: { x: 82, y: 38 } },
      { type: 'run', actorId: 'generic-rw', from: { x: 80, y: 38 }, to: { x: 88, y: 13 } },
      { type: 'run', actorId: 'generic-st', from: { x: 50, y: 24 }, to: { x: 61, y: 11 } },
    ],
  },
  {
    id: 'corner-attack',
    title: '코너킥 공격',
    note: '니어 유인 후 파포스트 공략',
    context: { ...defaultContext, teamName: '우리 팀', opponentName: '상대 팀', phase: '후반전', minute: 88, objective: '득점 필요' },
    formation: '4-3-3', opponentFormation: '5-3-2',
    tactics: { pressing: 62, width: 72, tempo: 68, risk: 74 },
    positions: [[50, 88], [78, 62], [58, 28], [43, 22], [22, 55], [50, 55], [68, 18], [34, 15], [86, 8], [51, 10], [18, 12]],
    routes: [
      { type: 'pass', actorId: 'generic-rw', from: { x: 86, y: 8 }, to: { x: 48, y: 10 } },
      { type: 'run', actorId: 'generic-st', from: { x: 51, y: 10 }, to: { x: 39, y: 8 } },
      { type: 'run', actorId: 'generic-lcm', from: { x: 34, y: 15 }, to: { x: 62, y: 9 } },
    ],
  },
]

function createScene(id: string, name: string, squad: Player[], opponents: Player[], formation: FormationKey, opponentFormation: FormationKey, routes: TacticalRoute[]): TacticalScene {
  return { id, name, squad: clonePlayers(squad), opponents: clonePlayers(opponents), formation, opponentFormation, routes: cloneRoutes(routes) }
}

function loadSavedStudio(): SavedStudio | null {
  try {
    const raw = window.localStorage.getItem(storageKey)
    return raw ? JSON.parse(raw) as SavedStudio : null
  } catch {
    return null
  }
}

export default function UniversalStudio() {
  const saved = useRef(loadSavedStudio()).current
  const [context, setContext] = useState<MatchContext>(saved?.context ?? defaultContext)
  const [formation, setFormation] = useState<FormationKey>(saved?.formation ?? '4-3-3')
  const [tactics, setTactics] = useState<Tactics>(saved?.tactics ?? defaultTactics)
  const [squad, setSquad] = useState<Player[]>(saved?.squad ?? createGenericSquad())
  const [opponentFormation, setOpponentFormation] = useState<FormationKey>(saved?.opponentFormation ?? '4-3-3')
  const [opponentVisible, setOpponentVisible] = useState(saved?.opponentVisible ?? true)
  const [opponents, setOpponents] = useState<Player[]>(saved?.opponents ?? createOpponentSquad())
  const [planSlots, setPlanSlots] = useState<PlanSlots>(saved?.planSlots ?? {})
  const [routes, setRoutes] = useState<TacticalRoute[]>(saved?.routes ?? saved?.scenes?.[0]?.routes ?? [])
  const initialScenes = useRef<TacticalScene[]>(saved?.scenes?.length
    ? saved.scenes
    : [createScene('scene-1', sceneNames[0], squad, opponents, formation, opponentFormation, saved?.routes ?? [])])
  const [scenes, setScenes] = useState<TacticalScene[]>(initialScenes.current)
  const [activeSceneId, setActiveSceneId] = useState(saved?.activeSceneId ?? initialScenes.current[0].id)
  const [layoutHistory, setLayoutHistory] = useState<LayoutSnapshot[]>([])
  const [selectedId, setSelectedId] = useState<string>('generic-st')
  const [draggingId, setDraggingId] = useState<string | null>(null)
  const [boardTool, setBoardTool] = useState<BoardTool>('move')
  const [rightPanelTab, setRightPanelTab] = useState<RightPanelTab>('player')
  const [isPlaying, setIsPlaying] = useState(false)
  const [coachPreview, setCoachPreview] = useState<CoachPreview | null>(null)
  const [hasDragged, setHasDragged] = useState(false)
  const [lastSaved, setLastSaved] = useState('방금')
  const [exportStatus, setExportStatus] = useState<'idle' | 'working' | 'done' | 'error'>('idle')
  const pitchRef = useRef<HTMLDivElement>(null)
  const exportRef = useRef<HTMLDivElement>(null)
  const playbackTimerRef = useRef<number | null>(null)

  const onPitch = squad.filter((player) => player.onPitch)
  const bench = squad.filter((player) => !player.onPitch)
  const selectedPlayer = squad.find((player) => player.id === selectedId) ?? null
  const activeScene = scenes.find((scene) => scene.id === activeSceneId) ?? scenes[0]
  const previewPlayer = coachPreview ? onPitch.find((player) => player.id === coachPreview.playerId) ?? null : null
  const ghostPlayers = activeScene?.squad.filter((player) => player.onPitch && onPitch.some((current) => current.id === player.id && distance(current, player) > 1.5)) ?? []
  const today = new Intl.DateTimeFormat('en-CA').format(new Date())

  useEffect(() => {
    const timer = window.setTimeout(() => {
      window.localStorage.setItem(storageKey, JSON.stringify({ context, formation, tactics, squad, opponentFormation, opponentVisible, opponents, planSlots, routes, scenes, activeSceneId }))
      setLastSaved(new Intl.DateTimeFormat('ko', { hour: '2-digit', minute: '2-digit' }).format(new Date()))
    }, 350)
    return () => window.clearTimeout(timer)
  }, [activeSceneId, context, formation, tactics, squad, opponentFormation, opponentVisible, opponents, planSlots, routes, scenes])

  useEffect(() => () => {
    if (playbackTimerRef.current !== null) window.clearTimeout(playbackTimerRef.current)
  }, [])

  const diagnostics = useMemo(() => {
    const defenders = onPitch.filter((player) => player.position === 'DF')
    const teamWidth = Math.round(Math.max(...onPitch.map((player) => player.x)) - Math.min(...onPitch.map((player) => player.x)))
    const defensiveLine = defenders.length ? Math.round(defenders.reduce((sum, player) => sum + player.y, 0) / defenders.length) : 75
    const forwardSupport = onPitch.filter((player) => player.y < 45).length
    const transitionRisk = Math.round(tactics.risk * 0.42 + tactics.pressing * 0.28 + (formation === '3-4-3' ? 18 : formation === '5-3-2' ? -8 : 4))
    const passOptions = selectedPlayer?.onPitch
      ? onPitch.filter((player) => player.id !== selectedPlayer.id && distance(player, selectedPlayer) <= 34).length
      : 0
    const nearestDistances = onPitch.map((player) => Math.min(...onPitch.filter((target) => target.id !== player.id).map((target) => distance(player, target))))
    const compactness = Math.round(nearestDistances.reduce((sum, value) => sum + value, 0) / nearestDistances.length)
    const attackingOverload = onPitch.filter((player) => player.y < 38).length - opponents.filter((player) => player.y < 38).length

    return {
      teamWidth,
      defensiveLine,
      forwardSupport,
      passOptions,
      compactness,
      attackingOverload,
      transitionRisk: clamp(transitionRisk),
      checks: [
        { tone: teamWidth >= 58 ? 'good' : 'warn', title: '공격 폭', detail: teamWidth >= 58 ? `좌우 간격 ${teamWidth} · 측면 활용 가능` : `좌우 간격 ${teamWidth} · 공격 간격이 좁습니다` },
        { tone: passOptions >= 3 ? 'good' : passOptions >= 2 ? 'warn' : 'risk', title: '패스 선택지', detail: selectedPlayer?.onPitch ? `${selectedPlayer.shortName} 주변 34m 이내 ${passOptions}명` : '필드 선수를 선택해 연결 옵션을 확인하세요' },
        { tone: transitionRisk < 62 ? 'good' : 'risk', title: '전환 안전', detail: transitionRisk < 62 ? '공을 잃은 뒤 복귀 가능한 범위' : '압박과 위험 감수가 함께 높습니다' },
        { tone: compactness <= 23 ? 'good' : 'warn', title: '선수 간격', detail: `가장 가까운 동료까지 평균 ${compactness}m` },
        { tone: attackingOverload >= 0 ? 'good' : 'warn', title: '공격 지역 수적 관계', detail: attackingOverload >= 0 ? `동수 이상 · 차이 ${signed(attackingOverload)}` : `상대 수비가 ${Math.abs(attackingOverload)}명 더 많습니다` },
      ],
    }
  }, [formation, onPitch, opponents, selectedPlayer, tactics])

  const sceneBaseline = useMemo(() => {
    const players = activeScene?.squad.filter((player) => player.onPitch) ?? []
    const defenders = players.filter((player) => player.position === 'DF')
    return {
      teamWidth: players.length ? Math.round(Math.max(...players.map((player) => player.x)) - Math.min(...players.map((player) => player.x))) : diagnostics.teamWidth,
      defensiveLine: defenders.length ? Math.round(defenders.reduce((sum, player) => sum + player.y, 0) / defenders.length) : diagnostics.defensiveLine,
    }
  }, [activeScene, diagnostics.defensiveLine, diagnostics.teamWidth])

  const applyFormation = (next: FormationKey) => {
    pushLayoutHistory()
    setCoachPreview(null)
    setFormation(next)
    setSquad((current) => current.map((player) => {
      if (!player.onPitch || player.slot === null) return player
      return { ...player, ...formationPositions[next][player.slot] }
    }))
    setHasDragged(false)
  }

  const handlePointerDown = (event: ReactPointerEvent<HTMLButtonElement>, playerId: string) => {
    event.stopPropagation()
    setSelectedId(playerId)
    setRightPanelTab('player')
    setCoachPreview(null)
    if (boardTool !== 'move' || isPlaying) return
    pushLayoutHistory()
    event.currentTarget.setPointerCapture(event.pointerId)
    setDraggingId(playerId)
  }

  const handlePointerMove = (event: ReactPointerEvent<HTMLButtonElement>, playerId: string) => {
    if (draggingId !== playerId || !pitchRef.current) return
    const rect = pitchRef.current.getBoundingClientRect()
    const x = clamp((event.clientX - rect.left) / rect.width * 100, 7, 93)
    const y = clamp((event.clientY - rect.top) / rect.height * 100, 6, 94)
    setSquad((current) => current.map((player) => player.id === playerId ? { ...player, x, y } : player))
    setHasDragged(true)
  }

  const handlePointerUp = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
    setDraggingId(null)
  }

  const handlePitchPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (boardTool === 'move') {
      setRightPanelTab('analysis')
      setCoachPreview(null)
      return
    }
    if (isPlaying || !pitchRef.current || !selectedPlayer?.onPitch) return
    const rect = pitchRef.current.getBoundingClientRect()
    const to = {
      x: clamp((event.clientX - rect.left) / rect.width * 100, 4, 96),
      y: clamp((event.clientY - rect.top) / rect.height * 100, 3, 97),
    }
    if (distance(selectedPlayer, to) < 3) return
    const route: TacticalRoute = {
      id: `route-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      type: boardTool,
      actorId: selectedPlayer.id,
      from: { x: selectedPlayer.x, y: selectedPlayer.y },
      to,
    }
    setRoutes((current) => [...current.slice(-11), route])
    setHasDragged(true)
  }

  const handleOpponentPointerMove = (event: ReactPointerEvent<HTMLButtonElement>, playerId: string) => {
    if (draggingId !== playerId || !pitchRef.current) return
    const rect = pitchRef.current.getBoundingClientRect()
    const x = clamp((event.clientX - rect.left) / rect.width * 100, 7, 93)
    const y = clamp((event.clientY - rect.top) / rect.height * 100, 6, 94)
    setOpponents((current) => current.map((player) => player.id === playerId ? { ...player, x, y } : player))
    setHasDragged(true)
  }

  const applyOpponentFormation = (next: FormationKey) => {
    pushLayoutHistory()
    setCoachPreview(null)
    setOpponentFormation(next)
    setOpponents(createOpponentSquad(next))
  }

  const updatePlayer = (changes: Partial<Player>) => {
    if (!selectedPlayer) return
    setSquad((current) => current.map((player) => player.id === selectedPlayer.id ? { ...player, ...changes } : player))
  }

  const swapWithBench = (incomingId: string) => {
    if (!selectedPlayer?.onPitch || selectedPlayer.slot === null) {
      setSelectedId(incomingId)
      return
    }
    pushLayoutHistory()
    const slot = selectedPlayer.slot
    const coordinate = { x: selectedPlayer.x, y: selectedPlayer.y }
    setSquad((current) => current.map((player) => {
      if (player.id === selectedPlayer.id) return { ...player, onPitch: false, slot: null, x: 0, y: 0 }
      if (player.id === incomingId) return { ...player, onPitch: true, slot, ...coordinate }
      return player
    }))
    setSelectedId(incomingId)
  }

  const resetStudio = () => {
    if (!window.confirm('현재 전술안을 지우고 기본 4-3-3으로 초기화할까요?')) return
    setContext(defaultContext)
    setFormation('4-3-3')
    setTactics(defaultTactics)
    setSquad(createGenericSquad())
    setOpponentFormation('4-3-3')
    setOpponentVisible(true)
    setOpponents(createOpponentSquad())
    setSelectedId('generic-st')
    setHasDragged(false)
    setPlanSlots({})
    setRoutes([])
    const resetSquad = createGenericSquad()
    const resetOpponents = createOpponentSquad()
    const resetScene = createScene('scene-1', sceneNames[0], resetSquad, resetOpponents, '4-3-3', '4-3-3', [])
    setScenes([resetScene])
    setActiveSceneId(resetScene.id)
    setBoardTool('move')
    setRightPanelTab('player')
    setCoachPreview(null)
    setLayoutHistory([])
    window.localStorage.removeItem(storageKey)
  }

  function pushLayoutHistory() {
    const snapshot: LayoutSnapshot = {
      squad: squad.map((player) => ({ ...player })),
      opponents: opponents.map((player) => ({ ...player })),
      routes: cloneRoutes(routes),
      formation,
      opponentFormation,
    }
    setLayoutHistory((current) => [...current.slice(-9), snapshot])
  }

  const undoLayout = () => {
    const previous = layoutHistory[layoutHistory.length - 1]
    if (!previous) return
    setSquad(previous.squad)
    setOpponents(previous.opponents)
    setRoutes(previous.routes)
    setFormation(previous.formation)
    setOpponentFormation(previous.opponentFormation)
    setLayoutHistory((current) => current.slice(0, -1))
  }

  const captureCurrentScene = (id = activeSceneId, name = activeScene?.name ?? sceneNames[0]) =>
    createScene(id, name, squad, opponents, formation, opponentFormation, routes)

  const restoreScene = (scene: TacticalScene, trackHistory = true) => {
    if (trackHistory) pushLayoutHistory()
    setSquad(clonePlayers(scene.squad))
    setOpponents(clonePlayers(scene.opponents))
    setFormation(scene.formation)
    setOpponentFormation(scene.opponentFormation)
    setRoutes(cloneRoutes(scene.routes))
    setActiveSceneId(scene.id)
    setCoachPreview(null)
  }

  const saveCurrentScene = () => {
    const snapshot = captureCurrentScene()
    setScenes((current) => current.map((scene) => scene.id === activeSceneId ? snapshot : scene))
  }

  const addScene = () => {
    if (scenes.length >= 5) return
    const currentSnapshot = captureCurrentScene()
    const id = `scene-${Date.now()}`
    const next = createScene(id, sceneNames[scenes.length] ?? `장면 ${scenes.length + 1}`, squad, opponents, formation, opponentFormation, routes)
    setScenes((current) => [...current.map((scene) => scene.id === activeSceneId ? currentSnapshot : scene), next])
    setActiveSceneId(id)
  }

  const deleteScene = (id: string) => {
    if (scenes.length <= 1) return
    const remaining = scenes.filter((scene) => scene.id !== id)
    setScenes(remaining)
    if (activeSceneId === id) restoreScene(remaining[Math.max(0, remaining.length - 1)], false)
  }

  const stopPlayback = () => {
    if (playbackTimerRef.current !== null) window.clearTimeout(playbackTimerRef.current)
    playbackTimerRef.current = null
    setIsPlaying(false)
  }

  const playScenes = () => {
    if (scenes.length < 2) return
    stopPlayback()
    const currentSnapshot = captureCurrentScene()
    const frames = scenes.map((scene) => scene.id === activeSceneId ? currentSnapshot : scene)
    setScenes(frames)
    setIsPlaying(true)
    setBoardTool('move')
    const advance = (index: number) => {
      restoreScene(frames[index], false)
      playbackTimerRef.current = window.setTimeout(() => {
        if (index < frames.length - 1) advance(index + 1)
        else {
          playbackTimerRef.current = null
          setIsPlaying(false)
        }
      }, 1050)
    }
    advance(0)
  }

  const applyTemplate = (template: StudioTemplate) => {
    stopPlayback()
    pushLayoutHistory()
    const nextSquad = createGenericSquad().map((player) => {
      if (!player.onPitch || player.slot === null) return player
      const custom = template.positions?.[player.slot]
      return custom ? { ...player, x: custom[0], y: custom[1] } : { ...player, ...formationPositions[template.formation][player.slot] }
    })
    const nextOpponents = createOpponentSquad(template.opponentFormation)
    const nextRoutes = template.routes.map((route, index) => ({ ...route, id: `${template.id}-route-${index}` }))
    const nextScene = createScene(`scene-${Date.now()}`, sceneNames[0], nextSquad, nextOpponents, template.formation, template.opponentFormation, nextRoutes)
    setContext({ ...template.context })
    setFormation(template.formation)
    setOpponentFormation(template.opponentFormation)
    setTactics({ ...template.tactics })
    setSquad(nextSquad)
    setOpponents(nextOpponents)
    setOpponentVisible(true)
    setRoutes(nextRoutes)
    setScenes([nextScene])
    setActiveSceneId(nextScene.id)
    setSelectedId('generic-st')
    setPlanSlots({})
    setBoardTool('move')
    setRightPanelTab('analysis')
    setCoachPreview(null)
    setHasDragged(true)
  }

  const previewCoachRecommendation = () => {
    let target = onPitch.find((player) => player.id === selectedId) ?? onPitch.find((player) => player.position === 'MF') ?? onPitch[0]
    let x = target.x
    let y = target.y
    let title = '선수 간격을 한 단계 조정합니다.'

    if (diagnostics.teamWidth < 58) {
      target = onPitch.reduce((widest, player) => Math.abs(player.x - 50) > Math.abs(widest.x - 50) ? player : widest, onPitch[0])
      x = target.x < 50 ? clamp(target.x - 9, 8, 92) : clamp(target.x + 9, 8, 92)
      y = target.y
      title = `${target.shortName}을 터치라인 쪽으로 이동해 공격 폭을 넓힙니다.`
    } else if (diagnostics.transitionRisk >= 62) {
      target = onPitch.find((player) => player.position === 'MF' && player.role.includes('앵커')) ?? onPitch.find((player) => player.position === 'MF') ?? target
      x = clamp(target.x + (50 - target.x) * .35, 8, 92)
      y = clamp(target.y + 9, 8, 92)
      title = `${target.shortName}을 후방에 남겨 전환 시 중앙을 보호합니다.`
    } else if (diagnostics.passOptions < 3 && selectedPlayer?.onPitch) {
      target = onPitch.filter((player) => player.id !== selectedPlayer.id).reduce((nearest, player) => distance(player, selectedPlayer) < distance(nearest, selectedPlayer) ? player : nearest, onPitch.find((player) => player.id !== selectedPlayer.id) ?? onPitch[0])
      x = clamp(target.x + (selectedPlayer.x - target.x) * .35, 8, 92)
      y = clamp(target.y + (selectedPlayer.y - target.y) * .35, 8, 92)
      title = `${target.shortName}을 지원 거리로 이동해 ${selectedPlayer.shortName}의 패스 선택지를 늘립니다.`
    }
    setCoachPreview({ playerId: target.id, x, y, title })
  }

  const applyCoachRecommendation = () => {
    if (!coachPreview) return
    pushLayoutHistory()
    setSquad((current) => current.map((player) => player.id === coachPreview.playerId ? { ...player, x: coachPreview.x, y: coachPreview.y } : player))
    setSelectedId(coachPreview.playerId)
    setHasDragged(true)
    setCoachPreview(null)
  }

  const savePlan = (slot: 'A' | 'B') => {
    const snapshot: PlanSnapshot = {
      savedAt: new Intl.DateTimeFormat('ko', { hour: '2-digit', minute: '2-digit' }).format(new Date()),
      context: { ...context },
      formation,
      tactics: { ...tactics },
      squad: squad.map((player) => ({ ...player })),
      opponentFormation,
      opponents: opponents.map((player) => ({ ...player })),
      routes: cloneRoutes(routes),
      teamWidth: diagnostics.teamWidth,
      defensiveLine: diagnostics.defensiveLine,
      transitionRisk: diagnostics.transitionRisk,
    }
    setPlanSlots((current) => ({ ...current, [slot]: snapshot }))
  }

  const loadPlan = (snapshot: PlanSnapshot) => {
    pushLayoutHistory()
    setContext(snapshot.context)
    setFormation(snapshot.formation)
    setTactics(snapshot.tactics)
    setSquad(snapshot.squad)
    setOpponentFormation(snapshot.opponentFormation)
    setOpponents(snapshot.opponents)
    setRoutes(cloneRoutes(snapshot.routes ?? []))
  }

  const exportPng = async () => {
    if (!exportRef.current || exportStatus === 'working') return
    setExportStatus('working')
    try {
      await document.fonts.ready
      const dataUrl = await toPng(exportRef.current, {
        cacheBust: true,
        pixelRatio: 2,
        backgroundColor: '#07130f',
        width: 1200,
        height: 675,
        style: { position: 'static', left: '0', top: '0', zIndex: '0' },
      })
      const anchor = document.createElement('a')
      const safeTeam = (context.teamName || 'team').replace(/[^가-힣a-zA-Z0-9-_]/g, '-')
      anchor.download = `retactic-${safeTeam}-${formation}.png`
      anchor.href = dataUrl
      anchor.click()
      setExportStatus('done')
      window.setTimeout(() => setExportStatus('idle'), 2200)
    } catch {
      setExportStatus('error')
      window.setTimeout(() => setExportStatus('idle'), 3000)
    }
  }

  return (
    <main className="universal-studio page-wrap wide">
      <header className="studio-heading">
        <div>
          <p className="eyebrow">UNIVERSAL TACTICS STUDIO</p>
          <h1><span>모든 경기를 위한</span><span>범용 전술 보드</span></h1>
          <p>경기 정보와 선수를 입력하세요.<br />전술의 구조적 위험을 바로 확인할 수 있습니다.</p>
        </div>
        <div className="studio-actions">
          <span className="save-status"><i /> 자동 저장 {lastSaved}</span>
          <button className="secondary-button compact" type="button" onClick={resetStudio}>초기화</button>
          <button className="primary-button compact" type="button" onClick={exportPng} disabled={exportStatus === 'working'}>{exportStatus === 'working' ? '이미지 생성 중…' : 'PNG 저장'}</button>
        </div>
      </header>

      <div className="studio-steps" aria-label="사용 순서">
        <span className="active"><b>1</b><i>경기 정보</i><small>팀과 목표 입력</small></span>
        <span className="active"><b>2</b><i>선수 배치</i><small>포메이션과 위치</small></span>
        <span className="active"><b>3</b><i>장면 구성</i><small>경로와 타임라인</small></span>
        <span className="active"><b>4</b><i>분석·공유</i><small>코치 제안과 저장</small></span>
      </div>

      <div className="studio-layout">
        <aside className="studio-left">
          <section className="panel studio-panel" id="studio-context">
            <div className="panel-title"><span>01</span><div><small>MATCH CONTEXT</small><h2>경기 정보</h2></div></div>
            <div className="form-grid">
              <label>우리 팀<input value={context.teamName} onChange={(event) => setContext({ ...context, teamName: event.target.value })} /></label>
              <label>상대 팀<input value={context.opponentName} onChange={(event) => setContext({ ...context, opponentName: event.target.value })} /></label>
              <label>경기 단계<select value={context.phase} onChange={(event) => setContext({ ...context, phase: event.target.value as MatchContext['phase'] })}>{['경기 전', '전반전', '하프타임', '후반전', '연장전'].map((item) => <option key={item}>{item}</option>)}</select></label>
              <label>분<input type="number" min="0" max="130" value={context.minute} onChange={(event) => setContext({ ...context, minute: Number(event.target.value) })} /></label>
            </div>
            <div className="score-editor">
              <label><span>{context.teamName || '우리 팀'}</span><input type="number" min="0" max="20" value={context.ourScore} onChange={(event) => setContext({ ...context, ourScore: Number(event.target.value) })} /></label>
              <b>:</b>
              <label><span>{context.opponentName || '상대 팀'}</span><input type="number" min="0" max="20" value={context.theirScore} onChange={(event) => setContext({ ...context, theirScore: Number(event.target.value) })} /></label>
            </div>
            <label className="full-field">이번 전술의 목표<select value={context.objective} onChange={(event) => setContext({ ...context, objective: event.target.value as MatchContext['objective'] })}>{['균형 유지', '득점 필요', '리드 보호', '압박 탈출', '상대 역습 차단'].map((item) => <option key={item}>{item}</option>)}</select></label>
          </section>

          <section className="panel studio-panel">
            <div className="panel-title"><span>03</span><div><small>TEAM INSTRUCTIONS</small><h2>팀 지시</h2></div></div>
            <RangeControl label="압박 강도" low="기다리기" high="즉시 압박" value={tactics.pressing} onChange={(value) => setTactics({ ...tactics, pressing: value })} />
            <RangeControl label="공격 폭" low="좁게" high="넓게" value={tactics.width} onChange={(value) => setTactics({ ...tactics, width: value })} />
            <RangeControl label="공격 템포" low="차분하게" high="빠르게" value={tactics.tempo} onChange={(value) => setTactics({ ...tactics, tempo: value })} />
            <RangeControl label="위험 감수" low="안전하게" high="과감하게" value={tactics.risk} onChange={(value) => setTactics({ ...tactics, risk: value })} />
          </section>
        </aside>

        <section className="studio-board">
          <section className="template-library" aria-label="전술 템플릿">
            <header><div><small>QUICK START</small><strong>검증된 시작점에서 편집하세요</strong></div><span>템플릿 선택 시 현재 보드를 교체합니다</span></header>
            <div>{studioTemplates.map((template) => <button type="button" key={template.id} onClick={() => applyTemplate(template)}><b>{template.title}</b><small>{template.note}</small></button>)}</div>
          </section>
          <div className="board-toolbar">
            <div><small>02 · FORMATION</small><strong>{formation}</strong></div>
            <div className="formation-pills">{formations.map((item) => <button type="button" className={formation === item ? 'active' : ''} key={item} onClick={() => applyFormation(item)}>{item}</button>)}</div>
            <div className="board-actions"><button className="reset-layout" type="button" disabled={layoutHistory.length === 0} onClick={undoLayout}>↶ 실행 취소</button><button className="reset-layout" type="button" onClick={() => applyFormation(formation)}>배치 원위치</button></div>
          </div>
          <div className="opponent-toolbar">
            <label><input type="checkbox" checked={opponentVisible} onChange={(event) => setOpponentVisible(event.target.checked)} /><span>상대팀 표시</span></label>
            <div><small>{context.opponentName || '상대 팀'} 대형</small>{formations.map((item) => <button type="button" className={opponentFormation === item ? 'active' : ''} key={item} onClick={() => applyOpponentFormation(item)}>{item}</button>)}</div>
            <p>붉은 선수를 직접 움직여 상대 압박 구조를 재현하세요.</p>
          </div>
          <div className="drawing-toolbar" aria-label="전술 그리기 도구">
            <div role="group" aria-label="보드 도구">
              <button type="button" className={boardTool === 'move' ? 'active' : ''} onClick={() => setBoardTool('move')}><i>↕</i><span>선수 이동</span></button>
              <button type="button" className={boardTool === 'pass' ? 'active pass' : 'pass'} onClick={() => setBoardTool('pass')}><i>→</i><span>패스</span></button>
              <button type="button" className={boardTool === 'run' ? 'active run' : 'run'} onClick={() => setBoardTool('run')}><i>⇢</i><span>침투</span></button>
            </div>
            <p>{boardTool === 'move' ? '선수를 끌어 배치하세요.' : `${selectedPlayer?.shortName ?? '선수'} 선택됨 · 경기장의 도착 지점을 누르세요.`}</p>
            <div><span>경로 {routes.length}</span><button type="button" disabled={routes.length === 0} onClick={() => setRoutes((current) => current.slice(0, -1))}>되돌리기</button><button type="button" disabled={routes.length === 0} onClick={() => setRoutes([])}>모두 지우기</button></div>
          </div>
          <div className={`pitch studio-pitch tool-${boardTool} ${isPlaying ? 'story-playing' : ''}`} ref={pitchRef} onPointerDown={handlePitchPointerDown}>
            <div className="pitch-lines"><i className="halfway" /><i className="circle" /><i className="box top" /><i className="box bottom" /></div>
            {draggingId && selectedPlayer && <div className={`position-zone zone-${selectedPlayer.position.toLowerCase()}`} aria-hidden="true" />}
            <svg className="board-routes" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="패스와 침투 경로">
              <defs>
                <marker id="pass-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" /></marker>
                <marker id="run-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" /></marker>
              </defs>
              {routes.map((route) => <line key={route.id} className={`board-route ${route.type}`} x1={route.from.x} y1={route.from.y} x2={route.to.x} y2={route.to.y} markerEnd={`url(#${route.type}-arrow)`} />)}
              {coachPreview && previewPlayer && <line className="coach-preview-route" x1={previewPlayer.x} y1={previewPlayer.y} x2={coachPreview.x} y2={coachPreview.y} />}
            </svg>
            <div className="attack-label">↑ {context.teamName || '우리 팀'} 공격 방향</div>
            {!hasDragged && boardTool === 'move' && <div className="drag-coachmark"><span>↕</span><strong>선수를 끌어서 위치를 바꿔보세요</strong><small>클릭하면 이름과 역할을 수정할 수 있습니다</small></div>}
            {boardTool !== 'move' && <div className="route-coachmark"><b>{boardTool === 'pass' ? 'PASS' : 'RUN'}</b><span>{selectedPlayer?.shortName ?? '선수'}에서 시작 · 도착 지점을 클릭</span></div>}
            {ghostPlayers.map((player) => <span className="ghost-player" key={`ghost-${player.id}`} style={{ left: `${player.x}%`, top: `${player.y}%` }}><i>{player.number}</i></span>)}
            {coachPreview && <span className="coach-preview-target" style={{ left: `${coachPreview.x}%`, top: `${coachPreview.y}%` }}>AI</span>}
            {opponentVisible && opponents.map((player) => (
              <button
                className={`player-token opponent-token ${draggingId === player.id ? 'dragging' : ''}`}
                type="button"
                key={player.id}
                style={{ left: `${player.x}%`, top: `${player.y}%` }}
                onPointerDown={(event) => { event.stopPropagation(); if (boardTool !== 'move' || isPlaying) return; pushLayoutHistory(); event.currentTarget.setPointerCapture(event.pointerId); setDraggingId(player.id) }}
                onPointerMove={(event) => handleOpponentPointerMove(event, player.id)}
                onPointerUp={handlePointerUp}
                aria-label={`${context.opponentName} ${player.shortName}`}
              ><span>{player.number}</span><strong>{player.shortName}</strong></button>
            ))}
            {onPitch.map((player) => (
              <button
                className={`player-token position-${player.position.toLowerCase()} ${selectedId === player.id ? 'selected' : ''} ${draggingId === player.id ? 'dragging' : ''}`}
                type="button"
                key={player.id}
                style={{ left: `${player.x}%`, top: `${player.y}%` }}
                onPointerDown={(event) => handlePointerDown(event, player.id)}
                onPointerMove={(event) => handlePointerMove(event, player.id)}
                onPointerUp={handlePointerUp}
                aria-label={`${player.name}, ${player.role}`}
              ><span>{player.number}</span><strong>{player.shortName}</strong></button>
            ))}
          </div>
          <section className="storyboard panel" aria-label="전술 장면 타임라인">
            <header><div><small>SCENE TIMELINE</small><strong>배치를 장면으로 저장하고 순서대로 재생하세요</strong></div><div><button type="button" onClick={saveCurrentScene}>현재 장면 저장</button><button type="button" disabled={scenes.length >= 5} onClick={addScene}>＋ 다음 장면</button><button className="play-story" type="button" disabled={scenes.length < 2} onClick={isPlaying ? stopPlayback : playScenes}>{isPlaying ? '■ 정지' : '▶ 전체 재생'}</button></div></header>
            <div className="scene-track">
              {scenes.map((scene, index) => <div className={`scene-item ${activeSceneId === scene.id ? 'active' : ''}`} key={scene.id}>
                <button type="button" onClick={() => { stopPlayback(); restoreScene(scene) }}><b>{String(index + 1).padStart(2, '0')}</b><span>{scene.name}</span><small>{scene.formation} · 경로 {scene.routes.length}</small></button>
                {scenes.length > 1 && <button className="delete-scene" type="button" aria-label={`${scene.name} 삭제`} onClick={() => deleteScene(scene.id)}>×</button>}
              </div>)}
              {scenes.length < 5 && <button className="empty-scene" type="button" onClick={addScene}><b>＋</b><span>다음 움직임 추가</span></button>}
            </div>
            <p>현재 장면을 저장한 뒤 다음 장면에서 선수 위치를 바꾸세요. 재생 시 장면 사이를 부드럽게 연결합니다.</p>
          </section>
          <div className="bench panel studio-bench">
            <div className="bench-heading"><div><small>BENCH</small><strong>필드 선수를 선택한 뒤 교체 선수를 누르세요</strong></div><span>{bench.length}명</span></div>
            <div className="bench-list">{bench.map((player) => <button type="button" key={player.id} onClick={() => swapWithBench(player.id)}><span>{player.number}</span><div><strong>{player.shortName}</strong><small>{player.name}</small></div>{selectedPlayer?.onPitch && <em>교체</em>}</button>)}</div>
          </div>
        </section>

        <aside className="studio-right">
          <nav className="context-panel-tabs" aria-label="설정 패널">
            <button type="button" className={rightPanelTab === 'player' ? 'active' : ''} onClick={() => setRightPanelTab('player')}>선수</button>
            <button type="button" className={rightPanelTab === 'analysis' ? 'active' : ''} onClick={() => setRightPanelTab('analysis')}>분석</button>
            <button type="button" className={rightPanelTab === 'plan' ? 'active' : ''} onClick={() => setRightPanelTab('plan')}>전술안</button>
          </nav>

          {rightPanelTab === 'player' && <section className="panel studio-panel player-editor context-panel-content">
            <div className="panel-title"><span>02</span><div><small>SELECTED PLAYER</small><h2>선수 정보</h2></div></div>
            {selectedPlayer && <>
              <div className="selected-summary"><b>{selectedPlayer.number}</b><div><strong>{selectedPlayer.name}</strong><small>{selectedPlayer.position} · {selectedPlayer.onPitch ? '필드 선수' : '벤치 선수'}</small></div></div>
              <div className="player-quick-metrics"><span><small>주변 패스</small><b>{diagnostics.passOptions}</b></span><span><small>현재 좌표</small><b>{Math.round(selectedPlayer.x)}·{Math.round(selectedPlayer.y)}</b></span></div>
              <div className="form-grid player-fields">
                <label>표시 이름<input maxLength={10} value={selectedPlayer.shortName} onChange={(event) => updatePlayer({ shortName: event.target.value, name: event.target.value })} /></label>
                <label>등번호<input type="number" min="1" max="99" value={selectedPlayer.number} onChange={(event) => updatePlayer({ number: Number(event.target.value) })} /></label>
              </div>
              <label className="full-field">역할<select value={selectedPlayer.role} onChange={(event) => updatePlayer({ role: event.target.value })}>{roleOptions[selectedPlayer.position].map((role) => <option key={role}>{role}</option>)}</select></label>
              <p className="player-editor-hint">선수 마커는 최소 정보만 표시합니다. 역할과 연결 지표는 이 패널에서 확인하세요.</p>
            </>}
          </section>}

          {rightPanelTab === 'analysis' && <section className="panel studio-panel diagnostics-panel context-panel-content">
            <div className="panel-title"><span>04</span><div><small>STRUCTURE CHECK</small><h2>실시간 전술 점검</h2></div></div>
            <div className="diagnostic-summary four"><span><small>팀 폭</small><b>{diagnostics.teamWidth}</b><em>{signed(diagnostics.teamWidth - sceneBaseline.teamWidth)}</em></span><span><small>수비 라인</small><b>{diagnostics.defensiveLine}</b><em>{signed(diagnostics.defensiveLine - sceneBaseline.defensiveLine)}</em></span><span><small>패스 선택</small><b>{diagnostics.passOptions}</b><em>{selectedPlayer?.shortName}</em></span><span><small>전환 위험</small><b>{diagnostics.transitionRisk}</b><em>{diagnostics.transitionRisk < 62 ? '안정' : '주의'}</em></span></div>
            <p className="baseline-caption">변화값은 저장된 현재 장면 대비입니다.</p>
            <ul className="diagnostic-list">{diagnostics.checks.map((check) => <li className={check.tone} key={check.title}><i>{check.tone === 'good' ? '✓' : '!'}</i><div><strong>{check.title}</strong><span>{check.detail}</span></div></li>)}</ul>
            <div className={`studio-recommendation ${coachPreview ? 'previewing' : ''}`}><small>AI-STYLE COACH · RULE BASED</small><strong>{coachPreview?.title ?? context.objective}</strong><p>{getStudioNote(context, tactics, diagnostics.transitionRisk, diagnostics.passOptions, diagnostics.compactness)}</p><div><button type="button" onClick={previewCoachRecommendation}>{coachPreview ? '다른 제안 보기' : '미리 보기'}</button><button type="button" disabled={!coachPreview} onClick={applyCoachRecommendation}>적용하기</button></div></div>
          </section>}

          {rightPanelTab === 'plan' && <section className="panel studio-panel session-summary context-panel-content">
            <small>현재 전술안</small>
            <h3>{context.teamName} vs {context.opponentName}</h3>
            <p>{context.phase} {context.minute > 0 ? `${context.minute}분` : ''} · {context.ourScore}:{context.theirScore} · {formation} · 장면 {scenes.length}개</p>
            <div className="plan-slot-buttons"><button type="button" onClick={() => savePlan('A')}>A안에 저장</button><button type="button" onClick={() => savePlan('B')}>B안에 저장</button></div>
            <div className="plan-slots">
              <PlanSlot label="A" snapshot={planSlots.A} onLoad={loadPlan} />
              <PlanSlot label="B" snapshot={planSlots.B} onLoad={loadPlan} />
            </div>
            {planSlots.A && planSlots.B && <div className="ab-difference"><small>A/B 차이</small><span>전환 위험 <b>{signed(planSlots.B.transitionRisk - planSlots.A.transitionRisk)}</b></span><span>팀 폭 <b>{signed(planSlots.B.teamWidth - planSlots.A.teamWidth)}</b></span></div>}
            <div className="export-actions"><button className="secondary-button" type="button" onClick={() => window.print()}>PDF</button><button className="primary-button" type="button" onClick={exportPng}>PNG 저장 <span>↓</span></button></div>
            {exportStatus === 'done' && <p className="export-feedback success">PNG 저장이 완료됐습니다.</p>}
            {exportStatus === 'error' && <p className="export-feedback error">이미지 생성에 실패했습니다. 다시 시도해주세요.</p>}
          </section>}
        </aside>
      </div>

      <div className="export-card" ref={exportRef} aria-hidden="true">
        <header><div><span>R:</span><strong>RE:TACTIC</strong></div><small>TACTICAL PLAN · {today}</small></header>
        <div className="export-card-body">
          <section className="export-pitch">
            <div className="export-pitch-lines"><i /><i /><i /></div>
            <svg className="export-routes" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              {routes.map((route) => <g key={`export-${route.id}`}><line x1={route.from.x} y1={route.from.y} x2={route.to.x} y2={route.to.y} stroke={route.type === 'pass' ? '#a79cff' : '#ffc65d'} strokeWidth="0.65" strokeDasharray={route.type === 'pass' ? '2.4 1.7' : '1.2 1.2'} /><polygon points={arrowPoints(route)} fill={route.type === 'pass' ? '#a79cff' : '#ffc65d'} /></g>)}
            </svg>
            {opponentVisible && opponents.map((player) => <span className="export-player opponent" key={`export-${player.id}`} style={{ left: `${player.x}%`, top: `${player.y}%` }}>{player.shortName}</span>)}
            {onPitch.map((player) => <span className="export-player ours" key={`export-${player.id}`} style={{ left: `${player.x}%`, top: `${player.y}%` }}>{player.shortName}</span>)}
          </section>
          <aside className="export-summary">
            <p>MATCH PLAN</p>
            <h2>{context.teamName} <b>{context.ourScore}:{context.theirScore}</b> {context.opponentName}</h2>
            <span>{context.phase} {context.minute ? `${context.minute}′` : ''} · 목표: {context.objective}</span>
            <div className="export-formations"><i><small>우리 대형</small><b>{formation}</b></i><i><small>상대 대형</small><b>{opponentFormation}</b></i></div>
            <div className="export-metrics"><i><small>팀 폭</small><b>{diagnostics.teamWidth}</b></i><i><small>수비 라인</small><b>{diagnostics.defensiveLine}</b></i><i><small>전환 위험</small><b>{diagnostics.transitionRisk}</b></i></div>
            <div className="export-note"><small>COACHING NOTE</small><strong>{context.objective}</strong><p>{getStudioNote(context, tactics, diagnostics.transitionRisk, diagnostics.passOptions, diagnostics.compactness)}</p></div>
          </aside>
        </div>
        <footer><span>Designed with RE:TACTIC Universal Tactics Studio</span><b>jinwon25.github.io/Dacon</b></footer>
      </div>
    </main>
  )
}

function PlanSlot({ label, snapshot, onLoad }: { label: 'A' | 'B'; snapshot?: PlanSnapshot; onLoad: (snapshot: PlanSnapshot) => void }) {
  return <div className={`plan-slot ${snapshot ? 'saved' : ''}`}><b>{label}</b>{snapshot ? <><span>{snapshot.formation}</span><small>위험 {snapshot.transitionRisk} · {snapshot.savedAt}</small><button type="button" onClick={() => onLoad(snapshot)}>불러오기</button></> : <><span>비어 있음</span><small>위 버튼으로 현재 안 저장</small></>}</div>
}

const signed = (value: number) => value > 0 ? `+${value}` : `${value}`

function RangeControl({ label, low, high, value, onChange }: { label: string; low: string; high: string; value: number; onChange: (value: number) => void }) {
  return <label className="tactic-slider"><span><strong>{label}</strong><b>{value}</b></span><input type="range" min="0" max="100" value={value} onChange={(event) => onChange(Number(event.target.value))} style={{ '--range': `${value}%` } as React.CSSProperties} /><small><i>{low}</i><i>{high}</i></small></label>
}

function getStudioNote(context: MatchContext, tactics: Tactics, risk: number, passOptions = 3, compactness = 20) {
  if (risk > 65) return '공을 잃은 순간 중앙을 지킬 선수를 한 명 지정하세요. 공격과 압박을 동시에 높이면 복귀 거리가 길어집니다.'
  if (passOptions < 2) return '선택한 선수 주변의 패스 선택지가 부족합니다. 가장 가까운 미드필더를 지원 거리 안으로 이동해 삼각형을 만드세요.'
  if (compactness > 26) return '선수 간 평균 간격이 넓습니다. 빌드업 구간의 두 선을 가깝게 두고 반대편 폭은 한 명이 유지하세요.'
  if (context.objective === '득점 필요' && tactics.tempo < 55) return '득점이 필요하지만 템포가 낮습니다. 공격 전환 속도를 한 단계 높이는 안을 비교해보세요.'
  if (context.objective === '리드 보호' && tactics.risk > 55) return '리드 보호 목표에 비해 위험 감수가 높습니다. 풀백 한 명의 전진을 제한하면 균형이 좋아집니다.'
  if (context.objective === '압박 탈출') return '첫 번째 패스 옵션과 반대편 전환 선수를 멀리 두어 상대 압박 폭을 늘리세요.'
  return '현재 구조는 큰 불균형이 없습니다. 선수 역할이 실제 움직임과 일치하는지 마지막으로 확인하세요.'
}
