import type { FormationKey, Player, Tactics } from '../types'
import { matchEvidence, playerEvidence, type PlayerMatchEvidence, type TeamWindowEvidence } from './evidence'
import { formationPositions, initialSquad } from './match'
import { spatialEvidence, type SpatialPass, type SpatialShot } from './spatialEvidence'

export type GuidedScenarioId = 'korea-portugal-65' | 'argentina-netherlands-83' | 'korea-south-africa-64'

export interface OfficialMatchReport {
  inContestPossession: number
  lineBreaks: [number, number]
  finalThirdReceptions: [number, number]
  crosses: [number, number]
  forcedTurnovers: [number, number]
  secondBalls: [number, number]
  phases: Array<{ label: string; ours: number; opponent: number }>
}

export interface PassNetworkNode {
  id: string
  label: string
  x: number
  y: number
  involvements: number
}

export interface PassNetworkEdge {
  from: string
  to: string
  count: number
}

export interface PassNetworkData {
  nodes: PassNetworkNode[]
  edges: PassNetworkEdge[]
  completedPasses: number
  minimumEdgeCount: number
  nodeMetricLabel?: string
}

export interface BallFlowAction {
  id: string
  time: string
  type: 'pass' | 'carry' | 'shot'
  player: string
  recipient?: string
  start: [number, number]
  end: [number, number]
  outcome?: string
}

export interface GuidedScenario {
  id: GuidedScenarioId
  order: string
  tournament: string
  missionType: string
  difficulty: string
  matchId: number
  sourceUrl: string
  sourceName?: string
  sourceKind?: 'event' | 'official-report'
  sourceNote?: string
  extractedAt: string
  windowLabel: string
  minute: number
  score: [number, number]
  ours: { name: string; short: string; flag: string; status: string }
  opponent: { name: string; short: string; flag: string; status: string }
  objective: string
  intro: { eyebrow: string; title: string; accent: string; lead: string }
  briefing: {
    title: string
    description: string
    diagnosisTitle: string
    diagnosisQuote: string
    contextNumber: string
    contextLabel: string
    successTitle: string
    successDetail: string
    failureTitle: string
    failureDetail: string
    optionTitle: string
    optionPlayer: string
    optionNumber: number
    optionPosition: string
    optionRole: string
    optionAvailability: string
    optionTraits: string[]
    optionQuote: string
  }
  comparisonRows: Array<{ label: string; ours: string; opponent: string; oursValue: number; opponentValue: number; max: number; invert?: boolean }>
  possessionEstimate: [number, number]
  evidence: { ours: TeamWindowEvidence; opponent: TeamWindowEvidence }
  spatial: { ours: { finalThirdEntries: SpatialPass[]; shots: SpatialShot[] }; opponent: { finalThirdEntries: SpatialPass[]; shots: SpatialShot[] } }
  spatialCopy: {
    entriesTitle: string
    entriesBody: string
    entriesOurs: string
    entriesOpponent: string
    shotsTitle: string
    shotsBody: string
    shotsOurs: string
    shotsOpponent: string
  }
  networks: { ours: PassNetworkData; opponent: PassNetworkData }
  networkCopy: { title: string; body: string }
  flow: { title: string; summary: string; actions: BallFlowAction[] }
  squad: Player[]
  playerEvidence: Record<string, PlayerMatchEvidence>
  defaultFormation: FormationKey
  defaultTactics: Tactics
  selectedPlayerId: string
  impactPlayerId: string
  metricLabels: [string, string, string, string]
  result: {
    baseline: string
    impactOn: string
    impactOff: string
    actualChoice: string
    actualOutcome: string
    planOn: string
    planOff: string
    operatingSafe: string
    operatingRisk: string
  }
  officialReport?: OfficialMatchReport
  networkPositionBasis?: string
}

const koreaNetwork: PassNetworkData = {
  completedPasses: 42,
  minimumEdgeCount: 2,
  nodes: [
    ['Young-Gwon Kim', '김영권', 19.6, 20.2, 12], ['In-Beom Hwang', '황인범', 42.9, 58.6, 10],
    ['Jin-Su Kim', '김진수', 47.8, 5, 10], ['Gue-Sung Cho', '조규성', 60.8, 39, 9],
    ['Kyung-Won Kwon', '권경원', 26.1, 81.1, 9], ['Seung-Gyu Kim', '김승규', 10, 49.8, 8],
    ['Jae-Sung Lee', '이재성', 55.9, 52.8, 6], ['Woo-Young Jung', '정우영', 27.8, 26.5, 6],
    ['Moon-Hwan Kim', '김문환', 30.4, 91.4, 6], ['Kang-In Lee', '이강인', 42.4, 23.3, 4],
    ['Heung-Min Son', '손흥민', 63.3, 67.5, 4],
  ].map(([id, label, x, y, involvements]) => ({ id: String(id), label: String(label), x: Number(x), y: Number(y), involvements: Number(involvements) })),
  edges: [
    ['Kyung-Won Kwon', 'In-Beom Hwang', 3], ['Seung-Gyu Kim', 'Gue-Sung Cho', 2],
    ['Seung-Gyu Kim', 'Young-Gwon Kim', 2], ['Jin-Su Kim', 'Young-Gwon Kim', 2],
    ['Jin-Su Kim', 'Woo-Young Jung', 2], ['Woo-Young Jung', 'Young-Gwon Kim', 2],
    ['Young-Gwon Kim', 'Jin-Su Kim', 2], ['Gue-Sung Cho', 'Heung-Min Son', 2],
    ['In-Beom Hwang', 'Kyung-Won Kwon', 2],
  ].map(([from, to, count]) => ({ from: String(from), to: String(to), count: Number(count) })),
}

const portugalNetwork: PassNetworkData = {
  completedPasses: 153,
  minimumEdgeCount: 3,
  nodes: [
    ['Rúben Diogo Da Silva Neves', '네베스', 51.6, 51.7, 51], ['Vitor Machado Ferreira', '비티냐', 60.7, 29.8, 44],
    ['João Pedro Cavaco Cancelo', '칸셀루', 69, 13.4, 43], ['Kléper Laveran Lima Ferreira', '페페', 42.2, 72.3, 40],
    ['João Mário Naval da Costa Eduardo', '주앙 마리우', 70.6, 22.7, 32], ['Matheus Luiz Nunes', '마테우스', 62.7, 66.3, 25],
    ['José Diogo Dalot Teixeira', '달로트', 58, 82.5, 24], ['António João Pereira Albuquerque Tavares Silva', '안토니우', 43.8, 25.2, 17],
    ['Ricardo Jorge Luz Horta', '오르타', 75.7, 67.3, 12], ['Cristiano Ronaldo dos Santos Aveiro', '호날두', 69.5, 27.7, 11],
    ['Diogo Meireles Costa', '코스타', 13.6, 46.1, 7],
  ].map(([id, label, x, y, involvements]) => ({ id: String(id), label: String(label), x: Number(x), y: Number(y), involvements: Number(involvements) })),
  edges: [
    ['Kléper Laveran Lima Ferreira', 'Rúben Diogo Da Silva Neves', 8], ['João Pedro Cavaco Cancelo', 'Vitor Machado Ferreira', 8],
    ['Vitor Machado Ferreira', 'João Pedro Cavaco Cancelo', 6], ['João Pedro Cavaco Cancelo', 'João Mário Naval da Costa Eduardo', 6],
    ['Rúben Diogo Da Silva Neves', 'José Diogo Dalot Teixeira', 5], ['José Diogo Dalot Teixeira', 'Kléper Laveran Lima Ferreira', 5],
    ['João Mário Naval da Costa Eduardo', 'João Pedro Cavaco Cancelo', 5], ['Rúben Diogo Da Silva Neves', 'João Pedro Cavaco Cancelo', 5],
    ['Rúben Diogo Da Silva Neves', 'Kléper Laveran Lima Ferreira', 5], ['Vitor Machado Ferreira', 'Matheus Luiz Nunes', 4],
    ['Rúben Diogo Da Silva Neves', 'António João Pereira Albuquerque Tavares Silva', 4], ['João Mário Naval da Costa Eduardo', 'Vitor Machado Ferreira', 4],
  ].map(([from, to, count]) => ({ from: String(from), to: String(to), count: Number(count) })),
}

const argentinaNetwork: PassNetworkData = {
  completedPasses: 17,
  minimumEdgeCount: 1,
  nodes: [
    ['Nicolás Hernán Otamendi', '오타멘디', 31.8, 68.5, 6], ['Enzo Fernandez', '엔소', 49.9, 55.4, 5],
    ['Alexis Mac Allister', '맥알리스터', 70.4, 48.3, 4], ['Marcos Javier Acuña', '아쿠냐', 37.9, 13.8, 3],
    ['Lisandro Martínez', '리산드로', 16.4, 34.6, 3], ['Leandro Daniel Paredes', '파레데스', 46.8, 25.9, 3],
    ['Lionel Andrés Messi Cuccittini', '메시', 50.8, 58.4, 3], ['Damián Emiliano Martínez', 'E.마르티네스', 9.2, 56.3, 2],
    ['Nahuel Molina Lucero', '몰리나', 35.7, 84.7, 1], ['Germán Alejandro Pezzella', '페첼라', 34, 82.8, 1],
    ['Julián Álvarez', '알바레스', 87.3, 25.2, 1], ['Lautaro Javier Martínez', '라우타로', 50, 50, 1],
  ].map(([id, label, x, y, involvements]) => ({ id: String(id), label: String(label), x: Number(x), y: Number(y), involvements: Number(involvements) })),
  edges: [],
}

const netherlandsNetwork: PassNetworkData = {
  completedPasses: 28,
  minimumEdgeCount: 2,
  nodes: [
    ['Nathan Aké', '아케', 52.5, 13.5, 9], ['Virgil van Dijk', '반다이크', 39.8, 26.9, 8],
    ['Jurriën David Norman Timber', '팀버', 45.8, 63.4, 8], ['Andries Noppert', '노퍼르트', 22.2, 41.9, 7],
    ['Steven Berghuis', '베르하위스', 75.2, 96.3, 7], ['Teun Koopmeiners', '코프메이너르스', 62, 81.4, 4],
    ['Frenkie de Jong', '더용', 67.6, 13.5, 3], ['Cody Mathès Gakpo', '각포', 71.4, 7.2, 3],
    ['Wout Weghorst', '베호르스트', 74.4, 46.2, 3], ['Memphis Depay', '데파이', 58, 41.3, 2],
    ['Denzel Dumfries', '둠프리스', 61.8, 91.6, 2],
  ].map(([id, label, x, y, involvements]) => ({ id: String(id), label: String(label), x: Number(x), y: Number(y), involvements: Number(involvements) })),
  edges: [
    ['Virgil van Dijk', 'Nathan Aké', 3], ['Nathan Aké', 'Virgil van Dijk', 2],
    ['Jurriën David Norman Timber', 'Steven Berghuis', 2], ['Andries Noppert', 'Jurriën David Norman Timber', 2],
    ['Nathan Aké', 'Cody Mathès Gakpo', 2], ['Teun Koopmeiners', 'Steven Berghuis', 2],
  ].map(([from, to, count]) => ({ from: String(from), to: String(to), count: Number(count) })),
}

const argentinaBase: Array<Omit<Player, 'x' | 'y'>> = [
  { id: 'arg-emiliano', number: 23, name: '에밀리아노 마르티네스', shortName: 'E.마르티네스', position: 'GK', role: '스위퍼 키퍼', onPitch: true, slot: 0 },
  { id: 'arg-molina', number: 26, name: '나우엘 몰리나', shortName: '몰리나', position: 'DF', role: '공격형 풀백', onPitch: true, slot: 1 },
  { id: 'arg-pezzella', number: 6, name: '헤르만 페첼라', shortName: '페첼라', position: 'DF', role: '커버 센터백', onPitch: true, slot: 2 },
  { id: 'arg-otamendi', number: 19, name: '니콜라스 오타멘디', shortName: '오타멘디', position: 'DF', role: '빌드업 센터백', onPitch: true, slot: 3 },
  { id: 'arg-lisandro', number: 25, name: '리산드로 마르티네스', shortName: '리산드로', position: 'DF', role: '커버 센터백', onPitch: true, slot: 4 },
  { id: 'arg-tagliafico', number: 3, name: '니콜라스 탈리아피코', shortName: '탈리아피코', position: 'DF', role: '수비형 풀백', onPitch: true, slot: 5 },
  { id: 'arg-paredes', number: 5, name: '레안드로 파레데스', shortName: '파레데스', position: 'MF', role: '앵커', onPitch: true, slot: 6 },
  { id: 'arg-enzo', number: 24, name: '엔소 페르난데스', shortName: '엔소', position: 'MF', role: '딥라잉 플레이메이커', onPitch: true, slot: 7 },
  { id: 'arg-macallister', number: 20, name: '알렉시스 맥알리스터', shortName: '맥알리스터', position: 'MF', role: '볼 위닝 미드필더', onPitch: true, slot: 8 },
  { id: 'arg-messi', number: 10, name: '리오넬 메시', shortName: '메시', position: 'FW', role: '인사이드 포워드', onPitch: true, slot: 9 },
  { id: 'arg-lautaro', number: 22, name: '라우타로 마르티네스', shortName: '라우타로', position: 'FW', role: '라인 브레이커', onPitch: true, slot: 10 },
  { id: 'arg-montiel', number: 4, name: '곤살로 몬티엘', shortName: '몬티엘', position: 'DF', role: '수비형 풀백', onPitch: false, slot: null },
  { id: 'arg-dimaria', number: 11, name: '앙헬 디마리아', shortName: '디마리아', position: 'MF', role: '와이드 플레이메이커', onPitch: false, slot: null },
  { id: 'arg-palacios', number: 14, name: '에세키엘 팔라시오스', shortName: '팔라시오스', position: 'MF', role: '볼 위닝 미드필더', onPitch: false, slot: null },
  { id: 'arg-dybala', number: 21, name: '파울로 디발라', shortName: '디발라', position: 'FW', role: '인사이드 포워드', onPitch: false, slot: null },
  { id: 'arg-foyth', number: 2, name: '후안 포이스', shortName: '포이스', position: 'DF', role: '커버 센터백', onPitch: false, slot: null },
]

const argentinaSquad = argentinaBase.map((player) => ({
  ...player,
  ...(player.slot === null ? { x: 0, y: 0 } : formationPositions['5-3-2'][player.slot]),
}))

const argentinaPlayerEvidence: Record<string, PlayerMatchEvidence> = {
  'arg-emiliano': { passesCompleted: 9, passesAttempted: 22, pressures: 0, shots: 0, recoveries: 4, carries: 9 },
  'arg-molina': { passesCompleted: 29, passesAttempted: 36, pressures: 12, shots: 1, recoveries: 4, carries: 27 },
  'arg-pezzella': { passesCompleted: 0, passesAttempted: 1, pressures: 0, shots: 0, recoveries: 0, carries: 1, note: '77분 교체 투입' },
  'arg-otamendi': { passesCompleted: 54, passesAttempted: 60, pressures: 7, shots: 0, recoveries: 5, carries: 43 },
  'arg-lisandro': { passesCompleted: 19, passesAttempted: 24, pressures: 9, shots: 0, recoveries: 1, carries: 15 },
  'arg-tagliafico': { passesCompleted: 0, passesAttempted: 0, pressures: 0, shots: 0, recoveries: 0, carries: 0, note: '77분 교체 투입' },
  'arg-paredes': { passesCompleted: 2, passesAttempted: 3, pressures: 6, shots: 0, recoveries: 0, carries: 1, note: '65분 교체 투입' },
  'arg-enzo': { passesCompleted: 39, passesAttempted: 47, pressures: 20, shots: 0, recoveries: 1, carries: 34 },
  'arg-macallister': { passesCompleted: 24, passesAttempted: 28, pressures: 20, shots: 0, recoveries: 3, carries: 26 },
  'arg-messi': { passesCompleted: 26, passesAttempted: 28, pressures: 13, shots: 4, recoveries: 2, carries: 30 },
  'arg-lautaro': { passesCompleted: 1, passesAttempted: 1, pressures: 0, shots: 0, recoveries: 0, carries: 0, note: '80분 교체 투입' },
  'arg-montiel': { passesCompleted: 0, passesAttempted: 0, pressures: 0, shots: 0, recoveries: 0, carries: 0, note: '83분 시점 벤치' },
}

const argentinaSpatial: GuidedScenario['spatial'] = {
  ours: {
    finalThirdEntries: [
      { minute: 70, player: 'Cristian Romero', recipient: 'Alexis Mac Allister', start: [57.9, 100], end: [69.8, 59.6] },
    ],
    shots: [
      { minute: 72, player: 'Lionel Messi', location: [90, 50], xg: 0.784, outcome: 'Goal' },
    ],
  },
  opponent: {
    finalThirdEntries: [
      { minute: 73, player: 'Andries Noppert', recipient: 'Steven Berghuis', start: [16.9, 38.5], end: [84.5, 96.8] },
      { minute: 76, player: 'Virgil van Dijk', recipient: 'Frenkie de Jong', start: [61.3, 13.4], end: [67.6, 10.1] },
      { minute: 76, player: 'Jurriën Timber', recipient: 'Steven Berghuis', start: [55.4, 58.8], end: [72.2, 94.8] },
      { minute: 78, player: 'Nathan Aké', recipient: 'Cody Gakpo', start: [58.1, 8.9], end: [71.2, 8.4] },
      { minute: 80, player: 'Nathan Aké', recipient: 'Cody Gakpo', start: [54.3, 18.9], end: [70.5, 5.9] },
      { minute: 81, player: 'Jurriën Timber', recipient: 'Steven Berghuis', start: [60.8, 76.1], end: [72.7, 96] },
      { minute: 82, player: 'Teun Koopmeiners', recipient: 'Steven Berghuis', start: [55.4, 85.6], end: [72.2, 97.5] },
      { minute: 82, player: 'Jurriën Timber', recipient: 'Teun Koopmeiners', start: [61.1, 62.9], end: [68.7, 77.6] },
    ],
    shots: [
      { minute: 82, player: 'Wout Weghorst', location: [90.6, 58], xg: 0.047, outcome: 'Goal' },
    ],
  },
}

const korea2026Base: Array<Omit<Player, 'x' | 'y'>> = [
  { id: 'kor26-kim-seunggyu', number: 1, name: '김승규', shortName: '김승규', position: 'GK', role: '스위퍼 키퍼', onPitch: true, slot: 0 },
  { id: 'kor26-lee-hanbeom', number: 2, name: '이한범', shortName: '이한범', position: 'DF', role: '커버 센터백', onPitch: true, slot: 1 },
  { id: 'kor26-kim-minjae', number: 4, name: '김민재', shortName: '김민재', position: 'DF', role: '빌드업 센터백', onPitch: true, slot: 2 },
  { id: 'kor26-lee-gihyuk', number: 3, name: '이기혁', shortName: '이기혁', position: 'DF', role: '빌드업 센터백', onPitch: true, slot: 3 },
  { id: 'kor26-seol-youngwoo', number: 22, name: '설영우', shortName: '설영우', position: 'DF', role: '공격형 풀백', onPitch: true, slot: 4 },
  { id: 'kor26-hwang-inbeom', number: 6, name: '황인범', shortName: '황인범', position: 'MF', role: '딥라잉 플레이메이커', onPitch: true, slot: 5 },
  { id: 'kor26-castrop', number: 23, name: '옌스 카스트로프', shortName: '카스트로프', position: 'MF', role: '볼 위닝 미드필더', onPitch: true, slot: 6 },
  { id: 'kor26-kim-jingyu', number: 24, name: '김진규', shortName: '김진규', position: 'MF', role: '공간 침투형', onPitch: true, slot: 7 },
  { id: 'kor26-lee-kangin', number: 19, name: '이강인', shortName: '이강인', position: 'MF', role: '인버티드 윙어', onPitch: true, slot: 8 },
  { id: 'kor26-oh-hyeongyu', number: 18, name: '오현규', shortName: '오현규', position: 'FW', role: '타깃 포워드', onPitch: true, slot: 9 },
  { id: 'kor26-son-heungmin', number: 7, name: '손흥민', shortName: '손흥민', position: 'FW', role: '인사이드 포워드', onPitch: true, slot: 10 },
  { id: 'kor26-park-jinseob', number: 16, name: '박진섭', shortName: '박진섭', position: 'DF', role: '빌드업 센터백', onPitch: false, slot: null },
  { id: 'kor26-cho-guesung', number: 9, name: '조규성', shortName: '조규성', position: 'FW', role: '라인 브레이커', onPitch: false, slot: null },
  { id: 'kor26-lee-jaesung', number: 10, name: '이재성', shortName: '이재성', position: 'MF', role: '공간 침투형', onPitch: false, slot: null },
  { id: 'kor26-bae-junho', number: 17, name: '배준호', shortName: '배준호', position: 'MF', role: '공간 침투형', onPitch: false, slot: null },
  { id: 'kor26-hwang-heechan', number: 11, name: '황희찬', shortName: '황희찬', position: 'FW', role: '라인 브레이커', onPitch: false, slot: null },
]

const korea2026Squad = korea2026Base.map((player) => ({
  ...player,
  ...(player.slot === null ? { x: 0, y: 0 } : formationPositions['3-4-3'][player.slot]),
}))

const korea2026Evidence: Record<string, PlayerMatchEvidence> = {
  'kor26-kim-seunggyu': { passesCompleted: 35, passesAttempted: 40, pressures: 0, shots: 0, recoveries: 3, carries: 0, note: 'FIFA 전체 경기 보고서' },
  'kor26-lee-hanbeom': { passesCompleted: 77, passesAttempted: 80, pressures: 0, shots: 0, recoveries: 0, carries: 0, note: 'FIFA 전체 경기 보고서' },
  'kor26-kim-minjae': { passesCompleted: 82, passesAttempted: 82, pressures: 0, shots: 1, recoveries: 0, carries: 0, note: '65분 교체 전 · FIFA 전체 경기 보고서' },
  'kor26-lee-gihyuk': { passesCompleted: 86, passesAttempted: 93, pressures: 0, shots: 0, recoveries: 0, carries: 0, note: 'FIFA 전체 경기 보고서' },
  'kor26-hwang-inbeom': { passesCompleted: 71, passesAttempted: 79, pressures: 0, shots: 0, recoveries: 0, carries: 0, note: 'FIFA 전체 경기 보고서' },
  'kor26-seol-youngwoo': { passesCompleted: 39, passesAttempted: 41, pressures: 0, shots: 1, recoveries: 0, carries: 0, note: 'FIFA 전체 경기 보고서' },
  'kor26-castrop': { passesCompleted: 27, passesAttempted: 30, pressures: 0, shots: 0, recoveries: 0, carries: 0, note: '후반 투입 · FIFA 전체 경기 보고서' },
  'kor26-kim-jingyu': { passesCompleted: 51, passesAttempted: 52, pressures: 0, shots: 0, recoveries: 0, carries: 0, note: '후반 투입 · FIFA 전체 경기 보고서' },
  'kor26-lee-kangin': { passesCompleted: 53, passesAttempted: 62, pressures: 0, shots: 1, recoveries: 0, carries: 0, note: 'FIFA 전체 경기 보고서' },
  'kor26-oh-hyeongyu': { passesCompleted: 8, passesAttempted: 11, pressures: 0, shots: 2, recoveries: 0, carries: 0, note: '74분 교체 전 · FIFA 전체 경기 보고서' },
  'kor26-son-heungmin': { passesCompleted: 20, passesAttempted: 24, pressures: 0, shots: 1, recoveries: 0, carries: 0, note: '후반 투입 · FIFA 전체 경기 보고서' },
  'kor26-park-jinseob': { passesCompleted: 45, passesAttempted: 47, pressures: 0, shots: 1, recoveries: 0, carries: 0, note: '65분 교체 투입 · FIFA 전체 경기 보고서' },
}

const korea2026Network: PassNetworkData = {
  completedPasses: 657,
  minimumEdgeCount: 16,
  nodeMetricLabel: '패스 시도',
  nodes: [
    ['김승규', '김승규', 9, 50, 40], ['이한범', '이한범', 31, 79, 80], ['김민재', '김민재', 27, 50, 82],
    ['이기혁', '이기혁', 34, 21, 93], ['설영우', '설영우', 55, 90, 41], ['황인범', '황인범', 53, 58, 79],
    ['카스트로프', '카스트로프', 55, 35, 30], ['김진규', '김진규', 70, 48, 52], ['이강인', '이강인', 72, 82, 62],
    ['오현규', '오현규', 84, 50, 11], ['손흥민', '손흥민', 74, 18, 24], ['박진섭', '박진섭', 18, 68, 47],
  ].map(([id, label, x, y, involvements]) => ({ id: String(id), label: String(label), x: Number(x), y: Number(y), involvements: Number(involvements) })),
  edges: [
    ['이기혁', '김민재', 19], ['박진섭', '이기혁', 19], ['김민재', '이기혁', 18],
    ['이한범', '김민재', 16], ['이한범', '이강인', 16],
  ].map(([from, to, count]) => ({ from: String(from), to: String(to), count: Number(count) })),
}

const southAfrica2026Network: PassNetworkData = {
  completedPasses: 279,
  minimumEdgeCount: 10,
  nodeMetricLabel: '패스 시도',
  nodes: [
    ['Williams', '윌리엄스', 9, 50, 39], ['Mudau', '무다우', 38, 84, 42], ['Okon', '오콘', 28, 65, 51],
    ['Mbokazi', '음보카지', 29, 35, 30], ['Modiba', '모디바', 41, 15, 22], ['Sithole', '시톨레', 54, 61, 53],
    ['Mbatha', '음바타', 55, 38, 34], ['Appollis', '아폴리스', 72, 85, 24], ['Mofokeng', '모포켕', 71, 49, 22],
    ['Maseko', '마세코', 74, 17, 10], ['Makgopa', '마크고파', 85, 50, 7],
  ].map(([id, label, x, y, involvements]) => ({ id: String(id), label: String(label), x: Number(x), y: Number(y), involvements: Number(involvements) })),
  edges: [
    ['Mudau', 'Okon', 18], ['Okon', 'Williams', 14], ['Okon', 'Mudau', 14], ['Mudau', 'Sithole', 11], ['Williams', 'Okon', 10],
  ].map(([from, to, count]) => ({ from: String(from), to: String(to), count: Number(count) })),
}

export const guidedScenarios: Record<GuidedScenarioId, GuidedScenario> = {
  'korea-portugal-65': {
    id: 'korea-portugal-65', order: '01', tournament: '2022 월드컵 · 조별리그', missionType: '득점 필요', difficulty: '입문',
    matchId: matchEvidence.matchId, sourceUrl: matchEvidence.sourceUrl, extractedAt: matchEvidence.extractedAt, windowLabel: '45′–64′', minute: 65, score: [1, 1],
    ours: { name: '대한민국', short: 'KOR', flag: '🇰🇷', status: '승리 필요' }, opponent: { name: '포르투갈', short: 'POR', flag: '🇵🇹', status: '조 1위 확정권' },
    objective: '균형을 잃지 않고 결승골을 만들어라',
    intro: { eyebrow: '경기 개입 연구실 · 2022 월드컵', title: '남은 시간', accent: '25분.', lead: '포르투갈과 1–1. 이대로라면 탈락입니다.\n실제 경기 데이터를 확인하고 65분 개입안을 설계하세요.' },
    briefing: {
      title: '직전 20분의 문제를 먼저 정의합니다.', description: 'StatsBomb 실제 이벤트 45–64분을 기준으로 만든 코칭 스태프용 경기 스냅샷입니다.', diagnosisTitle: '전진보다 압박에 에너지를 썼습니다',
      diagnosisQuote: '압박은 더 많았지만 전진 진입은 4 대 16입니다. 탈취 이후 첫 두 번의 패스를 개선해야 합니다.', contextNumber: '1', contextLabel: '득점\n필요',
      successTitle: '승리 시', successDetail: '다른 경기 결과에 따라 16강 진출', failureTitle: '무승부 시', failureDetail: '조별리그 탈락', optionTitle: '전환 속도를 높일 교체안',
      optionPlayer: '황희찬', optionNumber: 11, optionPosition: '공격수', optionRole: '라인 브레이커', optionAvailability: '65′ 투입 가능', optionTraits: ['이재성 제외', '황희찬 투입', '전환 공격'],
      optionQuote: '실제 경기에서도 65분에 실행된 교체입니다. 여기서는 결과를 복제하지 않고 전술적 이점과 리스크를 비교합니다.',
    },
    comparisonRows: [
      { label: '패스 성공률', ours: '65.6%', opponent: '85.5%', oursValue: 65.6, opponentValue: 85.5, max: 100 },
      { label: '공격 지역 진입', ours: '4회', opponent: '16회', oursValue: 4, opponentValue: 16, max: 16 },
      { label: '박스 진입', ours: '1회', opponent: '3회', oursValue: 1, opponentValue: 3, max: 4 },
      { label: '압박', ours: '28회', opponent: '10회', oursValue: 28, opponentValue: 10, max: 30, invert: true },
    ],
    possessionEstimate: [23.1, 76.9],
    evidence: { ours: matchEvidence.southKorea, opponent: matchEvidence.portugal },
    spatial: { ours: spatialEvidence['South Korea'], opponent: spatialEvidence.Portugal },
    spatialCopy: { entriesTitle: '한국의 진입은 적고 직접적이었습니다', entriesBody: '4회 중 2회가 김승규에서 조규성으로 향한 긴 패스였습니다. 반면 포르투갈은 칸셀루·비티냐를 거쳐 양 측면으로 16회 진입했습니다.', entriesOurs: '4', entriesOpponent: '16', shotsTitle: '양 팀 모두 양질의 슈팅을 만들지 못했습니다', shotsBody: '20분 동안 한국 2회, 포르투갈 1회의 슈팅이 나왔지만 양 팀의 합산 xG는 0.098에 그쳤습니다.', shotsOurs: '0.045', shotsOpponent: '0.053' },
    networks: { ours: koreaNetwork, opponent: portugalNetwork }, networkCopy: { title: '중앙 연결망이 끊기고 전진 패스가 제한됐습니다', body: '노드 크기는 패스 관여, 선 굵기는 같은 방향의 완료 패스 횟수입니다. 포르투갈은 네베스–비티냐–칸셀루 축이 반복적으로 연결됐습니다.' },
    flow: {
      title: '55분 한국의 슈팅 전개', summary: '이재성의 운반에서 시작해 이강인–황인범–조규성을 거쳐 손흥민의 슈팅으로 이어진 실제 이벤트 순서입니다.',
      actions: [
        { id: 'kor-1', time: '55:20', type: 'carry', player: '이재성', start: [19.2, 20.8], end: [24.3, 15.8] },
        { id: 'kor-2', time: '55:21', type: 'pass', player: '이재성', recipient: '조규성', start: [24.3, 15.8], end: [35.2, 22.5] },
        { id: 'kor-3', time: '55:22', type: 'pass', player: '조규성', recipient: '이강인', start: [35.2, 22.5], end: [30.7, 26.3] },
        { id: 'kor-4', time: '55:23', type: 'carry', player: '이강인', start: [30.7, 26.3], end: [31.2, 29.7] },
        { id: 'kor-5', time: '55:23', type: 'pass', player: '이강인', recipient: '황인범', start: [31.2, 29.7], end: [45.8, 45.1] },
        { id: 'kor-6', time: '55:26', type: 'carry', player: '황인범', start: [45.8, 45.1], end: [55.7, 48.8] },
        { id: 'kor-7', time: '55:29', type: 'pass', player: '황인범', recipient: '조규성', start: [55.7, 48.8], end: [72.8, 33] },
        { id: 'kor-8', time: '55:30', type: 'pass', player: '조규성', recipient: '손흥민', start: [72.8, 33], end: [104.5, 54.7] },
        { id: 'kor-9', time: '55:33', type: 'shot', player: '손흥민', start: [104.5, 54.7], end: [105.5, 53.6], outcome: 'Blocked · xG 0.024' },
      ],
    },
    squad: initialSquad, playerEvidence, defaultFormation: '4-3-3', defaultTactics: { pressing: 58, width: 62, tempo: 64, risk: 56 }, selectedPlayerId: 'son-heungmin', impactPlayerId: 'hwang-heechan',
    metricLabels: ['침투 가능성', '볼 순환 안정성', '전환 노출', '고강도 부담'],
    result: { baseline: '45–64분 실제 관측값', impactOn: '황희찬 투입으로 탈취 후 측면 뒷공간을 바로 공격할 선택지가 생깁니다.', impactOff: '기존 인원 유지 시 패스 성공률 격차를 줄일 별도 전진 패턴이 필요합니다.', actualChoice: '실제 경기에서는 65분 황희찬을 투입했습니다.', actualOutcome: '추가시간 손흥민의 전진 패스를 받은 황희찬이 결승골을 기록했습니다.', planOn: '전환 속도 강화안', planOff: '점유 안정화안', operatingSafe: '첫 전진 패스 실패 시 즉시 수비 블록 복귀', operatingRisk: '공을 잃은 직후 중앙 미드필더 한 명은 반드시 잔류' },
  },
  'argentina-netherlands-83': {
    id: 'argentina-netherlands-83', order: '02', tournament: '2022 월드컵 · 8강', missionType: '리드 보호', difficulty: '중급',
    matchId: 3869321, sourceUrl: 'https://github.com/statsbomb/open-data', extractedAt: '2026-07-12', windowLabel: '70′–82′', minute: 83, score: [2, 1],
    ours: { name: '아르헨티나', short: 'ARG', flag: '🇦🇷', status: '한 골 리드' }, opponent: { name: '네덜란드', short: 'NED', flag: '🇳🇱', status: '동점 필요' },
    objective: '박스 앞을 지키되 역습 출구를 잃지 마라',
    intro: { eyebrow: '경기 운영 연구실 · 2022 월드컵', title: '버텨야 할 시간', accent: '10분+.', lead: '네덜란드가 82분 추격골을 넣었습니다.\n긴 패스와 크로스에 대응하면서 리드를 지킬 운영안을 설계하세요.' },
    briefing: {
      title: '상대의 공격 방식이 바뀐 순간을 읽습니다.', description: 'StatsBomb 실제 이벤트 70–82분에서 네덜란드의 직접 전개와 측면 집중을 분리한 코칭 스냅샷입니다.', diagnosisTitle: '경기는 점유전에서 세컨드볼 싸움으로 바뀌었습니다',
      diagnosisQuote: '네덜란드는 13분 동안 긴 패스 17회와 크로스 5회를 시도했습니다. 라인을 내리기만 하면 두 번째 공을 계속 내줄 수 있습니다.', contextNumber: '1', contextLabel: '한 골\n우세',
      successTitle: '리드 유지 시', successDetail: '준결승 진출', failureTitle: '동점 허용 시', failureDetail: '연장전과 체력 리스크', optionTitle: '오른쪽 크로스를 닫을 교체안',
      optionPlayer: '곤살로 몬티엘', optionNumber: 4, optionPosition: '수비수', optionRole: '수비형 풀백', optionAvailability: '83′ 투입 가능', optionTraits: ['몰리나 제외', '몬티엘 투입', '측면 보호'],
      optionQuote: '신선한 풀백으로 크로스 시작점을 압박할 수 있지만, 전진 출구가 줄어들면 수비 블록이 지나치게 낮아질 수 있습니다.',
    },
    comparisonRows: [
      { label: '패스 성공률', ours: '73.9%', opponent: '77.8%', oursValue: 73.9, opponentValue: 77.8, max: 100 },
      { label: '공격 지역 진입', ours: '1회', opponent: '8회', oursValue: 1, opponentValue: 8, max: 8 },
      { label: '긴 패스 시도', ours: '7회', opponent: '17회', oursValue: 7, opponentValue: 17, max: 18 },
      { label: '크로스', ours: '0회', opponent: '5회', oursValue: 0, opponentValue: 5, max: 5 },
    ],
    possessionEstimate: [56.8, 43.2],
    evidence: { ours: { passesAttempted: 23, passesCompleted: 17, passCompletion: 73.9, finalThirdEntries: 1, boxEntries: 1, shots: 1, xg: 0.784, pressures: 7, counterpressures: 1 }, opponent: { passesAttempted: 36, passesCompleted: 28, passCompletion: 77.8, finalThirdEntries: 8, boxEntries: 1, shots: 1, xg: 0.047, pressures: 10, counterpressures: 1 } },
    spatial: argentinaSpatial,
    spatialCopy: { entriesTitle: '네덜란드는 양쪽 바깥에서 반복 진입했습니다', entriesBody: '8번의 공격 지역 진입 중 베르하위스와 각포를 향한 측면 연결이 6번이었습니다. 아르헨티나는 같은 구간 전진 진입이 1회에 그쳤습니다.', entriesOurs: '1', entriesOpponent: '8', shotsTitle: '낮은 xG의 헤더가 경기 양상을 바꿨습니다', shotsBody: '베호르스트의 추격골 xG는 0.047이었습니다. 슈팅 수보다 크로스 시작점과 박스 안 대인 배치가 중요한 장면입니다.', shotsOurs: '0.784', shotsOpponent: '0.047' },
    networks: { ours: argentinaNetwork, opponent: netherlandsNetwork }, networkCopy: { title: '아르헨티나의 전진 연결은 사실상 멈췄습니다', body: '70–82분 완료 패스 기준입니다. 네덜란드는 후방 3명에서 양 측면으로 반복 연결했고, 아르헨티나는 2회 이상 반복된 동일 방향 연결이 없었습니다.' },
    flow: {
      title: '82분 네덜란드의 추격골 전개', summary: '팀버가 오른쪽으로 연결한 뒤 코프메이너르스–베르하위스–베호르스트로 이어진 마지막 5초의 실제 볼 이동입니다.',
      actions: [
        { id: 'ned-1', time: '82:12', type: 'pass', player: '팀버', recipient: '코프메이너르스', start: [73.3, 50.3], end: [82.4, 62.1] },
        { id: 'ned-2', time: '82:13', type: 'pass', player: '코프메이너르스', recipient: '베르하위스', start: [82.4, 62.1], end: [91.8, 78.6] },
        { id: 'ned-3', time: '82:15', type: 'pass', player: '베르하위스', recipient: '베호르스트', start: [91.8, 76.8], end: [108.7, 46.4] },
        { id: 'ned-4', time: '82:17', type: 'shot', player: '베호르스트', start: [108.7, 46.4], end: [120, 37.2], outcome: 'Goal · xG 0.047' },
      ],
    },
    squad: argentinaSquad, playerEvidence: argentinaPlayerEvidence, defaultFormation: '5-3-2', defaultTactics: { pressing: 44, width: 68, tempo: 46, risk: 34 }, selectedPlayerId: 'arg-otamendi', impactPlayerId: 'arg-montiel',
    metricLabels: ['역습 출구', '세컨드볼 안정성', '박스 노출', '수비 부담'],
    result: { baseline: '70–82분 실제 관측값', impactOn: '몬티엘 투입으로 오른쪽 크로스 시작점에 더 빠르게 접근할 수 있습니다.', impactOff: '기존 인원 유지 시 몰리나 앞 공간을 미드필더가 함께 닫는 운영 조건이 필요합니다.', actualChoice: '실제 경기에서는 83분 직후 즉각적인 추가 교체 없이 경기를 이어갔습니다.', actualOutcome: '네덜란드는 추가시간 프리킥 패턴으로 2–2를 만들었고 경기는 연장전으로 향했습니다.', planOn: '측면 봉쇄 강화안', planOff: '블록 균형 유지안', operatingSafe: '공을 걷어낸 뒤 메시·라우타로 중 한 명은 역습 출구로 유지', operatingRisk: '양 풀백이 동시에 내려서면 박스 앞 세컨드볼 담당을 명확히 지정' },
  },
  'korea-south-africa-64': {
    id: 'korea-south-africa-64', order: '03', tournament: '2026 월드컵 · 조별리그', missionType: '추격 설계', difficulty: '심화',
    matchId: 54,
    sourceUrl: 'https://www.fifatrainingcentre.com/media/native/tournaments/fifa-world-cup/2026/PMSR-M54-RSA-V-KOR.pdf',
    sourceName: 'FIFA Training Centre', sourceKind: 'official-report', extractedAt: '2026-07-12',
    sourceNote: '공개된 FIFA 전체 경기 보고서를 바탕으로 64분의 의사결정 상황을 사후 재구성했습니다. 시점별 원시 이벤트 좌표는 사용하지 않습니다.',
    windowLabel: '전체 경기 · 사후 보고서', minute: 64, score: [0, 1],
    ours: { name: '대한민국', short: 'KOR', flag: '🇰🇷', status: '동점 필요' }, opponent: { name: '남아프리카공화국', short: 'RSA', flag: '🇿🇦', status: '한 골 리드' },
    objective: '점유를 슈팅으로 바꾸되 역습 통로를 닫아라',
    intro: { eyebrow: '공식 경기 보고서 재구성 · 2026 월드컵', title: '추격할 시간', accent: '26분+.', lead: '마세코의 선제골로 0–1. 이제 점유만으로는 부족합니다.\nFIFA 공식 경기 보고서를 읽고 64분 이후의 추격안을 다시 설계하세요.' },
    briefing: {
      title: '많이 가진 공을 위협적인 장면으로 바꿉니다.', description: 'FIFA Training Centre 전체 경기 보고서로 되짚는 사후 코칭 리뷰입니다. 64분 당시 실시간 통계로 오해하지 않도록 전체 경기 수치와 재구성 가정을 분리했습니다.', diagnosisTitle: '점유 우위가 박스 안 위협으로 이어지지 않았습니다',
      diagnosisQuote: '한국은 점유 60.9%, 패스 성공률 92%, 공격 지역 리셉션 187회를 기록했지만 슈팅은 8회였습니다. 넓게 돌린 공을 박스 안의 다음 행동으로 연결해야 합니다.', contextNumber: '1', contextLabel: '동점골\n필요',
      successTitle: '동점 시', successDetail: '경기 흐름과 조별리그 생존 가능성 회복', failureTitle: '추가 실점 시', failureDetail: '남아공의 전환 공격에 경기 결정 위험', optionTitle: '후방 안정성을 유지할 교체안',
      optionPlayer: '박진섭', optionNumber: 16, optionPosition: '수비수', optionRole: '빌드업 센터백', optionAvailability: '65′ 투입 가능', optionTraits: ['김민재 제외', '박진섭 투입', '양 윙백 전진'],
      optionQuote: '실제 경기의 65분 교체를 출발점으로 삼되, 공격 숫자를 늘릴 때 후방 세 명의 간격을 어떻게 지킬지 함께 검토합니다.',
    },
    comparisonRows: [
      { label: '패스 성공률', ours: '92%', opponent: '81%', oursValue: 92, opponentValue: 81, max: 100 },
      { label: '공격 지역 리셉션', ours: '187회', opponent: '53회', oursValue: 187, opponentValue: 53, max: 187 },
      { label: '슈팅', ours: '8회', opponent: '13회', oursValue: 8, opponentValue: 13, max: 13 },
      { label: '강제 턴오버', ours: '28회', opponent: '40회', oursValue: 28, opponentValue: 40, max: 40 },
    ],
    possessionEstimate: [60.9, 30.3],
    evidence: {
      ours: { passesAttempted: 718, passesCompleted: 657, passCompletion: 92, finalThirdEntries: 187, boxEntries: 0, shots: 8, xg: 0.74, pressures: 199, counterpressures: 0 },
      opponent: { passesAttempted: 345, passesCompleted: 279, passCompletion: 81, finalThirdEntries: 53, boxEntries: 0, shots: 13, xg: 0.89, pressures: 321, counterpressures: 0 },
    },
    spatial: { ours: { finalThirdEntries: [], shots: [] }, opponent: { finalThirdEntries: [], shots: [] } },
    spatialCopy: { entriesTitle: '시점별 좌표는 공개되지 않았습니다', entriesBody: '공식 보고서의 집계값과 패스 연결을 사용하며, 존재하지 않는 이벤트 좌표를 임의 생성하지 않습니다.', entriesOurs: '187', entriesOpponent: '53', shotsTitle: '전체 경기 슈팅 집계', shotsBody: '한국 8회, 남아공 13회입니다.', shotsOurs: '0.74', shotsOpponent: '0.89' },
    networks: { ours: korea2026Network, opponent: southAfrica2026Network },
    networkCopy: { title: '높은 패스량은 후방 연결에 집중됐습니다', body: 'FIFA 전체 경기 패스 매트릭스의 상위 연결입니다. 한국은 이기혁–김민재 축, 남아공은 무다우–오콘 축의 반복이 두드러졌습니다.' },
    networkPositionBasis: '위치는 64분 전후 포메이션을 바탕으로 한 참조 배치이며 실제 평균 위치 좌표가 아닙니다.',
    flow: { title: '원시 볼 이동 좌표 미공개', summary: '공식 보고서에 연속 이벤트 좌표가 없어 임의 애니메이션을 제공하지 않습니다.', actions: [] },
    squad: korea2026Squad, playerEvidence: korea2026Evidence, defaultFormation: '3-4-3', defaultTactics: { pressing: 62, width: 72, tempo: 68, risk: 58 }, selectedPlayerId: 'kor26-lee-kangin', impactPlayerId: 'kor26-park-jinseob',
    metricLabels: ['박스 진입 위협', '전개 안정성', '역습 노출', '고강도 부담'],
    result: { baseline: 'FIFA 전체 경기 보고서 기반 사후 기준선', impactOn: '박진섭이 후방 중앙을 맡으면 좌우 센터백과 윙백의 전진 타이밍을 분리해 전환 위험을 관리할 수 있습니다.', impactOff: '김민재를 유지한다면 공격 가담보다 전환 시 중앙 통로 보호 역할을 우선해야 합니다.', actualChoice: '실제 경기에서는 65분 김민재 대신 박진섭을 투입했습니다.', actualOutcome: '한국은 점유 우위를 이어갔지만 0–1로 경기를 마쳤습니다.', planOn: '후방 균형형 추격안', planOff: '기존 수비축 유지안', operatingSafe: '한쪽 윙백이 전진할 때 반대쪽 윙백과 중앙 수비수는 잔류', operatingRisk: '양쪽을 동시에 올리면 남아공의 측면 전환에 3대3 상황이 열릴 수 있음' },
    officialReport: {
      inContestPossession: 8.9, lineBreaks: [114, 69], finalThirdReceptions: [187, 53], crosses: [39, 7], forcedTurnovers: [28, 40], secondBalls: [63, 68],
      phases: [
        { label: '비압박 빌드업', ours: 48, opponent: 33 }, { label: '전진 전개', ours: 18, opponent: 12 },
        { label: '공격 지역', ours: 18, opponent: 9 }, { label: '공격 전환', ours: 10, opponent: 15 },
        { label: '중간 블록', ours: 16, opponent: 30 }, { label: '카운터프레스', ours: 9, opponent: 6 },
      ],
    },
  },
}

export const defaultScenarioId: GuidedScenarioId = 'korea-portugal-65'

export const cloneScenarioSquad = (scenario: GuidedScenario) => scenario.squad.map((player) => ({ ...player }))
