export interface FreezeFramePlayer {
  id: string
  label: string
  team: 'korea' | 'portugal'
  x: number
  y: number
  actor?: boolean
  keeper?: boolean
}

export const koreaPortugal360 = {
  source: 'StatsBomb Open Data 360',
  matchId: 3857262,
  pass: {
    eventId: 'c424a395-49ea-4e9a-abcd-f008143008eb',
    label: '조규성 → 손흥민 패스 시점',
    players: [
      { id: 'pass-p1', label: 'P', team: 'portugal', x: 49.2, y: 30.1 },
      { id: 'pass-p2', label: 'P', team: 'portugal', x: 52.7, y: 55.6 },
      { id: 'pass-k1', label: 'K', team: 'korea', x: 52.8, y: 51.2 },
      { id: 'pass-p3', label: 'P', team: 'portugal', x: 53.1, y: 40.1 },
      { id: 'pass-p4', label: 'P', team: 'portugal', x: 55.0, y: 56.0 },
      { id: 'pass-p5', label: 'P', team: 'portugal', x: 57.3, y: 54.9 },
      { id: 'pass-cho', label: '조', team: 'korea', x: 60.7, y: 41.2, actor: true },
      { id: 'pass-k2', label: 'K', team: 'korea', x: 61.5, y: 35.7 },
      { id: 'pass-p6', label: 'P', team: 'portugal', x: 66.2, y: 39.6 },
      { id: 'pass-p7', label: 'P', team: 'portugal', x: 67.3, y: 59.2 },
      { id: 'pass-p8', label: 'P', team: 'portugal', x: 68.0, y: 32.5 },
      { id: 'pass-k3', label: 'K', team: 'korea', x: 70.2, y: 53.9 },
      { id: 'pass-p9', label: 'P', team: 'portugal', x: 71.8, y: 50.1 },
    ] satisfies FreezeFramePlayer[],
  },
  shot: {
    eventId: '13d00263-c09d-4690-a00f-3abc46a2caff',
    label: '손흥민 슈팅 시점',
    players: [
      { id: 'shot-p1', label: 'P', team: 'portugal', x: 69.1, y: 52.3 },
      { id: 'shot-k1', label: 'K', team: 'korea', x: 77.5, y: 46.0 },
      { id: 'shot-p2', label: 'P', team: 'portugal', x: 79.9, y: 63.7 },
      { id: 'shot-p3', label: 'P', team: 'portugal', x: 80.4, y: 46.4 },
      { id: 'shot-p4', label: 'P', team: 'portugal', x: 81.4, y: 51.0 },
      { id: 'shot-son', label: '손', team: 'korea', x: 87.1, y: 68.4, actor: true },
      { id: 'shot-p5', label: 'P', team: 'portugal', x: 87.2, y: 65.3 },
      { id: 'shot-gk', label: 'GK', team: 'portugal', x: 97.6, y: 54.3, keeper: true },
    ] satisfies FreezeFramePlayer[],
  },
} as const
