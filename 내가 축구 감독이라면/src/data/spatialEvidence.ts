export interface SpatialPass {
  minute: number
  player: string
  recipient: string
  start: [number, number]
  end: [number, number]
}

export interface SpatialShot {
  minute: number
  player: string
  location: [number, number]
  xg: number
  outcome: string
}

export const spatialEvidence: Record<'South Korea' | 'Portugal', { finalThirdEntries: SpatialPass[]; shots: SpatialShot[] }> = {
  'South Korea': {
    finalThirdEntries: [
      { minute: 45, player: 'Seung-Gyu Kim', recipient: 'Gue-Sung Cho', start: [17.6, 65.9], end: [73.2, 78.2] },
      { minute: 47, player: 'Seung-Gyu Kim', recipient: 'Gue-Sung Cho', start: [19.2, 41.5], end: [72.0, 8.2] },
      { minute: 52, player: 'Jin-Su Kim', recipient: 'Gue-Sung Cho', start: [42.2, 4.6], end: [77.9, 21.4] },
      { minute: 55, player: 'Gue-Sung Cho', recipient: 'Heung-Min Son', start: [60.7, 41.2], end: [87.1, 68.4] },
    ],
    shots: [
      { minute: 55, player: 'Heung-Min Son', location: [87.1, 68.4], xg: 0.024, outcome: 'Blocked' },
      { minute: 56, player: 'Jae-Sung Lee', location: [81.6, 63.6], xg: 0.021, outcome: 'Off T' },
    ],
  },
  Portugal: {
    finalThirdEntries: [
      { minute: 45, player: 'Matheus Nunes', recipient: 'João Mário', start: [56.2, 54.5], end: [67.5, 34.0] },
      { minute: 48, player: 'Rúben Neves', recipient: 'João Cancelo', start: [64.5, 30.6], end: [77.6, 2.8] },
      { minute: 49, player: 'Diogo Dalot', recipient: 'Matheus Nunes', start: [58.6, 91.1], end: [67.7, 95.5] },
      { minute: 50, player: 'João Cancelo', recipient: 'João Mário', start: [60.2, 22.0], end: [78.4, 8.0] },
      { minute: 51, player: 'Vitinha', recipient: 'João Cancelo', start: [59.0, 27.2], end: [68.2, 4.2] },
      { minute: 54, player: 'Vitinha', recipient: 'João Cancelo', start: [59.8, 35.0], end: [72.2, 2.0] },
      { minute: 58, player: 'João Cancelo', recipient: 'Vitinha', start: [65.8, 19.0], end: [72.4, 3.0] },
      { minute: 58, player: 'Vitinha', recipient: 'João Mário', start: [66.4, 22.9], end: [68.8, 20.4] },
      { minute: 58, player: 'Matheus Nunes', recipient: 'João Cancelo', start: [66.1, 54.4], end: [79.0, 6.8] },
      { minute: 59, player: 'Vitinha', recipient: 'Ricardo Horta', start: [45.8, 30.6], end: [81.2, 89.8] },
      { minute: 60, player: 'Pepe', recipient: 'Matheus Nunes', start: [56.8, 65.6], end: [77.2, 96.2] },
      { minute: 60, player: 'António Silva', recipient: 'João Mário', start: [57.8, 36.0], end: [66.8, 28.5] },
      { minute: 61, player: 'Rúben Neves', recipient: 'João Cancelo', start: [57.8, 73.8], end: [89.4, 27.2] },
      { minute: 61, player: 'João Cancelo', recipient: 'Vitinha', start: [66.4, 5.2], end: [76.8, 4.0] },
      { minute: 62, player: 'Rúben Neves', recipient: 'João Mário', start: [63.3, 38.1], end: [82.4, 6.8] },
      { minute: 63, player: 'Rúben Neves', recipient: 'João Mário', start: [56.8, 43.4], end: [69.9, 26.4] },
    ],
    shots: [
      { minute: 46, player: 'Ricardo Horta', location: [84.8, 56.5], xg: 0.053, outcome: 'Blocked' },
    ],
  },
}
