export interface TeamWindowEvidence {
  passesAttempted: number
  passesCompleted: number
  passCompletion: number
  finalThirdEntries: number
  boxEntries: number
  shots: number
  xg: number
  pressures: number
  counterpressures: number
}

export interface PlayerMatchEvidence {
  passesCompleted: number
  passesAttempted: number
  pressures: number
  shots: number
  recoveries: number
  carries: number
  note?: string
}

export const matchEvidence = {
  matchId: 3857262,
  competitionId: 43,
  seasonId: 106,
  sourceName: 'StatsBomb Open Data',
  sourceUrl: 'https://github.com/statsbomb/open-data',
  rawEventsUrl: 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/3857262.json',
  extractedAt: '2026-07-11',
  window: { startMinute: 45, endMinuteExclusive: 65 },
  southKorea: {
    passesAttempted: 64,
    passesCompleted: 42,
    passCompletion: 65.6,
    finalThirdEntries: 4,
    boxEntries: 1,
    shots: 2,
    xg: 0.045,
    pressures: 28,
    counterpressures: 8,
  } satisfies TeamWindowEvidence,
  portugal: {
    passesAttempted: 179,
    passesCompleted: 153,
    passCompletion: 85.5,
    finalThirdEntries: 16,
    boxEntries: 3,
    shots: 1,
    xg: 0.053,
    pressures: 10,
    counterpressures: 6,
  } satisfies TeamWindowEvidence,
  actualSubstitution: {
    minute: 65,
    playerOut: '이재성',
    playerIn: '황희찬',
  },
  actualOutcome: '대한민국 2–1 포르투갈',
} as const

export const playerEvidence: Record<string, PlayerMatchEvidence> = {
  'kim-seunggyu': { passesCompleted: 23, passesAttempted: 29, pressures: 0, shots: 0, recoveries: 2, carries: 18 },
  'kim-moonhwan': { passesCompleted: 24, passesAttempted: 28, pressures: 2, shots: 0, recoveries: 1, carries: 14 },
  'kwon-kyungwon': { passesCompleted: 22, passesAttempted: 24, pressures: 2, shots: 0, recoveries: 1, carries: 17 },
  'kim-younggwon': { passesCompleted: 29, passesAttempted: 34, pressures: 3, shots: 1, recoveries: 3, carries: 22 },
  'kim-jinsu': { passesCompleted: 18, passesAttempted: 23, pressures: 2, shots: 0, recoveries: 3, carries: 20 },
  'jung-wooyoung': { passesCompleted: 38, passesAttempted: 45, pressures: 10, shots: 0, recoveries: 3, carries: 33 },
  'hwang-inbeom': { passesCompleted: 24, passesAttempted: 31, pressures: 5, shots: 1, recoveries: 3, carries: 23 },
  'lee-jaesung': { passesCompleted: 12, passesAttempted: 15, pressures: 24, shots: 1, recoveries: 4, carries: 12 },
  'lee-kangin': { passesCompleted: 21, passesAttempted: 26, pressures: 13, shots: 0, recoveries: 1, carries: 17 },
  'cho-guesung': { passesCompleted: 11, passesAttempted: 19, pressures: 9, shots: 1, recoveries: 2, carries: 5 },
  'son-heungmin': { passesCompleted: 10, passesAttempted: 13, pressures: 4, shots: 3, recoveries: 4, carries: 17 },
  'hwang-heechan': { passesCompleted: 0, passesAttempted: 0, pressures: 0, shots: 0, recoveries: 0, carries: 0, note: '65분 교체 투입 전' },
}

export const evidenceMethod = {
  observed: 'StatsBomb 이벤트에서 직접 집계한 45–64분 경기 관측값',
  modeled: '관측값을 기준선으로 포메이션·팀 지시에 휴리스틱 가중치를 적용한 시나리오 비교 점수',
  caution: '시나리오 점수는 승률이나 실제 경기 결과 예측이 아닙니다.',
} as const
