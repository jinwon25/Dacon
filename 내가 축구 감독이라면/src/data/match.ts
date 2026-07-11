import type { Coordinate, FormationKey, Player } from '../types'

export const formationPositions: Record<FormationKey, Coordinate[]> = {
  '4-3-3': [
    { x: 50, y: 91 },
    { x: 84, y: 73 }, { x: 62, y: 78 }, { x: 38, y: 78 }, { x: 16, y: 73 },
    { x: 50, y: 59 }, { x: 69, y: 48 }, { x: 31, y: 48 },
    { x: 81, y: 27 }, { x: 50, y: 18 }, { x: 19, y: 27 },
  ],
  '4-2-3-1': [
    { x: 50, y: 91 },
    { x: 84, y: 73 }, { x: 62, y: 78 }, { x: 38, y: 78 }, { x: 16, y: 73 },
    { x: 62, y: 59 }, { x: 38, y: 59 }, { x: 50, y: 42 },
    { x: 81, y: 34 }, { x: 50, y: 18 }, { x: 19, y: 34 },
  ],
  '4-4-2': [
    { x: 50, y: 91 },
    { x: 84, y: 73 }, { x: 62, y: 78 }, { x: 38, y: 78 }, { x: 16, y: 73 },
    { x: 62, y: 53 }, { x: 38, y: 53 }, { x: 82, y: 47 }, { x: 18, y: 47 },
    { x: 63, y: 22 }, { x: 37, y: 22 },
  ],
  '4-1-4-1': [
    { x: 50, y: 91 },
    { x: 84, y: 73 }, { x: 62, y: 78 }, { x: 38, y: 78 }, { x: 16, y: 73 },
    { x: 50, y: 63 }, { x: 63, y: 46 }, { x: 37, y: 46 },
    { x: 82, y: 40 }, { x: 50, y: 18 }, { x: 18, y: 40 },
  ],
  '4-3-1-2': [
    { x: 50, y: 91 },
    { x: 84, y: 73 }, { x: 62, y: 78 }, { x: 38, y: 78 }, { x: 16, y: 73 },
    { x: 50, y: 62 }, { x: 68, y: 50 }, { x: 32, y: 50 },
    { x: 50, y: 34 }, { x: 63, y: 18 }, { x: 37, y: 18 },
  ],
  '3-4-3': [
    { x: 50, y: 91 },
    { x: 72, y: 76 }, { x: 50, y: 80 }, { x: 28, y: 76 },
    { x: 84, y: 52 }, { x: 61, y: 56 }, { x: 39, y: 56 }, { x: 16, y: 52 },
    { x: 79, y: 27 }, { x: 50, y: 18 }, { x: 21, y: 27 },
  ],
  '5-3-2': [
    { x: 50, y: 91 },
    { x: 88, y: 67 }, { x: 69, y: 76 }, { x: 50, y: 80 }, { x: 31, y: 76 }, { x: 12, y: 67 },
    { x: 70, y: 50 }, { x: 50, y: 57 }, { x: 30, y: 50 },
    { x: 64, y: 24 }, { x: 36, y: 24 },
  ],
}

const starters: Omit<Player, 'x' | 'y'>[] = [
  { id: 'kim-seunggyu', number: 1, name: '김승규', shortName: '김승규', position: 'GK', role: '스위퍼 키퍼', onPitch: true, slot: 0 },
  { id: 'kim-moonhwan', number: 15, name: '김문환', shortName: '김문환', position: 'DF', role: '오버래핑 풀백', onPitch: true, slot: 1 },
  { id: 'kwon-kyungwon', number: 20, name: '권경원', shortName: '권경원', position: 'DF', role: '커버 센터백', onPitch: true, slot: 2 },
  { id: 'kim-younggwon', number: 19, name: '김영권', shortName: '김영권', position: 'DF', role: '빌드업 센터백', onPitch: true, slot: 3 },
  { id: 'kim-jinsu', number: 3, name: '김진수', shortName: '김진수', position: 'DF', role: '공격형 풀백', onPitch: true, slot: 4 },
  { id: 'jung-wooyoung', number: 5, name: '정우영', shortName: '정우영', position: 'MF', role: '앵커', onPitch: true, slot: 5 },
  { id: 'hwang-inbeom', number: 6, name: '황인범', shortName: '황인범', position: 'MF', role: '딥라잉 플레이메이커', onPitch: true, slot: 6 },
  { id: 'lee-jaesung', number: 10, name: '이재성', shortName: '이재성', position: 'MF', role: '공간 침투형', onPitch: true, slot: 7 },
  { id: 'lee-kangin', number: 18, name: '이강인', shortName: '이강인', position: 'MF', role: '인버티드 윙어', onPitch: true, slot: 8 },
  { id: 'cho-guesung', number: 9, name: '조규성', shortName: '조규성', position: 'FW', role: '타깃 포워드', onPitch: true, slot: 9 },
  { id: 'son-heungmin', number: 7, name: '손흥민', shortName: '손흥민', position: 'FW', role: '인사이드 포워드', onPitch: true, slot: 10 },
]

const substitutes: Omit<Player, 'x' | 'y'>[] = [
  { id: 'hwang-heechan', number: 11, name: '황희찬', shortName: '황희찬', position: 'FW', role: '라인 브레이커', onPitch: false, slot: null },
  { id: 'son-junho', number: 13, name: '손준호', shortName: '손준호', position: 'MF', role: '볼 위닝 미드필더', onPitch: false, slot: null },
  { id: 'hwang-uijo', number: 16, name: '황의조', shortName: '황의조', position: 'FW', role: '포처', onPitch: false, slot: null },
  { id: 'na-sangho', number: 17, name: '나상호', shortName: '나상호', position: 'MF', role: '와이드 플레이메이커', onPitch: false, slot: null },
  { id: 'paik-seungho', number: 8, name: '백승호', shortName: '백승호', position: 'MF', role: '전진형 플레이메이커', onPitch: false, slot: null },
]

export const initialSquad: Player[] = [...starters, ...substitutes].map((player) => {
  const coordinate = player.slot === null ? { x: 0, y: 0 } : formationPositions['4-3-3'][player.slot]
  return { ...player, ...coordinate }
})

export const roleOptions: Record<Player['position'], string[]> = {
  GK: ['스위퍼 키퍼', '수비형 골키퍼'],
  DF: ['공격형 풀백', '수비형 풀백', '빌드업 센터백', '커버 센터백'],
  MF: ['앵커', '딥라잉 플레이메이커', '공간 침투형', '볼 위닝 미드필더'],
  FW: ['라인 브레이커', '타깃 포워드', '인사이드 포워드', '포처'],
}
