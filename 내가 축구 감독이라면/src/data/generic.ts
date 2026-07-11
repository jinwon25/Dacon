import { formationPositions } from './match'
import type { FormationKey, Player } from '../types'

const starters: Omit<Player, 'x' | 'y'>[] = [
  { id: 'generic-gk', number: 1, name: '골키퍼', shortName: 'GK', position: 'GK', role: '스위퍼 키퍼', onPitch: true, slot: 0 },
  { id: 'generic-rb', number: 2, name: '오른쪽 풀백', shortName: 'RB', position: 'DF', role: '공격형 풀백', onPitch: true, slot: 1 },
  { id: 'generic-rcb', number: 4, name: '오른쪽 센터백', shortName: 'RCB', position: 'DF', role: '커버 센터백', onPitch: true, slot: 2 },
  { id: 'generic-lcb', number: 5, name: '왼쪽 센터백', shortName: 'LCB', position: 'DF', role: '빌드업 센터백', onPitch: true, slot: 3 },
  { id: 'generic-lb', number: 3, name: '왼쪽 풀백', shortName: 'LB', position: 'DF', role: '공격형 풀백', onPitch: true, slot: 4 },
  { id: 'generic-dm', number: 6, name: '수비형 미드필더', shortName: 'DM', position: 'MF', role: '앵커', onPitch: true, slot: 5 },
  { id: 'generic-rcm', number: 8, name: '오른쪽 미드필더', shortName: 'RCM', position: 'MF', role: '딥라잉 플레이메이커', onPitch: true, slot: 6 },
  { id: 'generic-lcm', number: 10, name: '왼쪽 미드필더', shortName: 'LCM', position: 'MF', role: '공간 침투형', onPitch: true, slot: 7 },
  { id: 'generic-rw', number: 7, name: '오른쪽 윙어', shortName: 'RW', position: 'FW', role: '인사이드 포워드', onPitch: true, slot: 8 },
  { id: 'generic-st', number: 9, name: '스트라이커', shortName: 'ST', position: 'FW', role: '타깃 포워드', onPitch: true, slot: 9 },
  { id: 'generic-lw', number: 11, name: '왼쪽 윙어', shortName: 'LW', position: 'FW', role: '라인 브레이커', onPitch: true, slot: 10 },
]

const bench: Omit<Player, 'x' | 'y'>[] = [
  { id: 'generic-b1', number: 12, name: '교체 골키퍼', shortName: 'B1', position: 'GK', role: '수비형 골키퍼', onPitch: false, slot: null },
  { id: 'generic-b2', number: 13, name: '교체 수비수', shortName: 'B2', position: 'DF', role: '수비형 풀백', onPitch: false, slot: null },
  { id: 'generic-b3', number: 14, name: '교체 미드필더', shortName: 'B3', position: 'MF', role: '볼 위닝 미드필더', onPitch: false, slot: null },
  { id: 'generic-b4', number: 15, name: '교체 공격수', shortName: 'B4', position: 'FW', role: '포처', onPitch: false, slot: null },
  { id: 'generic-b5', number: 16, name: '교체 윙어', shortName: 'B5', position: 'FW', role: '라인 브레이커', onPitch: false, slot: null },
]

export const createGenericSquad = (): Player[] => [...starters, ...bench].map((player) => {
  const coordinate = player.slot === null ? { x: 0, y: 0 } : formationPositions['4-3-3'][player.slot]
  return { ...player, ...coordinate }
})

export const createOpponentSquad = (formation: FormationKey = '4-3-3'): Player[] => starters.map((player, index) => {
  const coordinate = formationPositions[formation][index]
  return {
    ...player,
    id: `opponent-${index}`,
    name: `상대 ${player.shortName}`,
    shortName: player.shortName,
    number: index + 1,
    x: coordinate.x,
    y: 100 - coordinate.y,
  }
})
