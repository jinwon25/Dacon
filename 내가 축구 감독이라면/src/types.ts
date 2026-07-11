export type Stage = 'intro' | 'briefing' | 'tactics' | 'result'

export type FormationKey =
  | '4-3-3'
  | '4-2-3-1'
  | '4-4-2'
  | '4-1-4-1'
  | '4-3-1-2'
  | '3-4-3'
  | '5-3-2'

export type Position = 'GK' | 'DF' | 'MF' | 'FW'

export interface Coordinate {
  x: number
  y: number
}

export interface Player {
  id: string
  number: number
  name: string
  shortName: string
  position: Position
  role: string
  onPitch: boolean
  slot: number | null
  x: number
  y: number
}

export interface Tactics {
  pressing: number
  width: number
  tempo: number
  risk: number
}

export interface Metrics {
  threat: number
  control: number
  exposure: number
  fatigue: number
}
