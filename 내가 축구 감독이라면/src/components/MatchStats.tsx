import { useState } from 'react'
import type { GuidedScenario } from '../data/scenarios'

type DetailView = 'passing' | 'defending'

interface StatRowData {
  label: string
  description: string
  ours: string
  opponent: string
  oursValue: number
  opponentValue: number
}

const formatXg = (value: number) => value.toFixed(3).replace(/^0/, '')

function ComparisonBar({ ours, opponent }: { ours: number; opponent: number }) {
  const max = Math.max(ours, opponent, 1)
  return (
    <div className="match-stat-bar" aria-hidden="true">
      <span><i style={{ width: `${ours / max * 100}%` }} /></span>
      <span><i style={{ width: `${opponent / max * 100}%` }} /></span>
    </div>
  )
}

function DetailRow({ row }: { row: StatRowData }) {
  return (
    <div className="match-stat-row">
      <strong className={row.oursValue > row.opponentValue ? 'leading' : ''}>{row.ours}</strong>
      <span><b>{row.label}</b><small>{row.description}</small></span>
      <strong className={row.opponentValue > row.oursValue ? 'leading opponent' : ''}>{row.opponent}</strong>
      <ComparisonBar ours={row.oursValue} opponent={row.opponentValue} />
    </div>
  )
}

export default function MatchStats({ scenario }: { scenario: GuidedScenario }) {
  const [view, setView] = useState<DetailView>('passing')
  const ours = scenario.evidence.ours
  const opponent = scenario.evidence.opponent
  const details: Record<DetailView, StatRowData[]> = {
    passing: [
      {
        label: '패스 성공', description: '성공 / 시도 · 성공률',
        ours: `${ours.passesCompleted}/${ours.passesAttempted} · ${ours.passCompletion.toFixed(1)}%`,
        opponent: `${opponent.passesCompleted}/${opponent.passesAttempted} · ${opponent.passCompletion.toFixed(1)}%`,
        oursValue: ours.passCompletion, opponentValue: opponent.passCompletion,
      },
      {
        label: '공격 지역 진입', description: '패스 도착점이 파이널 서드인 횟수',
        ours: `${ours.finalThirdEntries}회`, opponent: `${opponent.finalThirdEntries}회`,
        oursValue: ours.finalThirdEntries, opponentValue: opponent.finalThirdEntries,
      },
      {
        label: '박스 진입', description: '패스 도착점이 상대 페널티박스인 횟수',
        ours: `${ours.boxEntries}회`, opponent: `${opponent.boxEntries}회`,
        oursValue: ours.boxEntries, opponentValue: opponent.boxEntries,
      },
    ],
    defending: [
      {
        label: '압박', description: '볼 보유자에게 압력을 가한 이벤트',
        ours: `${ours.pressures}회`, opponent: `${opponent.pressures}회`,
        oursValue: ours.pressures, opponentValue: opponent.pressures,
      },
      {
        label: '카운터프레싱', description: '볼을 잃은 직후 기록된 압박',
        ours: `${ours.counterpressures}회`, opponent: `${opponent.counterpressures}회`,
        oursValue: ours.counterpressures, opponentValue: opponent.counterpressures,
      },
    ],
  }

  const topStats = [
    {
      label: '인플레이 점유 추정', ours: `${scenario.possessionEstimate[0].toFixed(1)}%`, opponent: `${scenario.possessionEstimate[1].toFixed(1)}%`,
      oursValue: scenario.possessionEstimate[0], opponentValue: scenario.possessionEstimate[1],
    },
    { label: '기대득점 xG', ours: formatXg(ours.xg), opponent: formatXg(opponent.xg), oursValue: ours.xg, opponentValue: opponent.xg },
    { label: '슈팅', ours: `${ours.shots}`, opponent: `${opponent.shots}`, oursValue: ours.shots, opponentValue: opponent.shots },
  ]

  return (
    <article className="analysis-card match-stats-card pitch-analysis">
      <div className="card-heading">
        <span>01</span>
        <div><small>MATCH STATS · {scenario.windowLabel}</small><h2>{scenario.briefing.diagnosisTitle}</h2></div>
      </div>

      <div className="match-stat-teams" aria-label={`${scenario.ours.name}와 ${scenario.opponent.name} 경기 통계 비교`}>
        <strong><i>{scenario.ours.flag}</i>{scenario.ours.name}</strong>
        <span>관측 구간</span>
        <strong>{scenario.opponent.name}<i>{scenario.opponent.flag}</i></strong>
      </div>

      <div className="top-stat-grid">
        {topStats.map((stat) => (
          <div className="top-stat" key={stat.label}>
            <strong>{stat.ours}</strong><span>{stat.label}</span><strong>{stat.opponent}</strong>
            <ComparisonBar ours={stat.oursValue} opponent={stat.opponentValue} />
          </div>
        ))}
      </div>

      <div className="match-stat-tabs" role="tablist" aria-label="세부 통계 분류">
        <button type="button" role="tab" aria-selected={view === 'passing'} className={view === 'passing' ? 'active' : ''} onClick={() => setView('passing')}>패스·전개</button>
        <button type="button" role="tab" aria-selected={view === 'defending'} className={view === 'defending' ? 'active' : ''} onClick={() => setView('defending')}>수비·압박</button>
      </div>
      <div className="match-stat-details" role="tabpanel">
        {details[view].map((row) => <DetailRow key={row.label} row={row} />)}
      </div>

      <p className="stat-method"><b>산출 기준</b> 점유율은 이벤트별 포제션의 인플레이 지속시간을 합산한 구간 추정치이며, 나머지는 StatsBomb 이벤트 직접 집계입니다.</p>
      <p className="coach-quote">“{scenario.briefing.diagnosisQuote}”</p>
    </article>
  )
}
