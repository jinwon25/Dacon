import type { GuidedScenario } from '../data/scenarios'

function PhaseBar({ ours, opponent }: { ours: number; opponent: number }) {
  const max = Math.max(ours, opponent, 1)
  return (
    <div className="report-phase-bars" aria-hidden="true">
      <i><b style={{ width: `${ours / max * 100}%` }} /></i>
      <i><b style={{ width: `${opponent / max * 100}%` }} /></i>
    </div>
  )
}

export default function OfficialReportEvidence({ scenario }: { scenario: GuidedScenario }) {
  const report = scenario.officialReport
  if (!report) return null

  const comparisons = [
    { label: '라인 브레이크', ours: report.lineBreaks[0], opponent: report.lineBreaks[1], unit: '회' },
    { label: '공격 지역 리셉션', ours: report.finalThirdReceptions[0], opponent: report.finalThirdReceptions[1], unit: '회' },
    { label: '크로스', ours: report.crosses[0], opponent: report.crosses[1], unit: '회' },
    { label: '강제 턴오버', ours: report.forcedTurnovers[0], opponent: report.forcedTurnovers[1], unit: '회' },
  ]

  return (
    <section className="official-report panel" aria-label="FIFA 공식 경기 보고서 분석">
      <header className="official-report-heading">
        <div className="panel-title"><span>04</span><div><small>경기 양상 · FIFA 전체 경기 보고서</small><h2>점유의 양보다 점유 이후의 행동을 읽습니다</h2></div></div>
        <span className="report-badge">사후 코칭 리뷰</span>
      </header>

      <div className="official-report-grid">
        <article className="report-comparison-card">
          <div className="report-team-labels"><strong>{scenario.ours.name}</strong><span>전체 경기</span><strong>{scenario.opponent.name}</strong></div>
          {comparisons.map((item) => (
            <div className="report-comparison-row" key={item.label}>
              <strong>{item.ours}{item.unit}</strong><span>{item.label}</span><strong>{item.opponent}{item.unit}</strong>
              <PhaseBar ours={item.ours} opponent={item.opponent} />
            </div>
          ))}
        </article>

        <article className="phase-profile-card">
          <div><small>경기 국면 비중</small><strong>어디에서 시간을 썼는가</strong></div>
          <ul>
            {report.phases.map((phase) => (
              <li key={phase.label}>
                <span>{phase.label}</span><b>{phase.ours}%</b><PhaseBar ours={phase.ours} opponent={phase.opponent} /><b>{phase.opponent}%</b>
              </li>
            ))}
          </ul>
        </article>

        <aside className="report-reading">
          <span>코칭 스태프 해석</span>
          <strong>전진은 충분했지만 마지막 행동의 효율이 낮았습니다.</strong>
          <p>한국은 라인 브레이크와 공격 지역 리셉션에서 앞섰지만 슈팅 수는 8 대 13으로 뒤졌습니다. 양쪽 폭을 모두 넓히기보다 한쪽에서 수적 우위를 만든 뒤 반대편 마무리 선수를 남기는 편이 적절합니다.</p>
          <small>이 해석은 공식 집계값을 바탕으로 한 전술적 추론이며 FIFA의 공식 평가 문구가 아닙니다.</small>
        </aside>
      </div>
    </section>
  )
}
