const scenarios = [
  { matchId: 3857262, startMinute: 45, endMinute: 65, teams: ['South Korea', 'Portugal'] },
  { matchId: 3869321, startMinute: 70, endMinute: 83, teams: ['Argentina', 'Netherlands'] },
]

const isCompletedPass = (event) => event.type?.name === 'Pass' && !event.pass?.outcome
const isInsideBox = (location) => location?.[0] >= 102 && location?.[1] >= 18 && location?.[1] <= 62

function estimateInPlayPossession(windowEvents, teamNames) {
  const possessions = new Map()
  for (const event of windowEvents) {
    const team = event.possession_team?.name
    if (!teamNames.includes(team) || event.possession == null) continue
    const second = event.minute * 60 + event.second
    const eventEnd = second + Math.min(Number(event.duration || 0), 10)
    const key = `${event.possession}:${team}`
    const current = possessions.get(key) ?? { team, start: second, end: eventEnd }
    current.start = Math.min(current.start, second)
    current.end = Math.max(current.end, eventEnd)
    possessions.set(key, current)
  }

  const duration = Object.fromEntries(teamNames.map((team) => [team, 0]))
  for (const possession of possessions.values()) {
    duration[possession.team] += Math.max(1, possession.end - possession.start)
  }
  const total = Object.values(duration).reduce((sum, value) => sum + value, 0)
  return Object.fromEntries(teamNames.map((team) => [team, Number((duration[team] / total * 100).toFixed(1))]))
}

function summarizeTeam(windowEvents, teamName) {
  const teamEvents = windowEvents.filter((event) => event.team?.name === teamName)
  const passes = teamEvents.filter((event) => event.type?.name === 'Pass')
  const completed = passes.filter(isCompletedPass)
  const finalThirdEntries = completed.filter((event) => event.location?.[0] < 80 && event.pass.end_location?.[0] >= 80)
  const boxEntries = completed.filter((event) => !isInsideBox(event.location) && isInsideBox(event.pass.end_location))
  const shots = teamEvents.filter((event) => event.type?.name === 'Shot')

  return {
    passesAttempted: passes.length,
    passesCompleted: completed.length,
    passCompletion: Number((completed.length / passes.length * 100).toFixed(1)),
    finalThirdEntries: finalThirdEntries.length,
    boxEntries: boxEntries.length,
    shots: shots.length,
    xg: Math.round((shots.reduce((sum, event) => sum + event.shot.statsbomb_xg, 0) + Number.EPSILON) * 1000) / 1000,
    pressures: teamEvents.filter((event) => event.type?.name === 'Pressure').length,
    counterpressures: teamEvents.filter((event) => event.counterpress === true).length,
  }
}

const results = []
for (const scenario of scenarios) {
  const source = `https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/${scenario.matchId}.json`
  const response = await fetch(source)
  if (!response.ok) throw new Error(`StatsBomb data request failed: ${response.status}`)
  const events = await response.json()
  const windowEvents = events.filter((event) => event.minute >= scenario.startMinute && event.minute < scenario.endMinute)
  results.push({
    matchId: scenario.matchId,
    source,
    window: { startMinute: scenario.startMinute, endMinuteExclusive: scenario.endMinute },
    inPlayPossessionEstimate: estimateInPlayPossession(windowEvents, scenario.teams),
    teams: Object.fromEntries(scenario.teams.map((team) => [team, summarizeTeam(windowEvents, team)])),
  })
}

console.log(JSON.stringify(results, null, 2))
