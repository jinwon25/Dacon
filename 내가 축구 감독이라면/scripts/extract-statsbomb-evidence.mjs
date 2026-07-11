const MATCH_ID = 3857262
const SOURCE = `https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/${MATCH_ID}.json`
const START_MINUTE = 45
const END_MINUTE = 65

const response = await fetch(SOURCE)
if (!response.ok) throw new Error(`StatsBomb data request failed: ${response.status}`)

const events = await response.json()
const windowEvents = events.filter((event) => event.minute >= START_MINUTE && event.minute < END_MINUTE)

const isCompletedPass = (event) => event.type?.name === 'Pass' && !event.pass?.outcome
const isInsideBox = (location) => location?.[0] >= 102 && location?.[1] >= 18 && location?.[1] <= 62

function summarizeTeam(teamName) {
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
    xg: Number(shots.reduce((sum, event) => sum + event.shot.statsbomb_xg, 0).toFixed(3)),
    pressures: teamEvents.filter((event) => event.type?.name === 'Pressure').length,
    counterpressures: teamEvents.filter((event) => event.counterpress === true).length,
  }
}

const result = {
  matchId: MATCH_ID,
  source: SOURCE,
  window: { startMinute: START_MINUTE, endMinuteExclusive: END_MINUTE },
  southKorea: summarizeTeam('South Korea'),
  portugal: summarizeTeam('Portugal'),
}

console.log(JSON.stringify(result, null, 2))
