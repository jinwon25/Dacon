import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(fileURLToPath(new URL('.', import.meta.url)), '..')
const read = (path) => readFileSync(resolve(root, path), 'utf8')
const assert = (condition, message) => {
  if (!condition) throw new Error(message)
  console.log(`✓ ${message}`)
}

const html = read('dist/index.html')
assert(html.includes('<html lang="ko">'), '한국어 문서 언어가 지정되어 있습니다.')
assert(html.includes('RE:TACTIC'), '프로덕션 문서 제목이 포함되어 있습니다.')

const assetRefs = [...html.matchAll(/(?:src|href)="([^"#]+)"/g)]
  .map((match) => match[1])
  .filter((path) => path.includes('assets/'))
assert(assetRefs.length >= 2, '프로덕션 JS·CSS 에셋을 찾았습니다.')
for (const asset of assetRefs) {
  assert(!asset.startsWith('/'), `GitHub Pages용 상대 경로를 사용합니다: ${asset}`)
  assert(existsSync(resolve(root, 'dist', asset)), `빌드 에셋이 존재합니다: ${asset}`)
}

const bundle = assetRefs
  .filter((asset) => asset.endsWith('.js'))
  .map((asset) => readFileSync(resolve(root, 'dist', asset), 'utf8'))
  .join('\n')
assert(bundle.includes('StatsBomb Open Data'), '데이터 출처 문구가 번들에 포함되어 있습니다.')
assert(bundle.includes('시나리오 점수는 승률이나 실제 경기 결과 예측이 아닙니다.'), '휴리스틱 결과 주의 문구가 포함되어 있습니다.')
assert(bundle.includes('범용 전술 보드'), '범용 전술 스튜디오가 번들에 포함되어 있습니다.')
assert(bundle.includes('인플레이 점유 추정'), '점유 추정의 성격이 화면에 명시되어 있습니다.')
assert(bundle.includes('전술 변화 비교 실행'), '전술 변화 비교 흐름이 번들에 포함되어 있습니다.')
assert(bundle.includes('FIFA Training Centre'), '2026 공식 경기 보고서 출처가 번들에 포함되어 있습니다.')

const evidence = read('src/data/evidence.ts')
assert(/matchId:\s*3857262/.test(evidence), '검증 경기 ID 3857262가 고정되어 있습니다.')
assert(/finalThirdEntries:\s*4/.test(evidence) && /finalThirdEntries:\s*16/.test(evidence), '공격 지역 진입 집계값이 포함되어 있습니다.')

const scenarios = read('src/data/scenarios.ts')
assert(scenarios.includes('3869321'), '두 번째 검증 경기 ID 3869321이 고정되어 있습니다.')
assert(scenarios.includes("'korea-south-africa-64'"), '2026 남아공전 사후 재구성 시나리오가 포함되어 있습니다.')
assert(scenarios.includes('possessionEstimate: [60.9, 30.3]'), '2026 남아공전 FIFA 공식 점유율이 포함되어 있습니다.')
assert(scenarios.includes('inContestPossession: 8.9'), '2026 남아공전 점유 경합 비율이 포함되어 있습니다.')
assert(scenarios.includes('possessionEstimate: [23.1, 76.9]') && scenarios.includes('possessionEstimate: [56.8, 43.2]'), '두 시나리오의 점유 추정값이 포함되어 있습니다.')
assert((scenarios.match(/id: 'ned-/g) ?? []).length === 4, '네덜란드 추격골 볼 흐름 4개 이벤트를 확인했습니다.')

const spatial = read('src/data/spatialEvidence.ts')
const portugalStart = spatial.indexOf('Portugal:')
const koreaBlock = spatial.slice(spatial.indexOf("'South Korea':"), portugalStart)
const portugalBlock = spatial.slice(portugalStart)
const countItems = (block, key) => {
  const keyIndex = block.indexOf(`${key}: [`)
  const start = block.indexOf('[', keyIndex)
  let depth = 0
  for (let index = start; index < block.length; index += 1) {
    if (block[index] === '[') depth += 1
    if (block[index] === ']') depth -= 1
    if (depth === 0) return (block.slice(start, index + 1).match(/\{ minute:/g) ?? []).length
  }
  return -1
}
assert(countItems(koreaBlock, 'finalThirdEntries') === 4, '한국 공격 지역 진입 좌표 4건을 확인했습니다.')
assert(countItems(koreaBlock, 'shots') === 2, '한국 슈팅 좌표 2건을 확인했습니다.')
assert(countItems(portugalBlock, 'finalThirdEntries') === 16, '포르투갈 공격 지역 진입 좌표 16건을 확인했습니다.')
assert(countItems(portugalBlock, 'shots') === 1, '포르투갈 슈팅 좌표 1건을 확인했습니다.')

const freezeFrame = read('src/data/freezeFrameEvidence.ts')
assert(freezeFrame.includes('c424a395-49ea-4e9a-abcd-f008143008eb'), '조규성→손흥민 패스 360 이벤트 ID를 확인했습니다.')
assert(freezeFrame.includes('13d00263-c09d-4690-a00f-3abc46a2caff'), '손흥민 슈팅 360 이벤트 ID를 확인했습니다.')
assert((freezeFrame.match(/id: 'pass-/g) ?? []).length === 13, '패스 시점 360 가시 선수 13명을 확인했습니다.')
assert((freezeFrame.match(/id: 'shot-/g) ?? []).length === 8, '슈팅 시점 360 가시 선수 8명을 확인했습니다.')

console.log('\nRE:TACTIC 정적 배포 검증을 통과했습니다.')
