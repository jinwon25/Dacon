import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'

async function openDetailedEvidence(page: Page) {
  const summary = page.locator('.evidence-stack > summary')
  if (await summary.isVisible()) await summary.click()
}

async function showOpponentNetwork(page: Page) {
  const mobileControl = page.locator('.network-tabs button').nth(1)
  if (await mobileControl.isVisible()) await mobileControl.click()
  else await page.locator('.network-pitch-panel.opponent .network-team-select').click()
}

async function moveToTactics(page: Page) {
  const mobileAction = page.locator('.mobile-stage-action')
  if (await mobileAction.isVisible()) await mobileAction.click()
  else await page.getByRole('button', { name: /전술 보드로 이동/ }).click()
}

test('서비스 홈에서 정체성을 확인하고 실제 경기 분석으로 진입한다', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('link', { name: '본문 바로가기' })).toBeAttached()
  await expect(page.getByRole('heading', { name: /경기를 읽고, 전술로 답하다/ })).toBeVisible()
  await expect(page.getByText('실제 월드컵 장면을 데이터로 진단하고')).toBeVisible()
  await page.getByRole('button', { name: /실제 경기에서 시작/ }).click()
  await expect(page.getByRole('heading', { name: /어떤 순간에 개입하시겠습니까/ })).toBeVisible()
  await expect(page.locator('.scenario-picker svg[aria-label="대한민국 국기"]').first()).toBeVisible()
})

test('실제 경기 흐름에서 공간 증거와 결과까지 이동한다', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: /실제 경기에서 시작/ }).click()

  await expect(page.locator('.mission-heading > span')).toHaveText('남은 시간')
  await page.getByRole('button', { name: /데이터 브리핑 시작/ }).click()
  await expect(page.getByRole('heading', { name: '직전 20분의 문제를 먼저 정의합니다.' })).toBeVisible()
  await openDetailedEvidence(page)

  const eventPitch = page.locator('.event-pitch')
  await expect(eventPitch).toBeVisible()
  await expect(eventPitch.locator('.event-pass.korea')).toHaveCount(4)
  await expect(eventPitch.locator('.event-pass.portugal')).toHaveCount(16)

  await page.getByRole('tab', { name: '슈팅 위치' }).click()
  await expect(eventPitch.locator('.event-shot.korea')).toHaveCount(2)
  await expect(eventPitch.locator('.event-shot.portugal')).toHaveCount(1)
  for (const circle of await eventPitch.locator('.event-shot').all()) {
    const box = await circle.boundingBox()
    expect(box).not.toBeNull()
    expect(Math.abs((box?.width ?? 0) - (box?.height ?? 0))).toBeLessThan(1)
  }

  await moveToTactics(page)
  await expect(page).toHaveURL(/#tactics$/)
  await expect(page.getByRole('heading', { name: '균형을 잃지 않고 결승골을 만들어라' })).toBeVisible()
  const formationSelect = page.getByRole('combobox', { name: '포메이션 선택' })
  await expect(formationSelect.locator('option')).toHaveCount(16)
  await formationSelect.selectOption('4-4-2')
  await expect(formationSelect).toHaveValue('4-4-2')

  const firstPlayer = page.locator('.tactics-screen .pitch .player-token').first()
  const positionBefore = await firstPlayer.getAttribute('style')
  await firstPlayer.focus()
  await firstPlayer.press('ArrowRight')
  await expect.poll(() => firstPlayer.getAttribute('style')).not.toBe(positionBefore)

  await page.getByRole('button', { name: /전술 변화 비교 실행/ }).click()
  await expect(page).toHaveURL(/#result$/)
  await expect(page.getByText(/채택 권고|조건부 채택|재설계 필요/).first()).toBeVisible()
})

test('전술 보드에서 여러 교체를 계획하고 직전 교체를 되돌린다', async ({ page }) => {
  await page.goto('/#tactics')
  await expect(page.getByRole('combobox', { name: '포메이션 선택' }).locator('option')).toHaveCount(16)
  await expect(page.locator('.substitution-actions')).toContainText('0 / 5명')

  const pitchBox = await page.locator('.tactics-screen .pitch').boundingBox()
  const onPitch = page.locator('.tactics-screen .pitch .player-token')
  expect(pitchBox).not.toBeNull()
  for (const token of await onPitch.all()) {
    const tokenBox = await token.boundingBox()
    expect(tokenBox).not.toBeNull()
    expect((tokenBox?.y ?? 0) + (tokenBox?.height ?? 0) / 2).toBeGreaterThanOrEqual(pitchBox?.y ?? 0)
    expect((tokenBox?.y ?? 0) + (tokenBox?.height ?? 0) / 2).toBeLessThanOrEqual((pitchBox?.y ?? 0) + (pitchBox?.height ?? 0))
  }
  await onPitch.nth(1).click()
  await page.locator('.tactics-screen .bench-list button:not(:disabled)').first().click()
  await expect(page.locator('.substitution-log li')).toHaveCount(1)

  await onPitch.nth(2).click()
  await page.locator('.tactics-screen .bench-list button:not(:disabled)').first().click()
  await expect(page.locator('.substitution-log li')).toHaveCount(2)
  await expect(page.locator('.substitution-actions')).toContainText('2 / 5명')

  await page.getByRole('button', { name: /직전 취소/ }).click()
  await expect(page.locator('.substitution-log li')).toHaveCount(1)
  await expect(page.locator('.substitution-actions')).toContainText('1 / 5명')
})

test('두 번째 실제 경기에서 패스맵과 볼 흐름을 재생한다', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: /실제 경기에서 시작/ }).click()
  await expect(page.locator('.scenario-picker button')).toHaveCount(3)
  await page.locator('.scenario-picker button').nth(1).click()
  await expect(page.locator('.mission-heading > span')).toHaveText('버텨야 할 시간')
  await expect(page.locator('.mission-heading h1 b')).toHaveText('리드를 지켜라.')
  await expect(page.locator('.match-chip')).toContainText('ARG')
  await page.getByRole('button', { name: /데이터 브리핑 시작/ }).click()

  await expect(page.getByRole('heading', { name: '상대의 공격 방식이 바뀐 순간을 읽습니다.' })).toBeVisible()
  await openDetailedEvidence(page)
  await expect(page.locator('.event-pass.korea')).toHaveCount(1)
  await expect(page.locator('.event-pass.portugal')).toHaveCount(8)
  await expect(page.locator('.pass-network')).toBeVisible()
  await showOpponentNetwork(page)
  await expect(page.locator('.network-pitch.opponent .network-node')).toHaveCount(11)
  await expect(page.locator('.network-pitch.opponent .network-edge').first()).toHaveAttribute('d', /M .+ [LQ] /)
  expect(await page.locator('.network-pitch.opponent .network-edge').evaluateAll((paths) => paths.some((path) => path.getAttribute('d')?.includes(' Q ')))).toBe(true)
  const nodeDiameters = await page.locator('.network-pitch.opponent .network-node-visual').evaluateAll((nodes) => nodes.map((node) => node.getBoundingClientRect().width))
  expect(Math.max(...nodeDiameters)).toBeLessThan(40)
  await page.locator('.network-pitch.opponent .network-node').first().click()
  await expect(page.locator('.network-pitch.opponent .network-node.selected')).toHaveCount(1)
  await expect(page.locator('.network-pitch.opponent .network-node[tabindex="0"]')).toHaveCount(1)

  const flow = page.locator('.ball-flow')
  await expect(flow.locator('.flow-timeline button')).toHaveCount(4)
  await flow.getByRole('button', { name: /실제 흐름 재생/ }).click()
  await expect.poll(async () => flow.locator('.flow-progress i').getAttribute('style')).toContain('25%')
})

test('2026 남아공전은 FIFA 전체 경기 보고서와 재구성 가정을 분리한다', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: /실제 경기에서 시작/ }).click()
  await page.locator('.scenario-picker button').nth(2).click()

  await expect(page.locator('.mission-heading > span')).toHaveText('추격할 시간')
  await expect(page.locator('.mission-heading h1 b')).toHaveText('경기를 되돌려라.')
  await expect(page.locator('.scenario-picker svg[aria-label="대한민국 국기"]')).toHaveCount(2)
  await expect(page.locator('.intro-scoreboard svg[aria-label="남아프리카공화국 국기"]')).toBeVisible()
  await page.getByRole('button', { name: /데이터 브리핑 시작/ }).click()

  await expect(page.getByRole('heading', { name: '많이 가진 공을 위협적인 장면으로 바꿉니다.' })).toBeVisible()
  await openDetailedEvidence(page)
  await expect(page.getByRole('region', { name: 'FIFA 공식 경기 보고서 분석' })).toBeVisible()
  await expect(page.getByText('사후 코칭 리뷰').first()).toBeVisible()
  await expect(page.getByText(/시점별 원시 이벤트 좌표는 사용하지 않습니다/)).toBeVisible()
  await expect(page.locator('.spatial-evidence')).toHaveCount(0)
  await expect(page.locator('.ball-flow')).toHaveCount(0)
  await expect(page.locator('.network-pitch.ours .network-node')).toHaveCount(12)
  await expect(page.locator('.network-context-strip')).toContainText('참조 배치')
  await moveToTactics(page)
  await expect(page.getByRole('heading', { name: '점유를 슈팅으로 바꾸되 역습 통로를 닫아라' })).toBeVisible()
  await expect(page.getByRole('combobox', { name: '포메이션 선택' })).toHaveValue('3-4-3')
  await page.getByRole('button', { name: /전술 변화 비교 실행/ }).click()
  await expect(page.getByText(/FIFA Training Centre 전체 경기 보고서/).last()).toBeVisible()
})

test('범용 스튜디오가 편집값과 포메이션을 자동 저장한다', async ({ page }) => {
  await page.goto('/#studio')
  await expect(page.getByRole('heading', { name: /모든 경기를 위한/ })).toBeVisible()

  const teamInput = page.locator('#studio-context .form-grid input').first()
  await teamInput.fill('테스트 FC')
  await page.getByRole('combobox', { name: '우리 팀 포메이션' }).selectOption('4-1-4-1')
  await expect(page.locator('.board-toolbar strong')).toHaveText('4-1-4-1')
  await expect(page.getByRole('combobox', { name: '우리 팀 포메이션' }).locator('option')).toHaveCount(16)
  await page.getByRole('button', { name: '배치 실행 취소' }).click()
  await expect(page.locator('.board-toolbar strong')).toHaveText('4-3-3')
  await page.getByRole('button', { name: '배치 다시 실행' }).click()
  await expect(page.locator('.board-toolbar strong')).toHaveText('4-1-4-1')
  await page.getByRole('button', { name: '공간 지배도' }).click()
  await expect(page.locator('.control-surface i')).toHaveCount(96)

  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: '전술 파일' }).click(),
  ])
  expect(download.suggestedFilename()).toMatch(/^retactic-.*-4-1-4-1\.json$/)

  await expect.poll(async () => page.evaluate(() => {
    const raw = localStorage.getItem('retactic-universal-studio-v1')
    if (!raw) return null
    const saved = JSON.parse(raw) as { context?: { teamName?: string }; formation?: string }
    return `${saved.context?.teamName}:${saved.formation}`
  })).toBe('테스트 FC:4-1-4-1')
  await page.reload()
  await expect(page.locator('#studio-context .form-grid input').first()).toHaveValue('테스트 FC')
  await expect(page.locator('.board-toolbar strong')).toHaveText('4-1-4-1')
})

test('모바일 화면에서 가로 스크롤 없이 핵심 조작이 보인다', async ({ page }, testInfo) => {
  test.skip(!testInfo.project.name.includes('mobile'), '모바일 프로젝트 전용 검증')
  await page.goto('/#studio')

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
  await expect(page.getByRole('button', { name: 'PNG 저장' }).first()).toBeVisible()
  await expect(page.getByRole('region', { name: '전술 템플릿' })).toBeVisible()

  await page.goto('/')
  await page.getByRole('button', { name: /실제 경기에서 시작/ }).click()
  await page.getByRole('button', { name: /데이터 브리핑 시작/ }).click()
  await expect(page.locator('.mobile-stage-status')).toContainText('진단')
  await expect(page.locator('.mobile-stage-status')).toContainText('2 / 4')
  await expect(page.getByRole('button', { name: /전술 보드/ }).first()).toBeVisible()
  await expect(page.locator('.evidence-stack')).not.toHaveAttribute('open', '')
  await expect(page.locator('.evidence-stack')).not.toHaveAttribute('open', '')
  const guidedOverflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(guidedOverflow).toBeLessThanOrEqual(1)
})
