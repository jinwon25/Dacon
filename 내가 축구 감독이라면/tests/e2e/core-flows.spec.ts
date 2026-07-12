import { expect, test } from '@playwright/test'

test('실제 경기 흐름에서 공간 증거와 결과까지 이동한다', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: /남은 시간/ })).toBeVisible()
  await page.getByRole('button', { name: /데이터 브리핑 시작/ }).click()
  await expect(page.getByRole('heading', { name: '직전 20분의 문제를 먼저 정의합니다.' })).toBeVisible()

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

  await page.getByRole('button', { name: /전술 보드로 이동/ }).click()
  await expect(page.getByRole('heading', { name: '균형을 잃지 않고 결승골을 만들어라' })).toBeVisible()
  const formationSelect = page.getByRole('combobox', { name: '포메이션 선택' })
  await expect(formationSelect.locator('option')).toHaveCount(7)
  await formationSelect.selectOption('4-4-2')
  await expect(formationSelect).toHaveValue('4-4-2')

  await page.getByRole('button', { name: /WHAT-IF 비교 실행/ }).click()
  await expect(page.getByText(/채택 권고|조건부 채택|재설계 필요/).first()).toBeVisible()
})

test('두 번째 실제 경기에서 패스맵과 볼 흐름을 재생한다', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('.scenario-picker button')).toHaveCount(2)
  await page.locator('.scenario-picker button').nth(1).click()
  await expect(page.getByRole('heading', { name: /버텨야 할 시간/ })).toBeVisible()
  await expect(page.locator('.match-chip')).toContainText('ARG')
  await page.getByRole('button', { name: /데이터 브리핑 시작/ }).click()

  await expect(page.getByRole('heading', { name: '상대의 공격 방식이 바뀐 순간을 읽습니다.' })).toBeVisible()
  await expect(page.locator('.event-pass.korea')).toHaveCount(1)
  await expect(page.locator('.event-pass.portugal')).toHaveCount(8)
  await expect(page.locator('.pass-network')).toBeVisible()
  await page.locator('.network-tabs button').nth(1).click()
  await expect(page.locator('.network-pitch.opponent .network-node')).toHaveCount(11)

  const flow = page.locator('.ball-flow')
  await expect(flow.locator('.flow-timeline button')).toHaveCount(4)
  await flow.getByRole('button', { name: /실제 흐름 재생/ }).click()
  await expect.poll(async () => flow.locator('.flow-progress i').getAttribute('style')).toContain('25%')
})

test('범용 스튜디오가 편집값과 포메이션을 자동 저장한다', async ({ page }) => {
  await page.goto('/#studio')
  await expect(page.getByRole('heading', { name: /모든 경기를 위한/ })).toBeVisible()

  const teamInput = page.locator('#studio-context .form-grid input').first()
  await teamInput.fill('테스트 FC')
  await page.getByRole('combobox', { name: '우리 팀 포메이션' }).selectOption('4-1-4-1')
  await expect(page.locator('.board-toolbar strong')).toHaveText('4-1-4-1')
  await expect(page.getByRole('combobox', { name: '우리 팀 포메이션' }).locator('option')).toHaveCount(7)

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
})
