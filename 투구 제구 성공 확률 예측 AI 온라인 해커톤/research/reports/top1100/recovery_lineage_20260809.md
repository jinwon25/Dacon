# 초기 작업 계보 복원 기록

원래 작업 트리의 변경을 유지한 채 별도 계보에서 코드·테스트·감사 문서를 정리한 당시 기록입니다. 과거 임시 경로가 현재 존재하거나 현재도 유지된다는 뜻은 아닙니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Aimers lineage recovery — 2026-08-09

- Dirty working tree was **not reset, checkout, or cleaned**.
- Original branch: `codex/baram-2026`.
- Original HEAD: `889b1b83ccd89b84295ac6e593984f4f84aebb95`.
- Recovery pointer created: `codex/recovery-aimers-20260809` at the original HEAD.
- Exact Aimers baseline: `9b0cb8401336916f510669021005e48b4db2c92a`.
- Clean worktree: `.git-worktrees/aimers-clean-20260809`.
- Clean worktree branch: `codex/aimers-clean-20260809`.
- Local `data/`, `model/`, and the immutable `submit_v2.zip` are linked/copied only for validation; they are not staged as source changes.
- Champion SHA-256 remains `FE368AF109EF0BB8103D728F45794A6192599B691BF08DF02A22F31BD2D438A7`.

The commit made from the clean worktree contains Aimers source, tests, configuration, reports and audit manifests only. No public submission or push is performed.
