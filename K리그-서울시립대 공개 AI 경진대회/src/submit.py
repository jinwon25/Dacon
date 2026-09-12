"""DACON 제출 API로 submission CSV를 올리는 안전한 헬퍼.

⚠️ 개인 Token은 '비밀번호'입니다. 코드나 git에 절대 넣지 마세요.
   아래 둘 중 하나로 제공합니다(둘 다 git에 안 올라감):

   (1) 환경변수:   DACON_TOKEN, DACON_TEAM
   (2) 비밀파일:   이 대회 폴더에  .env  파일을 만들고 ↓ 두 줄 작성
           DACON_TOKEN=발급받은_토큰
           DACON_TEAM=내_팀이름
       (.env 는 .gitignore 에 걸려 있어 커밋되지 않습니다)

토큰 발급:  데이콘 > 마이페이지 > 계정관리
팀이름:     대회 상세페이지 > 팀 탭

준비물(최초 1회):  공지의 다운로드 링크에서 .whl 을 받은 뒤
    pip install dacon_submit_api-0.1.2-py3-none-any.whl --force-reinstall

사용법 (대회 폴더에서 실행):
    python src/submit.py submissions/baseline_end_eq_start.csv "end=start 베이스라인"
"""
import os
import sys

# 대회 ID는 비밀이 아니라 공개 정보이므로 코드에 박아둡니다.
COMPETITION_ID = "236647"  # K리그 경기 내 최종 패스 좌표 예측 AI 모델 개발


def load_secret():
    """환경변수 → .env 순서로 토큰/팀이름을 찾는다."""
    token = os.environ.get("DACON_TOKEN")
    team = os.environ.get("DACON_TEAM")
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if (not token or not team) and os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key == "DACON_TOKEN" and not token:
                token = val
            if key == "DACON_TEAM" and not team:
                team = val
    return token, team


def main():
    if len(sys.argv) < 2:
        print('사용법: python src/submit.py <submission.csv> ["메모(선택)"]')
        raise SystemExit(1)
    path = sys.argv[1]
    memo = sys.argv[2] if len(sys.argv) > 2 else ""

    if not os.path.exists(path):
        print(f"[오류] 제출 파일을 찾을 수 없습니다: {path}")
        raise SystemExit(1)

    token, team = load_secret()
    if not token or not team:
        print("[오류] 토큰/팀이름이 설정되지 않았습니다. 둘 중 하나로 설정하세요:")
        print("  · 환경변수 DACON_TOKEN, DACON_TEAM, 또는")
        print("  · 이 대회 폴더에 .env 파일 생성 후:")
        print("        DACON_TOKEN=발급받은_토큰")
        print("        DACON_TEAM=내_팀이름")
        print("  (토큰: 데이콘 마이페이지 > 계정관리.  .env 는 git에 안 올라갑니다)")
        raise SystemExit(1)

    try:
        from dacon_submit_api import dacon_submit_api
    except ImportError:
        print("[오류] dacon_submit_api 패키지가 설치되어 있지 않습니다.")
        print("  공지의 다운로드 링크에서 .whl 을 받은 뒤 설치하세요:")
        print("  pip install dacon_submit_api-0.1.2-py3-none-any.whl --force-reinstall")
        raise SystemExit(1)

    print(f"[제출] file={path}  comp={COMPETITION_ID}  team={team}  memo={memo!r}")
    result = dacon_submit_api.post_submission_file(
        path, token, COMPETITION_ID, team, memo
    )
    print("[응답]", result)
    if isinstance(result, dict) and result.get("isSubmitted"):
        print("✅ 제출 성공 — 점수는 데이콘 웹 리더보드에서 확인하세요.")
    else:
        print("❌ 제출 실패 — 위 detail 메시지를 확인하세요.")
        print("   (종료된 대회는 API 제출이 막혀 있을 수 있습니다.)")


if __name__ == "__main__":
    main()
