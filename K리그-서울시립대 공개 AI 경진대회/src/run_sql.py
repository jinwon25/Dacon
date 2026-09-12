"""DuckDB로 .sql 파일을 실행해 결과를 표로 보기 좋게 출력하는 작은 도구.

왜 이렇게 쓰나요?
- DuckDB는 '파일 기반 SQL 엔진'이라, 서버를 띄우거나 DB에 데이터를 적재(import)하지 않아도
  CSV 파일에 바로 SQL을 던질 수 있습니다.  ->  SELECT ... FROM 'data/train.csv'
- 쿼리를 .sql 파일에 모아두면 (1) 버전 관리가 되고 (2) 나중에 그대로 재실행해 재현됩니다.

사용법 (대회 폴더 안에서 실행):
    python src/run_sql.py sql/01_overview.sql

규칙:
- 세미콜론(;)으로 SQL 문을 구분합니다.
- 각 문 첫 줄을  -- @제목  형식으로 쓰면 그 제목을 출력해 줍니다.
"""
import sys
import duckdb
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 80)


def main(sql_path: str) -> None:
    text = open(sql_path, encoding="utf-8").read()
    con = duckdb.connect()  # 인메모리 DB (파일 안 만듦)
    stmts = [s.strip() for s in text.split(";") if s.strip()]
    for s in stmts:
        first = s.splitlines()[0].strip()
        label = first[4:].strip() if first.startswith("-- @") else ""
        print("\n" + "=" * 72)
        if label:
            print(f"# {label}")
        print(s)
        print("-" * 72)
        try:
            print(con.execute(s).df().to_string(index=False))
        except Exception as e:  # noqa: BLE001
            print("ERROR:", e)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python src/run_sql.py <쿼리.sql 경로>")
        raise SystemExit(1)
    main(sys.argv[1])
