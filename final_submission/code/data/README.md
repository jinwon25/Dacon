# 데이터 계약

대회가 제공한 다음 네 파일을 이 디렉터리에 직접 배치한다. 원본 데이터는 제출
패키지에 포함하지 않는다.

| 파일 | 크기(bytes) | SHA-256 |
|---|---:|---|
| `train.csv` | 368,527,723 | `D2081186B458B49F60B082BE480C273135833E15BA59A76D033AF28BCF8763FF` |
| `trackman_history.csv` | 353,823,031 | `F7818F9EE0CCEFE7C2CF69FA99EFE6E5CB882D8B886DD96D2394BCF3B53F33A9` |
| `test.csv` | 1,894 | `478D10B20C00443F6FE8270AB7348DE49D14395526130F0B2D915DA496821A19` |
| `sample_submission.csv` | 112 | `B2CF6BA6745C74C46DB23E620D74705F9B327EFB8C8A14DC0DD2AFAB0E898775` |

해시는 `python train.py preflight --data-dir data`로 한 번에 확인할 수 있다.
대회 데이터의 재배포 여부는 주최 측 이용 조건을 따른다.
