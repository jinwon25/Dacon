# 데이터 계약

원본 데이터는 GitHub에 업로드하지 않는다. 각 팀원이 DACON 대회 페이지에서 직접 내려받아 이 디렉터리에 배치한다.

| 파일 | 크기(bytes) | SHA-256 |
|---|---:|---|
| `train.csv` | 368,527,723 | `D2081186B458B49F60B082BE480C273135833E15BA59A76D033AF28BCF8763FF` |
| `trackman_history.csv` | 353,823,031 | `F7818F9EE0CCEFE7C2CF69FA99EFE6E5CB882D8B886DD96D2394BCF3B53F33A9` |
| `test.csv` | 1,894 | `478D10B20C00443F6FE8270AB7348DE49D14395526130F0B2D915DA496821A19` |
| `sample_submission.csv` | 112 | `B2CF6BA6745C74C46DB23E620D74705F9B327EFB8C8A14DC0DD2AFAB0E898775` |

PowerShell 검증 예시:

```powershell
Get-FileHash -Algorithm SHA256 data\train.csv
Get-FileHash -Algorithm SHA256 data\trackman_history.csv
```

대회 제공 데이터의 재배포가 명시적으로 허용되지 않는 한 private 저장소에도 원본 파일을 올리지 않는다.
