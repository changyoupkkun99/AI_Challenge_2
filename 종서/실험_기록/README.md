# 종서 실험 기록

- `team_results.csv`: 여러 실행의 GPU, 평가 범위, 정확도, 지연과 메모리 사용량을 기록한 원본 표.
- `best_model.json`: 원본 폴더에서 선정한 best run 기록.
- `원본_00_모델별_안내.md`: 원본 안내 보존본. 97.5957%는 Qwen 8B의 부분 평가 수치이며 전체 validation과 다릅니다.
- 모델별 `대표_실행/`: LoRA 어댑터(또는 Qwen 8B의 실행 ZIP), 설정, 데이터 분할표, 학습 요약, 검증 기록.

| 모델 | 대표 실행 원본 | 내부 validation | 대표 실행 장치 |
|---|---|---:|---|
| Qwen3-VL-4B | [full-none](https://drive.google.com/drive/folders/1UPC6y8K0i-kWwGRRVv7EEl9rQXOqfHDU) | 1284/1343 (95.6069%) | RTX 5060 Ti, BF16 LoRA |
| Qwen3-VL-8B | [full-low_lr](https://drive.google.com/drive/folders/1K0eK3Of9tzC_NshcsN1kXGz3otFm7o7d) | 1364/1434 (95.1185%) | A100 40GB, BF16 LoRA |
| Gemma-4-26B-A4B | [gemma4-full-bf16](https://drive.google.com/drive/folders/1FiPgCL01js75WNSVpWrofq_5N2G8oZ_w) | 1301/1343 (96.8727%) | RTX PRO 6000 Blackwell 95GB, BF16 LoRA |
| Gemma-4-31B | [gemma4-31b-full-bf16](https://drive.google.com/drive/folders/1YNGjDCTckieKhvG-w1QwYeFusPuqIG_E) | 1400/1434 (97.6290%) | RTX PRO 6000 Blackwell 95GB, BF16 LoRA |

원본 `training_runs`의 [다른 실행과 HTML 비교 보고서](https://drive.google.com/drive/folders/1JVt32Cg7S8iayKGo4MP_Y2gUjeoAqN4o)도 보존되어 있습니다. 내부 validation, 부분 평가, 사용자가 제공한 public leaderboard 점수는 평가 문항이 다르므로 혼용하지 않습니다.
