# 실행 환경 — Qwen2.5-VL-7B-Instruct

- GPU/방식: 데스크톱 RTX 5060 Ti 실험 노트북 후보; CUDA 12.8 Torch·LoRA 셀 포함
- Hugging Face 모델 ID: `Qwen/Qwen2.5-VL-7B-Instruct`
- 주 실행 노트북: `(260325)_baseline_desktop5060ti.ipynb`
- 핵심 의존성: `requirements.txt` 참조
- 데이터: `공통/cleaning_data`의 CSV와 원본 Drive의 이미지, 또는 원본 `data.zip`/정제 아카이브를 사용
- 권장 실행: Colab 새 세션에서 노트북의 환경 셀 → 필요 시 런타임 재시작 → 데이터 확인 → 추론/학습 → 제출 CSV 검증

## 원본과 구분할 점

이 노트북은 여러 모델과 앙상블 셀이 섞인 이전 실험본입니다. 0.89603을 만든 정확한 제출 CSV와 run은 현재 자료에서 미확인입니다.

정확한 Python/PyTorch/CUDA 빌드는 노트북에 기록된 성공 run을 우선합니다. `requirements.txt`는 확인 가능한 패키지만 담았으며 모든 CUDA wheel 설치 절차를 대체하지 않습니다.
