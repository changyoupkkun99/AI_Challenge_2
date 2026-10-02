# 실행 환경 — google/gemma-4-26B-A4B-it

- GPU/방식: G4 95GB급, BF16, 비양자화 LoRA 학습
- Hugging Face 모델 ID: `google/gemma-4-26B-A4B-it`
- 주 실행 노트북: `gemma4_26b_a4b_TRAINING.ipynb`
- 핵심 의존성: `requirements.txt` 참조
- 데이터: `공통/cleaning_data`의 CSV와 원본 Drive의 이미지, 또는 원본 `data.zip`/정제 아카이브를 사용
- 권장 실행: Colab 새 세션에서 노트북의 환경 셀 → 필요 시 런타임 재시작 → 데이터 확인 → 추론/학습 → 제출 CSV 검증

## 원본과 구분할 점

사용자 제공 public 점수와 기존 run validation 수치는 별도 지표로 기록합니다.

정확한 Python/PyTorch/CUDA 빌드는 노트북에 기록된 성공 run을 우선합니다. `requirements.txt`는 확인 가능한 패키지만 담았으며 모든 CUDA wheel 설치 절차를 대체하지 않습니다.
