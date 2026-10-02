# 실행 환경 — Qwen3-VL-8B-Instruct

- GPU/방식: A100 LoRA BF16 실행; full fine-tuning은 노트북에서 80GB급 검증 요구
- Hugging Face 모델 ID: `Qwen/Qwen3-VL-8B-Instruct`
- 주 실행 노트북: `qwen3vl_8b_TRAINING.ipynb`
- 핵심 의존성: `requirements.txt` 참조
- 데이터: `공통/cleaning_data`의 CSV와 원본 Drive의 이미지, 또는 원본 `data.zip`/정제 아카이브를 사용
- 권장 실행: Colab 새 세션에서 노트북의 환경 셀 → 필요 시 런타임 재시작 → 데이터 확인 → 추론/학습 → 제출 CSV 검증

## 원본과 구분할 점

기존 안내문에는 서로 다른 평가 범위의 validation 수치가 함께 있었습니다. public 점수와 validation 점수를 분리합니다.

정확한 Python/PyTorch/CUDA 빌드는 노트북에 기록된 성공 run을 우선합니다. `requirements.txt`는 확인 가능한 패키지만 담았으며 모든 CUDA wheel 설치 절차를 대체하지 않습니다.
