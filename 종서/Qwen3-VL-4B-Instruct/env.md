# 실행 환경 — Qwen3-VL-4B-Instruct

- 대표 실행 GPU/방식: NVIDIA GeForce RTX 5060 Ti, BF16 LoRA, 비양자화
- 대표 실행 설정: 1 epoch, learning rate 0.0002, LoRA r=16/alpha=32, gradient accumulation=8, seed=42
- Hugging Face 모델 ID: `Qwen/Qwen3-VL-4B-Instruct`
- 주 실행 노트북: `qwen3vl_4b_TRAINING.ipynb`
- 핵심 의존성: `requirements.txt` 참조
- 데이터: `공통/cleaning_data`의 CSV와 원본 Drive의 이미지, 또는 원본 `data.zip`/정제 아카이브를 사용
- 권장 실행: Colab 새 세션에서 노트북의 환경 셀 → 필요 시 런타임 재시작 → 데이터 확인 → 추론/학습 → 제출 CSV 검증

## 원본과 구분할 점

대표 실행의 `full`은 전체 데이터 프로필입니다. 파라미터 전체 미세조정으로 읽지 마세요. 정확한 설정은 [대표 실행의 config.json](https://drive.google.com/file/d/1h3r8M7f9ZLXho1eSlaSr8wN02KdatT7s/view)을 따릅니다.

정확한 Python/PyTorch/CUDA 빌드는 노트북에 기록된 성공 run을 우선합니다. `requirements.txt`는 확인 가능한 패키지만 담았으며 모든 CUDA wheel 설치 절차를 대체하지 않습니다.
