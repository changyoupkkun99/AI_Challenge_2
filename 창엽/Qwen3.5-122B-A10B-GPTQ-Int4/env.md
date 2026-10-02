# 실행 환경 — Qwen3.5-122B-A10B-GPTQ-Int4

- GPU/방식: GPTQ Int4, vLLM 0.29.1rc1.dev567+g8d16ca6cc, CUDA 13.2 계열 Torch
- Hugging Face 모델 ID: `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`
- 주 실행 노트북: `qwen3.5-122b-a10b-gptq-int4_verification(2).ipynb`
- 핵심 의존성: `requirements.txt` 참조
- 데이터: `공통/cleaning_data`의 CSV와 원본 Drive의 이미지, 또는 원본 `data.zip`/정제 아카이브를 사용
- 권장 실행: Colab 새 세션에서 노트북의 환경 셀 → 필요 시 런타임 재시작 → 데이터 확인 → 추론/학습 → 제출 CSV 검증

## 원본과 구분할 점

노트북의 STEP 0이 별도 wheel index와 PyTorch를 설치합니다. 일반 pip install만으로 같은 vLLM 빌드가 되지 않습니다.

정확한 Python/PyTorch/CUDA 빌드는 노트북에 기록된 성공 run을 우선합니다. `requirements.txt`는 확인 가능한 패키지만 담았으며 모든 CUDA wheel 설치 절차를 대체하지 않습니다.
