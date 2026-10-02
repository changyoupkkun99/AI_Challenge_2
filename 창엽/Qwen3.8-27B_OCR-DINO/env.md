# 실행 환경 — Qwen/Qwen3.8-27B+OCR/DINO

- GPU/방식: A100 80GB 실행 기록; GPTQ 4bit + LoRA, EasyOCR, Grounding DINO
- Hugging Face 모델 ID: `btbtyler09/Qwen3.8-27B-GPTQ-4bit`
- 주 실행 노트북: `(260908)_baseline_colab_hybrid_qwen38_v2 (1)(1).ipynb`
- 핵심 의존성: `requirements.txt` 참조
- 데이터: `공통/cleaning_data`의 CSV와 원본 Drive의 이미지, 또는 원본 `data.zip`/정제 아카이브를 사용
- 권장 실행: Colab 새 세션에서 노트북의 환경 셀 → 필요 시 런타임 재시작 → 데이터 확인 → 추론/학습 → 제출 CSV 검증

## 원본과 구분할 점

기존 hybrid 노트북은 여러 실험 섹션이 함께 있습니다. 0.95918 제출에 대응하는 체크포인트와 선택한 셀/shift는 원본 실행 기록을 대조해야 합니다.

정확한 Python/PyTorch/CUDA 빌드는 노트북에 기록된 성공 run을 우선합니다. `requirements.txt`는 확인 가능한 패키지만 담았으며 모든 CUDA wheel 설치 절차를 대체하지 않습니다.
