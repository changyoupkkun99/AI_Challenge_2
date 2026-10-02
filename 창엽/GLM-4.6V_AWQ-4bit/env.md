# 실행 환경 — GLM-4.6V AWQ 4bit

- GPU/방식: A100 80GB 또는 G4 96GB, AWQ INT4, vLLM zero-shot
- Hugging Face 모델 ID: `cyankiwi/GLM-4.6V-AWQ-4bit`
- 주 실행 노트북: `GLM-4.6V-AWQ-4bit_zero_shot_VQA_A100_G4(1).ipynb`
- 핵심 의존성: `requirements.txt` 참조
- 데이터: `공통/cleaning_data`의 CSV와 원본 Drive의 이미지, 또는 원본 `data.zip`/정제 아카이브를 사용
- 권장 실행: Colab 새 세션에서 노트북의 환경 셀 → 필요 시 런타임 재시작 → 데이터 확인 → 추론/학습 → 제출 CSV 검증

## 원본과 구분할 점

정리된 zero-shot 노트북과 제출 CSV가 연결됩니다. 별도 clean7169 LoRA 실험본과 0.78194 점수를 혼동하지 않도록 분리합니다.

정확한 Python/PyTorch/CUDA 빌드는 노트북에 기록된 성공 run을 우선합니다. `requirements.txt`는 확인 가능한 패키지만 담았으며 모든 CUDA wheel 설치 절차를 대체하지 않습니다.
