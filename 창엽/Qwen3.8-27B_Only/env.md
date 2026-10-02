# 실행 환경 — Qwen/Qwen3.8-27B_Only

- GPU/방식: A100 80GB, GPTQ 4bit + LoRA; OCR/DINO 없음
- Hugging Face 모델 ID: `btbtyler09/Qwen3.8-27B-GPTQ-4bit`
- 주 실행 노트북: `Qwen38_27B_clean7169_AgnesStyle_A100.ipynb`
- 핵심 의존성: `requirements.txt` 참조
- 데이터: `공통/cleaning_data`의 CSV와 원본 Drive의 이미지, 또는 원본 `data.zip`/정제 아카이브를 사용
- 권장 실행: Colab 새 세션에서 노트북의 환경 셀 → 필요 시 런타임 재시작 → 데이터 확인 → 추론/학습 → 제출 CSV 검증

## 원본과 구분할 점

clean7169 재학습 노트북이 같은 이름의 제출 파일을 생성합니다. 사용자 제공 점수 0.95621과 이 실행본의 직접 대응은 별도 점수 기록으로 확인 필요합니다.

정확한 Python/PyTorch/CUDA 빌드는 노트북에 기록된 성공 run을 우선합니다. `requirements.txt`는 확인 가능한 패키지만 담았으며 모든 CUDA wheel 설치 절차를 대체하지 않습니다.
