# Qwen3-VL-4B-Instruct

- 담당: 종서
- public 점수: **0.94399** (사용자 제공; private 점수와 구분)
- 실행 모델 ID: `Qwen/Qwen3-VL-4B-Instruct`
- 대표 실행 환경: NVIDIA GeForce RTX 5060 Ti, BF16 LoRA, 비양자화
- 실행 노트북: `qwen3vl_4b_TRAINING.ipynb`
- 제출 CSV: `Qwen3-VL-4B-Instruct_submission.csv`
- 원본 자료: https://drive.google.com/drive/folders/19MeOQrRckJgR9Dj06jxkXg9cJLJ5dXOd
- 대표 실행: [20260921T001126826670Z-full-none](https://drive.google.com/drive/folders/1UPC6y8K0i-kWwGRRVv7EEl9rQXOqfHDU) · 내부 validation 1284/1343 (95.6069%)

## 실행과 검증

1. `env.md`와 `requirements.txt`를 읽고 원본 노트북의 환경 준비 셀을 실행합니다.
2. 데이터 경로는 루트 `공통/데이터_위치.md`를 참고합니다. 암호화 ZIP을 기본 입력으로 가정하지 않습니다.
3. 노트북의 모델 revision, 데이터 split, 체크포인트, 이미지 처리와 제출 CSV 이름을 기록합니다.
4. 제출 CSV는 `id,answer` 형식과 test 6,714개 ID 일치를 확인합니다.

## 확인 메모

대표 실행의 `full`은 전체 데이터 실행 프로필이며 전체 파라미터 미세조정이라는 뜻이 아닙니다. 이 실행의 방식은 LoRA입니다. 재현 시 노트북의 현재 기본값보다 대표 실행 폴더의 `config.json`을 우선 확인하세요. 위 validation은 public 점수와 다른 지표입니다.

실제 soft probability가 있는 경우에만 출처, ID 순서, class 순서와 함께 기록합니다. hard submission을 확률로 바꾸어 표기하지 않습니다.
