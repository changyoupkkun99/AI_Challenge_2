# Qwen3-VL-8B-Instruct

- 담당: 종서
- public 점수: **0.95174** (사용자 제공; private 점수와 구분)
- 실행 모델 ID: `Qwen/Qwen3-VL-8B-Instruct`
- 방식/환경: A100 LoRA BF16 실행; full fine-tuning은 노트북에서 80GB급 검증 요구
- 실행 노트북: `qwen3vl_8b_TRAINING.ipynb`
- 제출 CSV: `Qwen3-VL-8B-Instruct_submission.csv`
- 원본 자료: https://drive.google.com/drive/folders/1aTVRfhymmmfdI-4mW2XEQGtCMEwAF-vV
- 대표 실행: [20260925T102849954716Z-full-low_lr](https://drive.google.com/drive/folders/1770H_WhTwZH6Kb7CuBJQEXlskcrVRenA) (어댑터 ZIP) · 내부 validation 1364/1434 (95.1185%)

## 실행과 검증

1. `env.md`와 `requirements.txt`를 읽고 원본 노트북의 환경 준비 셀을 실행합니다.
2. 데이터 경로는 루트 `공통/데이터_위치.md`를 참고합니다. 암호화 ZIP을 기본 입력으로 가정하지 않습니다.
3. 노트북의 모델 revision, 데이터 split, 체크포인트, 이미지 처리와 제출 CSV 이름을 기록합니다.
4. 제출 CSV는 `id,answer` 형식과 test 6,714개 ID 일치를 확인합니다.

## 확인 메모

공유 수치 97.5957%는 다른 실행의 `train_original` 부분 1096/1123이며, 위 대표 실행의 전체 validation과 다릅니다. public 점수도 별도 평가입니다.

실제 soft probability가 있는 경우에만 출처, ID 순서, class 순서와 함께 기록합니다. hard submission을 확률로 바꾸어 표기하지 않습니다.
