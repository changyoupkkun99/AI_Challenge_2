# google/gemma-4-26B-A4B-it

- 담당: 종서
- public 점수: **0.96216** (사용자 제공; private 점수와 구분)
- 실행 모델 ID: `google/gemma-4-26B-A4B-it`
- 방식/환경: G4 95GB급, BF16, 비양자화 LoRA 학습
- 실행 노트북: `gemma4_26b_a4b_TRAINING.ipynb`
- 제출 CSV: `Gemma-4-26B-A4B-it_submission.csv`
- 원본 자료: https://drive.google.com/drive/folders/1p384h8MgYAOYyecLNtd8yu--SqbmbWyg
- 대표 실행: [20260922T002054858922Z-gemma4-full-bf16](https://drive.google.com/drive/folders/1krmW6e9pZwKqHwFROth1zxC_RJqFPX95) · 내부 validation 1301/1343 (96.8727%)

## 실행과 검증

1. `env.md`와 `requirements.txt`를 읽고 원본 노트북의 환경 준비 셀을 실행합니다.
2. 데이터 경로는 루트 `공통/데이터_위치.md`를 참고합니다. 암호화 ZIP을 기본 입력으로 가정하지 않습니다.
3. 노트북의 모델 revision, 데이터 split, 체크포인트, 이미지 처리와 제출 CSV 이름을 기록합니다.
4. 제출 CSV는 `id,answer` 형식과 test 6,714개 ID 일치를 확인합니다.

## 확인 메모

사용자 제공 public 점수와 기존 run validation 수치는 별도 지표로 기록합니다.

실제 soft probability가 있는 경우에만 출처, ID 순서, class 순서와 함께 기록합니다. hard submission을 확률로 바꾸어 표기하지 않습니다.
