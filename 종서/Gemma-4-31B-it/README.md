# google/gemma-4-31B-A4B-it

- 담당: 종서
- public 점수: **0.96544** (사용자 제공; private 점수와 구분)
- 실행 모델 ID: `google/gemma-4-31B-it`
- 방식/환경: G4 95GB급, BF16, 비양자화 LoRA 학습
- 실행 노트북: `gemma4_31b_TRAINING.ipynb`
- 제출 CSV: `Gemma-4-31B-it_submission.csv`
- 원본 자료: https://drive.google.com/drive/folders/1yoyUCriQE4PE1KsvBtAFP-hRJDeud7S_
- 대표 실행: [20260925T124050809287Z-gemma4-31b-full-bf16](https://drive.google.com/drive/folders/1l4dgc8CT4Xi7zeUfgjtXCFR1GZT-wULo) · 내부 validation 1400/1434 (97.6290%)

## 실행과 검증

1. `env.md`와 `requirements.txt`를 읽고 원본 노트북의 환경 준비 셀을 실행합니다.
2. 데이터 경로는 루트 `공통/데이터_위치.md`를 참고합니다. 암호화 ZIP을 기본 입력으로 가정하지 않습니다.
3. 노트북의 모델 revision, 데이터 split, 체크포인트, 이미지 처리와 제출 CSV 이름을 기록합니다.
4. 제출 CSV는 `id,answer` 형식과 test 6,714개 ID 일치를 확인합니다.

## 확인 메모

사용자 표기의 31B-A4B와 실제 노트북 모델 ID gemma-4-31B-it가 다릅니다. 실행 ID를 노트북 기준으로 분리 표기합니다.
26B용 `GEMMA4_G4_TRAINING.ipynb`와 혼동하지 마세요. 이 폴더의 실행 노트북은 `gemma4_31b_TRAINING.ipynb`입니다.

실제 soft probability가 있는 경우에만 출처, ID 순서, class 순서와 함께 기록합니다. hard submission을 확률로 바꾸어 표기하지 않습니다.
