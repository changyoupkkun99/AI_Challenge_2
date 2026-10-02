# Qwen3.5-122B-A10B-GPTQ-Int4

- 담당: 창엽
- public 점수: **0.96097** (사용자 제공; private 점수와 구분)
- 실행 모델 ID: `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`
- 방식/환경: GPTQ Int4, vLLM 0.29.1rc1.dev567+g8d16ca6cc, CUDA 13.2 계열 Torch
- 실행 노트북: `qwen3.5-122b-a10b-gptq-int4_verification(2).ipynb`
- 제출 CSV: `122b_submission.csv`
- 원본 자료: https://drive.google.com/drive/folders/1CP_Me1Yqw1GuRvDhnuD4DTFda8Oqilbs

## 실행과 검증

1. `env.md`와 `requirements.txt`를 읽고 원본 노트북의 환경 준비 셀을 실행합니다.
2. 데이터 경로는 루트 `공통/데이터_위치.md`를 참고합니다. 암호화 ZIP을 기본 입력으로 가정하지 않습니다.
3. 노트북의 모델 revision, 데이터 split, 체크포인트, 이미지 처리와 제출 CSV 이름을 기록합니다.
4. 제출 CSV는 `id,answer` 형식과 test 6,714개 ID 일치를 확인합니다.

## 확인 메모

노트북의 STEP 0이 별도 wheel index와 PyTorch를 설치합니다. 일반 pip install만으로 같은 vLLM 빌드가 되지 않습니다.

실제 soft probability가 있는 경우에만 출처, ID 순서, class 순서와 함께 기록합니다. hard submission을 확률로 바꾸어 표기하지 않습니다.
