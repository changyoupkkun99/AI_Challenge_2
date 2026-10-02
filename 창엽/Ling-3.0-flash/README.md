# Ling-3.0-flash

- 담당: 창엽
- public 점수: **0.94638** (사용자 제공; private 점수와 구분)
- 실행 모델 ID: `inclusionAI/Ling-3.0-flash-VL-int4`
- 방식/환경: G4 약 95GB (노트북 90GiB 이상 검사), CUDA 13.0, Ling vLLM fork 소스 빌드, INT4 zero-shot
- 실행 노트북: `Ling-3.0-flash-VL-int4_zero_shot_VQA_G4(4).ipynb`
- 제출 CSV: `ling3_vl_int4_submission (1).csv`
- 원본 자료: https://drive.google.com/drive/folders/1Y-W6hhlf8Z5fVLuvoayT4jzSdmoRFkq0

## 실행과 검증

1. `env.md`와 `requirements.txt`를 읽고 원본 노트북의 환경 준비 셀을 실행합니다.
2. 데이터 경로는 루트 `공통/데이터_위치.md`를 참고합니다. 암호화 ZIP을 기본 입력으로 가정하지 않습니다.
3. 노트북의 모델 revision, 데이터 split, 체크포인트, 이미지 처리와 제출 CSV 이름을 기록합니다.
4. 제출 CSV는 `id,answer` 형식과 test 6,714개 ID 일치를 확인합니다.

## 확인 메모

일반 vLLM 설치로 재현되지 않습니다. 노트북 Step 0의 Ling fork/CUDA 빌드와 런타임 재시작을 따르세요.

실제 soft probability가 있는 경우에만 출처, ID 순서, class 순서와 함께 기록합니다. hard submission을 확률로 바꾸어 표기하지 않습니다.
