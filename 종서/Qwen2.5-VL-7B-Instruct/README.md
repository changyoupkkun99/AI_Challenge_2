# Qwen2.5-VL-7B-Instruct

- 담당: 종서
- public 점수: **0.89603** (사용자 제공; private 점수와 구분)
- 실행 모델 ID: `Qwen/Qwen2.5-VL-7B-Instruct`
- 방식/환경: 데스크톱 RTX 5060 Ti 실험 노트북 후보; CUDA 12.8 Torch·LoRA 셀 포함
- 실행 노트북: `(260325)_baseline_desktop5060ti.ipynb`
- 제출 CSV: `미확인`
- 원본 자료: 정확한 원본 위치 확인 필요

## 실행과 검증

1. `env.md`와 `requirements.txt`를 읽고 원본 노트북의 환경 준비 셀을 실행합니다.
2. 데이터 경로는 루트 `공통/데이터_위치.md`를 참고합니다. 암호화 ZIP을 기본 입력으로 가정하지 않습니다.
3. 노트북의 모델 revision, 데이터 split, 체크포인트, 이미지 처리와 제출 CSV 이름을 기록합니다.
4. 제출 CSV는 `id,answer` 형식과 test 6,714개 ID 일치를 확인합니다.

## 확인 메모

이 노트북은 여러 모델과 앙상블 셀이 섞인 이전 실험본입니다. 0.89603을 만든 정확한 제출 CSV와 run은 현재 자료에서 미확인입니다.

실제 soft probability가 있는 경우에만 출처, ID 순서, class 순서와 함께 기록합니다. hard submission을 확률로 바꾸어 표기하지 않습니다.
