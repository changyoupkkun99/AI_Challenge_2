# GLM-4.6V AWQ 4bit

- 담당: 창엽
- public 점수: **0.78194** (사용자 제공; private 점수와 구분)
- 실행 모델 ID: `cyankiwi/GLM-4.6V-AWQ-4bit`
- 방식/환경: A100 80GB 또는 G4 96GB, AWQ INT4, vLLM zero-shot
- 실행 노트북: `GLM-4.6V-AWQ-4bit_zero_shot_VQA_A100_G4(1).ipynb`
- 제출 CSV: `glm46v_awq4_submission.csv`
- 원본 자료: https://drive.google.com/drive/folders/1cjF23OrvlWhnmYEryHNAyWvXBm3faQeT

## 실행과 검증

1. `env.md`와 `requirements.txt`를 읽고 원본 노트북의 환경 준비 셀을 실행합니다.
2. 데이터 경로는 루트 `공통/데이터_위치.md`를 참고합니다. 암호화 ZIP을 기본 입력으로 가정하지 않습니다.
3. 노트북의 모델 revision, 데이터 split, 체크포인트, 이미지 처리와 제출 CSV 이름을 기록합니다.
4. 제출 CSV는 `id,answer` 형식과 test 6,714개 ID 일치를 확인합니다.

## 확인 메모

정리된 zero-shot 노트북과 제출 CSV가 연결됩니다. 별도 clean7169 LoRA 실험본과 0.78194 점수를 혼동하지 않도록 분리합니다.

실제 soft probability가 있는 경우에만 출처, ID 순서, class 순서와 함께 기록합니다. hard submission을 확률로 바꾸어 표기하지 않습니다.
