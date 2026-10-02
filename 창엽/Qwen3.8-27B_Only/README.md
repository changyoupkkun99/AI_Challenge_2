# Qwen/Qwen3.8-27B_Only

- 담당: 창엽
- public 점수: **0.95621** (사용자 제공; private 점수와 구분)
- 실행 모델 ID: `btbtyler09/Qwen3.8-27B-GPTQ-4bit`
- 방식/환경: A100 80GB, GPTQ 4bit + LoRA; OCR/DINO 없음
- 실행 노트북: `Qwen38_27B_clean7169_AgnesStyle_A100.ipynb`
- 제출 CSV: `submission_qwen38_27b_clean7169(1).csv`
- 원본 자료: https://drive.google.com/drive/folders/1gz5dJmZhAkDflw_ziT4vW7XRioAMBJq9

## 실행과 검증

1. `env.md`와 `requirements.txt`를 읽고 원본 노트북의 환경 준비 셀을 실행합니다.
2. 데이터 경로는 루트 `공통/데이터_위치.md`를 참고합니다. 암호화 ZIP을 기본 입력으로 가정하지 않습니다.
3. 노트북의 모델 revision, 데이터 split, 체크포인트, 이미지 처리와 제출 CSV 이름을 기록합니다.
4. 제출 CSV는 `id,answer` 형식과 test 6,714개 ID 일치를 확인합니다.

## 확인 메모

clean7169 재학습 노트북이 같은 이름의 제출 파일을 생성합니다. 사용자 제공 점수 0.95621과 이 실행본의 직접 대응은 별도 점수 기록으로 확인 필요합니다.

실제 soft probability가 있는 경우에만 출처, ID 순서, class 순서와 함께 기록합니다. hard submission을 확률로 바꾸어 표기하지 않습니다.
