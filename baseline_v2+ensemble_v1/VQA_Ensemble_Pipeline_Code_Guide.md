# VQA 앙상블 파이프라인: 슬라이드별 해설과 코드

모델편 다음에 읽는 앙상블편이다. **완료된 모델 run의 hard CSV와 실제 soft NPY를 입력받는 독립 노트북**을 설명한다. VLM 학습·이미지 추론 코드를 수정하거나 합치지 않았다.

대상 PPT: `VQA_Ensemble_Pipeline_Explained.pptx` 계열, 36장. 본문에 짧은 코드와 예시를 넣었고, 자세한 설명은 각 슬라이드 발표자 노트와 이 문서에 담았다.

## 근거와 해석 범위

- 직접 확인한 파일: `VQA_Ensemble_Baseline_v1.ipynb`, `VQA_Ensemble_Baseline.py`, `VQA_Ensemble_Config.json`, `VQA_Ensemble_README.md`.
- Python 코드 버전: `2026-10-05 v1`. 노트북은 42개 셀, 20개 코드 셀, 20단계다.
- 설명용 숫자: PPT의 배열·확률·guard 예시, 1,000개 meta 분할 예시, 100문항의 90/92개 정답 비교. 실제 대회 성능이 아니다.
- 실제 과거 성적: 사용자가 제공한 A·A 이후 제출 기록. B의 정확한 점수는 미확인이다.
- 새 노트북은 CPU 합성 fixture 검증 기록을 갖고 있다. 실제 모델 예측을 넣은 대회 성능과 Public/Private 점수는 미확인이다.
- 과거 RF_v1의 학습 입력·설정·수동 변경 감사 파일을 직접 대조하지 않았으므로 새 `rf_meta/rf_guard`를 과거 결과의 정확한 재현으로 설명하지 않는다.

원본 Python SHA256: `6aff541033ead06b2c3804051c9d00a187210ac017a00ea6ef5304752bf389b5`

## 처음 실행할 때

1. 같은 validation의 예측을 가진 모델 run들을 준비한다. 이미지 그룹, 정답, 질문·보기, 확률 의미도 확인한다.
2. 20번 셀의 `CFG["models"]`, 정답 CSV, 실제 sample submission, 출력 경로를 등록한다.
3. `inspect`로 입력 검사를 실행한다. 여기서는 메타 학습이나 제출 CSV를 만들지 않는다.
4. `evaluate`로 meta-fit·OOF·규칙 선택·holdout 결과를 확인한다.
5. 동일 설정·입력·분할을 유지해 `export`한다. export도 fitting을 다시 실행하고 test를 결합한다.

각 모드는 새 run을 만든다. `meta_split.manifest_path`로 분할을 고정할 수 있다. evaluate가 저장한 estimator 객체를 export에서 그대로 읽는 구조는 아니다. v1은 estimator 바이너리를 저장하지 않으므로 원본 예측·설정·환경을 유지해야 한다.

## 노트북 전체 20단계

| 단계 | 이름 | 실제 노트북 셀 번호 | 공부할 PPT |
|---|---|---|---|
| 01 | 환경 준비 | 설명 2 / 코드 3 | 4 |
| 02 | 라이브러리·공통 함수 | 설명 4 / 코드 5 | 5 |
| 03 | 전체 설정·초기값 | 설명 6 / 코드 7 | 5, 6, 22 |
| 04 | ID·정답·클래스 검증 | 설명 8 / 코드 9 | 7, 8 |
| 05 | 모델별 soft/hard 로드 | 설명 10 / 코드 11 | 6, 9, 10 |
| 06 | 공통 validation·test 정렬 | 설명 12 / 코드 13 | 7, 11 |
| 07 | 이미지 그룹 meta 분할 | 설명 14 / 코드 15 | 12 |
| 08 | 평가 지표·Rescue/Harm | 설명 16 / 코드 17 | 13, 26 |
| 09 | 온도·soft 가중치 학습 | 설명 18 / 코드 19 | 14, 15 |
| 10 | 투표·만장일치·기준 모델 보호 | 설명 20 / 코드 21 | 16~19 |
| 11 | Logistic/RF 선택형 메타 모델 | 설명 22 / 코드 23 | 20~22 |
| 12 | 후보 실행·예측 유형 | 설명 24 / 코드 25 | 23 |
| 13 | Meta-fit 내부 OOF | 설명 26 / 코드 27 | 24 |
| 14 | 후보 선택·holdout 동결 | 설명 28 / 코드 29 | 25 |
| 15 | 결과 비교·오답 검토 | 설명 30 / 코드 31 | 26, 32 |
| 16 | 실제 차트·실행 감사 | 설명 32 / 코드 33 | 27, 30 |
| 17 | 제출 CSV·NPY 저장 | 설명 34 / 코드 35 | 28~30 |
| 18 | Test-only·수동 검토 선택 분기 | 설명 36 / 코드 37 | 31, 32 |
| 19 | 실행기·CLI·CPU 자체 검사 | 설명 38 / 코드 39 | 33 |
| 20 | 실행 설정·실행 | 설명 40 / 코드 41 | 6, 22, 33 |

## 입력 파일과 확률 의미

| 입력 | 실제 의미 | 사용 방법 |
|---|---|---|
| `valid_probs.npy`, `test_probs.npy` | 원래 보기 순서로 복원한 최종 확률. TTA 평균이 포함될 수 있음 | soft 평균·guard·메타 feature |
| `valid_logits.npy`, `test_logits.npy` | 모델편 reference shift의 실제 choice logits | 이 앙상블 loader는 읽지 않음 |
| hard CSV | 모델이 고른 답 문자 | hard vote 또는 명시적 범주형 feature |
| `*_ids.json` | 각 배열 행에 해당하는 ID | canonical 순서로 재배열 |
| `valid_labels.npy` | 입력 class order 기준의 정수 label | canonical a/b/c/d로 변환하여 정답과 대조 |
| `probability_metadata.json` | class order·보기 복원·확률 출처·smoke·in-sample 정보 | 의미 확인 및 strict 검사 |

softmax(reference logits)와 TTA 최종 확률은 일반적으로 다르다. 최종 확률을 reference logits로 덮어쓰면 upstream TTA를 잃는다. NPY의 shape·행합·값만으로 외부 확률의 출처를 완전히 증명할 수는 없으므로 metadata와 실제 생성 코드를 함께 확인한다.

## 실행 옵션 예시

다음은 완료된 모델 run 경로를 등록하는 예시다. 파일과 VLM validation 제외 여부를 확인한 뒤 선언 값을 지정한다.

```python
CFG = default_config()
CFG["models"] = [
    {"name": "gemma31", "kind": "soft_run", "run_dir": "/path/runs/gemma31",
     "heldout_predictions_confirmed": True},
    {"name": "qwen27", "kind": "soft_run", "run_dir": "/path/runs/qwen27",
     "heldout_predictions_confirmed": True},
    {"name": "agnes", "kind": "soft_run", "run_dir": "/path/runs/agnes",
     "heldout_predictions_confirmed": True},
]
CFG["validation_csv"] = "/path/runs/gemma31/validation_rows.csv"
CFG["sample_submission_csv"] = "/path/data/sample_submission.csv"
CFG["output_root"] = "/path/ensemble_runs"
CFG["anchor"]["name"] = "gemma31"  # 실제로 기준 모델을 고정할 때
RESULT = run_ensemble(CFG, mode="inspect")
```

RF와 내부 OOF 선택을 켜는 예시:

```python
CFG["meta_models"]["rf"]["enabled"] = True
CFG["selection"]["policy"] = "fit_oof"
CFG["selection"]["candidates"] = ["anchor", "rf_meta", "rf_guard"]
# rf_guard는 soft anchor와 충분한 지지 모델이 있어야 생성된다.
# 활성화되지 않았거나 일부 fold에 없는 후보를 등록하면 오류가 난다.
RESULT = run_ensemble(CFG, mode="evaluate")
```

온도 보정·학습 soft 가중치·Logistic은 별도 옵션이다. RF를 활성화하면 기본 fixed 정책에서도 내부 OOF를 수행하지만, 자동 선택은 `policy="fit_oof"`일 때만 작동한다.

```python
CFG["calibration"]["mode"] = "final_probs_temperature"
CFG["learned_weights"]["enabled"] = True
CFG["guard"]["proposal"] = "soft_learned"
CFG["meta_models"]["logistic"]["enabled"] = True
```

## 슬라이드별 상세 해설

### 01. VQA 앙상블 파이프라인



모델편 다음에 읽는 앙상블편이다. 모델 학습·추론은 앞의 독립 노트북에서 끝내고, 이 노트북은 저장된 예측만 입력받는다. 실제 코드는 20단계로 구성되어 있다. 예시 확률·오류 수는 설명용이며 실제 대회 실행 점수와 구분한다. 과거 A의 성적은 사용자 제공 기록으로 마지막에 별도로 다룬다.

근거: VQA_Ensemble_Baseline_v1.ipynb의 개요 및 01~20단계, VQA_Ensemble_Config.json.

### 02. 모델 추론과 앙상블의 연결

단일 모델의 최종 예측을 파일로 전달한다

모델편의 TTA는 한 checkpoint에 보기 순서·이미지 변형을 바꿔 여러 번 추론하는 과정이다. 앙상블은 여러 모델 또는 여러 run이 이미 저장한 결과를 결합한다. 입력 run의 valid_probs/test_probs는 upstream의 TTA 평균을 포함할 수 있다. 여기서 VLM을 다시 불러오거나 LoRA를 추가 학습하지 않는다. 단, RF와 Logistic은 예측을 입력받는 별도 메타 분류기로 CPU에서 학습할 수 있다. 같은 checkpoint 변형 여러 개를 독립 모델이라고 해석하면 표 수가 늘어난 효과와 다양성 증가를 혼동할 수 있다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 개요
VQA_Ensemble_Baseline.py: load_model, run_ensemble

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `load_model`: 257~352행 | 슬라이드의 근거 함수·자료형 |
| `run_ensemble`: 1113~1229행 | 슬라이드의 근거 함수·자료형 |

### 03. 앙상블 전체 흐름

입력 검증 뒤에 고정 규칙과 학습형 후보를 비교한다

soft와 hard 입력을 각각 읽고 ID와 class order를 먼저 일치시킨다. 공통 validation을 이미지 그룹 기준으로 meta-fit과 meta-holdout으로 분리한다. soft 평균·hard 투표·core/guard 같은 규칙 후보와 선택적인 RF/Logistic 후보를 생성한다. 학습 옵션이나 fit_oof 선택을 사용하면 meta-fit 내부 OOF를 수행한다. 고정 정책 또는 fit OOF로 선택을 저장한 뒤 holdout 점수를 보고한다. export에서는 같은 fitted state로 test를 결합하고 sample submission ID 순서로 CSV를 만든다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 전체 구조
VQA_Ensemble_Baseline.py: run_ensemble, recipe_candidates, select_recipe

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `run_ensemble`: 1113~1229행 | 슬라이드의 근거 함수·자료형 |
| `recipe_candidates`: 727~788행 | 슬라이드의 근거 함수·자료형 |
| `select_recipe`: 833~872행 | 슬라이드의 근거 함수·자료형 |

### 04. 환경과 라이브러리

저장된 예측을 결합하므로 CPU에서 실행할 수 있다

설치 셀은 INSTALL_PACKAGES=False가 기본값이다. 작성 당시 CPU 검증 버전을 NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0으로 고정해 두었다. 이 값은 사용자 환경을 자동 교체한다는 뜻이 아니다. 실제 설치된 버전은 environment.json에 남긴다. NumPy는 NPY와 배열, pandas는 CSV, SciPy는 온도와 soft 가중치 최적화, sklearn은 그룹 분할과 RF/Logistic을 담당한다. 별도의 VLM 설치·GPU 메모리는 이 단계에 필요하지 않다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 01 환경 준비
VQA_Ensemble_Baseline.py: environment

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `environment`: 81~84행 | 슬라이드의 근거 함수·자료형 |

### 05. 파일과 프로젝트 설정

노트북에 코드 내장  /  아래 파일명은 VQA_Ensemble_ 접두사 생략

VQA_Ensemble_Baseline_v1.ipynb는 standalone 노트북이다. VQA_Ensemble_Baseline.py는 동일 기능을 CLI로 실행한다. Config.json은 펼친 기본값이며 README는 입력 규약과 실행 방법을 설명한다. 모델 목록과 경로는 빈 상태로 시작한다. 기본 seed는 42, 기본 anchor는 meta-fit 정확도로 선택, calibration과 learned weights 및 meta models는 off, selection은 fixed/anchor_soft_guard다. JSON·CSV·NPY 저장은 임시 파일을 만든 뒤 교체하는 방식이고 입력 파일 SHA256을 감사에 남긴다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 02 공통 함수 · 03 설정
VQA_Ensemble_Baseline.py: default_config, validate_config, save_json, hash_file

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `default_config`: 88~127행 | 슬라이드의 근거 함수·자료형 |
| `validate_config`: 129~160행 | 슬라이드의 근거 함수·자료형 |
| `save_json`: 56~61행 | 슬라이드의 근거 함수·자료형 |
| `hash_file`: 49~54행 | 슬라이드의 근거 함수·자료형 |

### 06. 모델 run 등록

이름·입력 종류·폴더·validation 제외 여부를 지정한다

20번 셀에 완료된 run 경로를 등록한다. soft_run은 앞 모델 노트북의 probs 폴더와 validation_rows.csv 등을 자동으로 찾아 읽는다. heldout_predictions_confirmed=True는 실제 VLM 학습에서 해당 validation을 제외했음을 확인한 뒤 지정한다. 존재하지 않는 run을 예시 이름만으로 등록하면 실행되지 않는다. validation_csv와 sample_submission_csv를 연결하고, 경로는 Colab/로컬 환경에 맞게 수정한다. 아래 코드는 경로만 바꾸는 짧은 등록 예이며 세 가지 모델 이름이 실제 준비된 것을 전제하지 않는다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 05 모델 로드 · 20 실행 설정
VQA_Ensemble_Baseline.py: load_model, default_config

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `load_model`: 257~352행 | 슬라이드의 근거 함수·자료형 |
| `default_config`: 88~127행 | 슬라이드의 근거 함수·자료형 |

### 07. ID 정렬

행 번호 대신 ID로 맞추고 문자열의 앞자리 0도 보존한다

모델마다 저장 행 순서가 달라도 ID 집합이 완전히 같으면 canonical 순서로 재배열한다. valid는 정답 CSV/validation_rows 순서, test는 sample submission의 원래 순서가 기준이다. 숫자형 정수 ID는 문자열로 바꾸고 문자열 ID의 0 접두사는 유지한다. ID 중복·누락·추가가 있으면 실패한다. 교집합만 뽑아 평가 집합을 줄이지 않는다. 실제 예시 ID 001과 010을 서로 바꾸면 정답을 다른 문항에 붙이게 되므로 정확한 문자열 join이 필요하다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 04 ID 검증 · 06 공통 정렬
VQA_Ensemble_Baseline.py: normalize_id, check_ids, align_array, canonical_test

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `normalize_id`: 164~169행 | 슬라이드의 근거 함수·자료형 |
| `check_ids`: 171~175행 | 슬라이드의 근거 함수·자료형 |
| `align_array`: 206~214행 | 슬라이드의 근거 함수·자료형 |
| `canonical_test`: 446~462행 | 슬라이드의 근거 함수·자료형 |

### 08. 클래스와 보기 순서 정렬

확률 열과 label을 원래 a/b/c/d 순서로 맞춘다

확률 배열의 열 순서가 d,c,b,a라면 canonical a,b,c,d를 위해 열 인덱스 3,2,1,0을 적용한다. valid_labels의 정수 인덱스도 입력 class_order에 따라 문자로 해석한 뒤 canonical 정수로 변환한다. upstream에서 보기 shift를 썼다면 원래 보기 순서로 역매핑했다는 metadata가 필요하다. 열 이름 정렬과 shift 역매핑은 다른 작업이다. 이미 복원된 결과에 shift를 다시 적용하면 오답을 만든다. 아래 값은 설명용 한 행이다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 04 클래스 검증
VQA_Ensemble_Baseline.py: class_permutation, answers_to_indices, load_model

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `class_permutation`: 201~204행 | 슬라이드의 근거 함수·자료형 |
| `answers_to_indices`: 188~199행 | 슬라이드의 근거 함수·자료형 |
| `load_model`: 257~352행 | 슬라이드의 근거 함수·자료형 |

### 09. soft·hard·test-only 입력

파일이 실제로 제공하는 정보를 기준으로 등록한다

soft_run은 모델편 run 폴더를, soft_files는 외부 NPY와 ID를, hard_csv는 답 CSV를 읽는다. 외부 soft는 class_order, original_option_order, confirmed_real_soft 등 출처 확인이 필요하다. 실제 soft 입력은 N×4이고 유한한 0~1 값, 행 합 1을 확인한다. 전체가 one-hot인 입력은 기본 거부한다. hard indicator는 RF의 범주형 feature로만 사용할 수 있다. test_only는 valid가 없다는 role 선언이며 학습과 후보 선택에서 제외한다. 파일 이름만 probs라고 붙였다고 실제 모델 확률임을 증명할 수는 없다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 05 모델별 soft/hard 로드
VQA_Ensemble_Baseline.py: load_model, input_catalog, check_probs

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `load_model`: 257~352행 | 슬라이드의 근거 함수·자료형 |
| `input_catalog`: 464~469행 | 슬라이드의 근거 함수·자료형 |
| `check_probs`: 216~228행 | 슬라이드의 근거 함수·자료형 |

### 10. TTA 이후의 최종 확률

모델편이 저장한 final_probs를 그대로 사용한다

모델편의 logits.npy는 원래 보기 순서로 복원한 reference shift의 실제 choice logits이고, probs.npy는 여러 shift의 확률을 각각 복원해 평균한 최종 결과다. 따라서 reference logits의 softmax와 final_probs는 일반적으로 다르다. 앙상블 코드는 final_probs를 읽고 raw_logits_consumed=False를 남긴다. 아래 shift0과 shift2 확률 예시를 평균하면 a=.50,b=.40,c=.05,d=.05다. shift0만 다시 softmax해 쓴다면 a=.70만 반영해 TTA 효과를 잃는다. 모델 run마다 metadata의 확률 의미를 확인한다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 05 모델별 soft/hard 로드
VQA_Ensemble_Baseline.py: load_model

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `load_model`: 257~352행 | 슬라이드의 근거 함수·자료형 |

### 11. 공통 validation 확인

ID뿐 아니라 정답·질문·보기·이미지 내용도 대조한다

공통 validation에는 원본 train_gold 정답을 사용한다. _src가 있으면 train_gold만 허용하고, 없으면 validation_is_gold_confirmed를 확인한다. validation_rows가 있으면 같은 ID의 정답과 질문/a/b/c/d/_hash를 대조한다. 파일이 없으면 same_task_confirmed와 heldout 선언에 의존하며 한계를 경고에 남긴다. training_rows가 있으면 원 모델 train과 valid의 ID, _group, _hash 중복을 검사한다. validation_signature가 달라도 ID 정렬 후 충분한 내용과 이미지 해시가 모두 같으면 서명 차이를 기록하며 허용한다. 672·1343·1434는 개수만 다를 수 있는 별도 집합이므로 교집합 자동 평가를 하지 않는다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 06 공통 validation·test 정렬
VQA_Ensemble_Baseline.py: canonical_validation, canonical_test

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `canonical_validation`: 356~444행 | 슬라이드의 근거 함수·자료형 |
| `canonical_test`: 446~462행 | 슬라이드의 근거 함수·자료형 |

### 12. 이미지 그룹 meta 분할

공통 validation 안에서 앙상블 학습용과 평가용을 나눈다

기본 meta_split holdout_fraction=.2는 5개 StratifiedGroupKFold 중 크기와 클래스 분포가 전체에 가까운 fold 하나를 holdout으로 고르는 방식이다. 같은 이미지 그룹은 fit과 holdout에 함께 들어갈 수 없다. 그룹 단위라 실제 행 수가 정확히 4:1일 필요는 없고 realized_holdout_fraction을 저장한다. label 비율은 원래 분포를 고려하며 평가 데이터를 무조건 각 25%로 재배열하지 않는다. 이 단계는 원 모델의 train/validation 분할과 별개다. 1000/800/200은 이해를 돕는 예시이고 실제는 meta_split.json의 ID가 기준이다. base VLM 선택에 이미 valid를 썼다면 이 holdout도 프로젝트 전체 수준에서 새로운 독립 데이터가 아니다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 07 이미지 그룹 meta 분할
VQA_Ensemble_Baseline.py: grouped_folds, make_meta_split

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `grouped_folds`: 473~486행 | 슬라이드의 근거 함수·자료형 |
| `make_meta_split`: 488~520행 | 슬라이드의 근거 함수·자료형 |

### 13. 평가 지표

정답률과 확률의 품질을 구분해서 계산한다

accuracy는 최종 선택 답이 정답인 비율이며 클래스별 정확도도 계산한다. NLL은 정답 클래스에 둔 확률의 -log 평균이다. Brier는 네 클래스의 확률과 one-hot target 차이의 제곱합 평균이며, 이 구현은 4로 나누지 않는다. ECE는 10개 confidence 구간에서 평균 confidence와 실제 정확도 차이를 가중 평균한다. hard-only 후보에는 확률 지표를 억지로 붙이지 않는다. 확률 지표의 계산 가능 여부와 probability_semantics를 함께 기록한다. rescue/harm은 기준 답의 맞고 틀림을 실제 정답과 비교하는 별도 변화 지표다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 08 평가 지표·Rescue/Harm
VQA_Ensemble_Baseline.py: metrics, confidence_margin, rescue_harm

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `metrics`: 528~545행 | 슬라이드의 근거 함수·자료형 |
| `confidence_margin`: 524~526행 | 슬라이드의 근거 함수·자료형 |
| `rescue_harm`: 547~550행 | 슬라이드의 근거 함수·자료형 |

### 14. 실제 soft 확률의 평균

설명용 확률  /  같은 문항·같은 클래스의 값을 결합한다

설명용 입력은 A=(.50,.40,.05,.05), B=(.10,.80,.05,.05), C=(.20,.70,.05,.05)다. 균등 평균은 (.2667,.6333,.05,.05), A:B:C=2:1:1 가중 평균은 (.325,.575,.05,.05)가 된다. 네 클래스 중 평균 확률이 가장 큰 b를 고른다. soft_mean은 실제 soft 모델만 균등 결합하고, soft_fixed는 명시 가중치의 양수 합으로 정규화한다. 지정하지 않은 모델 가중치는 1이며 전체 합이 0이거나 잘못된 모델 이름이면 실패한다. 이 확률 그림은 대회 실행 결과가 아니다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 09 온도·soft 가중치 · 12 후보 예측
VQA_Ensemble_Baseline.py: fixed_weights, recipe_candidates

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `fixed_weights`: 585~592행 | 슬라이드의 근거 함수·자료형 |
| `recipe_candidates`: 727~788행 | 슬라이드의 근거 함수·자료형 |

### 15. 온도 보정과 학습 가중치

선택 옵션이며 meta-fit 정답만 사용해 학습한다

온도 보정은 raw VLM logits를 쓰지 않고 final probability p를 p^(1/T)로 변환한 뒤 다시 정규화한다. 구현은 log(p)/T에 softmax를 적용한다. T의 기본 탐색 범위는 .5~3이며 meta-fit NLL을 줄이는 값을 찾는다. 개선하지 않으면 T=1을 유지한다. 한 모델에서 양수 T 보정은 확률 순위를 유지하지만 서로 다른 모델 확률을 결합할 때 상대 confidence가 달라진다. soft_learned는 합1·0~1의 simplex 가중치로 NLL을 최소화하며 균등 가중치에서 벗어나는 L2 규제 .01을 더한다. 두 기능은 기본 off다. OOF에서는 각 fold의 학습 부분만 사용해 다시 fitting한다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 09 온도·soft 가중치 학습
VQA_Ensemble_Baseline.py: temperature_probs, fit_temperature, fit_soft_weights

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `temperature_probs`: 566~571행 | 슬라이드의 근거 함수·자료형 |
| `fit_temperature`: 573~583행 | 슬라이드의 근거 함수·자료형 |
| `fit_soft_weights`: 594~610행 | 슬라이드의 근거 함수·자료형 |

### 16. hard 투표와 동점

soft 모델의 argmax도 투표에서는 한 표가 된다

hard_vote는 complete로 등록한 모든 모델의 hard 답을 1표씩 세며 soft 모델도 argmax가 1표다. hard_weighted는 명시된 모델 가중치로 표 점수를 결합한다. 득표 수와 가중 점수는 actual soft probabilities가 아니다. 동점에서는 anchor 답이 최대 득표 클래스 중 하나면 anchor를 선택하고, 아니면 canonical a/b/c/d 중 첫 최대값을 선택한다. 아래 첫 행은 3대2로 b, 두 번째는 b/c 동점에 anchor=c라 c, 세 번째는 anchor=d가 최대 득표에 없어서 b다. 임의 난수로 동점을 처리하지 않는다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 10 투표·만장일치
VQA_Ensemble_Baseline.py: vote_counts, resolve_tie, recipe_candidates

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `vote_counts`: 628~631행 | 슬라이드의 근거 함수·자료형 |
| `resolve_tie`: 633~639행 | 슬라이드의 근거 함수·자료형 |
| `recipe_candidates`: 727~788행 | 슬라이드의 근거 함수·자료형 |

### 17. core3와 4·8개 만장일치

선택한 모델 전원이 동의할 때만 anchor 답을 교체한다

core_unanimous의 기본 core는 anchor를 제외한 meta-fit 정확도 상위 최대 3개다. names를 지정하면 그 구성대로 사용한다. 구성원이 2개 미만이면 후보가 없다. unanimous_4/8은 complete 모델 수가 충분할 때 meta-fit 순위의 상위4/8개를 사용한다. 해당 subset이 같은 답이면 그 답을, 아니면 anchor를 쓴다. subset에 anchor가 들어 있고 모든 구성원이 동의하면 그 답은 이미 anchor 답과 같으므로 실제 교체가 없을 수 있다. 같은 모델의 비슷한 변형 여러 개는 독립된 판단을 보장하지 않는다. 아래 core 예시는 실제 답이 아니라 규칙 설명이다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 10 투표·만장일치
VQA_Ensemble_Baseline.py: rank_models, recipe_candidates

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `rank_models`: 623~624행 | 슬라이드의 근거 함수·자료형 |
| `recipe_candidates`: 727~788행 | 슬라이드의 근거 함수·자료형 |

### 18. anchor 답의 조건부 교체

proposal을 지지하는 수와 확률 조건을 함께 확인한다

anchor_soft_guard의 proposal 기본값은 soft_fixed다. anchor 자체가 soft이고 proposal도 확률 후보이며, anchor 외 최소2개의 complete 모델이 있어야 후보를 만든다. proposal이 anchor와 다른 답일 때, anchor 제외 지지수≥2, proposal confidence≥.8, margin≥.15, anchor 최대확률≤.9를 모두 만족해야 교체한다. soft_hard_guard는 여기에 전체 hard 다수결 답=proposal 및 최대 득표 비율≥.6 조건을 추가한다. RF guard도 이 hard gate를 사용한다. 값은 검증 전 시작 임계값이다. 교체한 행은 proposal 확률을, 다른 행은 anchor의 원래 확률을 저장하는 routing 방식이다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 10 기준 모델 보호
VQA_Ensemble_Baseline.py: guard_mask, recipe_candidates

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `guard_mask`: 641~652행 | 슬라이드의 근거 함수·자료형 |
| `recipe_candidates`: 727~788행 | 슬라이드의 근거 함수·자료형 |

### 19. guard의 통과·보류 예시

각 조건을 모두 만족한 행만 확률 벡터를 바꾼다

설명용 행1에서 anchor=a, 최대확률 .65, proposal=b, confidence .86, margin .77, anchor 외 지지수2를 가정한다. anchor_soft_guard는 모두 만족해 b로 바꾼다. 행2는 proposal은 같지만 confidence .72여서 유지한다. 행3은 proposal .86이어도 anchor confidence .96으로 .9 상한을 넘어서 유지한다. soft_hard_guard나 rf_guard는 추가로 hard majority와 .6 득표 조건을 통과해야 하므로 이 표만으로 그 후보의 통과를 판정할 수 없다. true label은 제공하지 않았으므로 이 교체가 정답 개선인지는 이 예시에서 판단하지 않는다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 10 기준 모델 보호
VQA_Ensemble_Baseline.py: guard_mask

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `guard_mask`: 641~652행 | 슬라이드의 근거 함수·자료형 |

### 20. Random Forest의 입력 feature

모델별 예측과 의견 차이를 한 행의 숫자로 구성한다

soft 모델 하나는 real_soft 4개, 선택적 derived_log_probability 4개, confidence/margin/entropy 3개를 제공한다. 기본으로 soft 모델당 11개다. hard-only 모델은 a/b/c/d 4개 indicator를 제공하고, 마지막에 전체 모델 vote fraction 4개를 붙인다. 따라서 soft3+hard1의 기본 입력 차원은 3×11+4+4=41이다. soft 모델의 hard indicator는 별도로 중복 추가하지 않는다. log probability는 derived feature이며 원본 logits가 아니다. hard indicator도 범주형 feature이며 실제 모델 soft로 저장하지 않는다. feature 이름과 순서를 fit_parameters에 기록하고 test와 동일하게 유지한다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 11 Logistic/RF 선택형 메타 모델
VQA_Ensemble_Baseline.py: meta_features

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `meta_features`: 656~675행 | 슬라이드의 근거 함수·자료형 |

### 21. RF 학습과 test 적용

학습한 메타 분류기가 네 클래스의 새 확률을 출력한다

RF는 원 이미지를 보고 VQA를 학습하는 모델이 아니라, base 모델 출력으로 구성한 X를 입력받아 y를 예측하는 메타 분류기다. fit_state는 meta-fit의 행만 이용해 온도·가중치·estimator를 학습한다. predict_proba의 class order도 classes_를 보고 0/1/2/3으로 재정렬한다. 같은 estimator에 holdout/test feature를 입력한다. rf_meta는 바로 그 argmax, rf_guard는 anchor/RF 확률을 조건에 따라 routing한 후보다. v1은 meta-holdout까지 포함하는 재학습을 허용하지 않는다. 현재 구현은 fitted estimator 바이너리를 별도로 저장하지 않고 설정·입력 해시·학습 IDs·feature 순서를 기록하므로 다시 export하려면 같은 입력으로 fitting을 다시 실행한다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 11 메타 학습 · 12 후보 예측
VQA_Ensemble_Baseline.py: fit_state, build_meta_model, model_probabilities, recipe_candidates

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `fit_state`: 692~706행 | 슬라이드의 근거 함수·자료형 |
| `build_meta_model`: 677~683행 | 슬라이드의 근거 함수·자료형 |
| `model_probabilities`: 685~690행 | 슬라이드의 근거 함수·자료형 |
| `recipe_candidates`: 727~788행 | 슬라이드의 근거 함수·자료형 |

### 22. RF 초기값과 실행 옵션

기본 off이며 설정에서 활성화한다

RF 초기값은 trees=300, max_depth=6, min_samples_leaf=12, max_features=sqrt, n_jobs=2, random_state=seed42다. 작은 validation에서 복잡도를 제한하는 시작 설정이다. 이 설정이 과거 RF_v1의 실제 설정과 같다고 확인하지 않았다. RF를 활성화하면 fit_oof 자동선택 여부와 상관없이 need_oof가 켜져 내부 OOF를 수행한다. 자동 선택을 켜려면 policy=fit_oof와 candidates를 명시하면 된다. 후보는 실제 활성화되어 있고 모든 fold OOF가 있는 이름이어야 한다. Logistic은 선택적 비교 후보로 StandardScaler와 C=.3, max_iter=1000을 사용한다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 11 Logistic/RF 선택형 메타 모델 · 20 설정
VQA_Ensemble_Baseline.py: default_config, build_meta_model

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `default_config`: 88~127행 | 슬라이드의 근거 함수·자료형 |
| `build_meta_model`: 677~683행 | 슬라이드의 근거 함수·자료형 |

### 23. 후보별 출력 의미

hard 답·확률·득표 수를 서로 다른 필드에 담는다

Candidate는 prediction, probability, counts, semantics, changed를 구분한다. soft_mean/fixed/learned는 실제 확률 평균이다. hard_vote와 core/unanimous는 hard 답과 득표 수를 담고 probability를 만들지 않는다. guard는 anchor 또는 proposal 확률을 행별로 선택한다. rf_meta/logistic_meta는 새 분류기의 predict_proba이며 VLM의 raw probabilities와 출처가 다르다. core fallback에서 counts는 진단용이므로 counts argmax와 최종 답이 항상 같다고 볼 수 없다. 최종 답은 prediction과 CSV가 기준이다. probability가 존재하면 export가 argmax와 prediction 일치를 검사한다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 12 후보 실행·예측 유형
VQA_Ensemble_Baseline.py: Candidate, recipe_candidates, export_candidate

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `Candidate`: 719~725행 | 슬라이드의 근거 함수·자료형 |
| `recipe_candidates`: 727~788행 | 슬라이드의 근거 함수·자료형 |
| `export_candidate`: 997~1018행 | 슬라이드의 근거 함수·자료형 |

### 24. meta-fit 내부 OOF

평가할 행을 제외하고 메타 설정을 다시 학습한다

기본 내부5-fold는 meta-fit 안에서만 구성하며 meta-holdout은 포함하지 않는다. 각 fold train에서 anchor와 모델 순위, 온도, soft 가중치, RF/Logistic을 다시 학습하고 제외한 fold를 예측한다. OOF를 합치면 각 meta-fit 행은 자신을 학습하지 않은 메타 state가 예측한 답을 하나 갖는다. base VLM 가중치를 fold마다 다시 학습하는 과정은 아니다. base 모델은 이 공통 validation을 원 VLM 학습에서 제외해야 한다. fold별 anchor 유형 변화 때문에 후보가 모든 fold에 없거나 probability 타입이 다르면 그 후보는 OOF 선택에서 제외하며 audit에 남긴다. fold-local anchor이므로 OOF의 기준 답은 한 고정 anchor 답과 다를 수 있다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 13 Meta-fit 내부 OOF
VQA_Ensemble_Baseline.py: build_fit_oof, grouped_folds, fit_state

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `build_fit_oof`: 792~829행 | 슬라이드의 근거 함수·자료형 |
| `grouped_folds`: 473~486행 | 슬라이드의 근거 함수·자료형 |
| `fit_state`: 692~706행 | 슬라이드의 근거 함수·자료형 |

### 25. 후보 선택과 holdout 평가

선택을 저장한 다음 holdout 결과를 계산한다

fixed는 사전에 지정한 fixed_recipe를 고르고, 현재 입력으로 후보를 만들 수 없으면 fallback_recipe=anchor를 쓴다. fit_oof는 meta-fit OOF의 accuracy 우선으로 비교한다. 기본 min_fit_oof_gain_count=1은 anchor보다 net1문항 이상인 비anchor 후보만 eligible로 한다. harm 상한은 기본None이며 지정할 수 있다. 동점이면 harm가 적고 changed가 적은 후보를 고른다. 선택 결과를 analysis/selection.json에 저장한 뒤 holdout 평가를 보고한다. OOF 점수는 후보 선택에 사용했으므로 독립 최종 성적이 아니다. holdout 비교표를 보고 같은 holdout에 맞춰 여러 번 규칙을 고치면 독립성이 약해진다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 14 후보 선택·holdout 동결
VQA_Ensemble_Baseline.py: select_recipe, run_ensemble

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `select_recipe`: 833~872행 | 슬라이드의 근거 함수·자료형 |
| `run_ensemble`: 1113~1229행 | 슬라이드의 근거 함수·자료형 |

### 26. Rescue·Harm과 실행 비교

설명용 100문항  /  바꾼 답의 구제와 정답 훼손을 비교한다

설명용 평가100문항에서 anchor가90개, 새 후보가92개를 맞혔다고 하자. anchor가 틀렸던5개를 구제하고 맞았던3개를 훼손하면 net=5-3=2로 정답 수가90에서92로 늘어난다. 변경 수는8개 이상일 수 있다. 두 후보 모두 틀리지만 서로 다른 오답을 선택한 행은 changed에는 들어가고 rescue/harm에는 들어가지 않는다. 실제 comparison.csv는 정확도, correct, changed, rescue, harm, net, 확률 지표를 평가 범위별로 저장한다. 후보끼리는 동일 IDs와 동일 evaluation_role에서 비교한다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 08 지표 · 15 결과 비교
VQA_Ensemble_Baseline.py: rescue_harm, evaluation_table

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `rescue_harm`: 547~550행 | 슬라이드의 근거 함수·자료형 |
| `evaluation_table`: 874~884행 | 슬라이드의 근거 함수·자료형 |

### 27. 실제 결과 차트와 평가 범위

run의 comparison.csv를 report.html에서 확인한다

write_report는 실제 실행 comparison.csv와 summary의 수치로 HTML/SVG 그래프를 만들고 검색과 정렬을 제공한다. 본 PPT의 예시 차트와 실제 run 차트를 구분한다. meta_fit_in_sample_diagnostic는 메타 학습에 쓴 부분, meta_fit_oof_selection_diagnostic는 내부 cross-fit 선택 점수, meta_holdout은 앙상블 선택에서 제외한 부분, full_validation_mixed_diagnostic는 두 범위가 섞인 진단이다. summary의 최상위 accuracy도 full mixed이므로 independent score로 말하면 안 된다. selected_holdout이 선택된 후보의 holdout 지표다. base_model_validation.csv는 common_base_model_validation 범위의 단일 모델 결과다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 16 실제 차트·실행 감사
VQA_Ensemble_Baseline.py: write_report, evaluation_table, run_ensemble

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `write_report`: 936~956행 | 슬라이드의 근거 함수·자료형 |
| `evaluation_table`: 874~884행 | 슬라이드의 근거 함수·자료형 |
| `run_ensemble`: 1113~1229행 | 슬라이드의 근거 함수·자료형 |

### 28. test 결합과 제출 CSV

동결한 설정을 test에 적용하고 sample ID 순서로 저장한다

export 모드는 먼저 complete 모델의 공통 valid를 검사하고 meta-fit state와 선택 규칙을 다시 실행한다. 같은 state로 test 후보를 만들며 valid/test 후보 구성 일치를 확인한다. sample_submission_csv의 ID 집합과 순서를 test 기준으로 사용한다. make_submission은 두 열 id,answer와 a/b/c/d 답, ID 중복, 행 수를 검사하고 저장 후 다시 읽어 확인한다. 이 프로젝트의 test는6714개라는 사용자 기록이 있지만 재사용 코드는 숫자를 하드코딩하지 않고 실제 sample 행 수에 따른다. 아래 CSV는 형식 예시로 실제 제출물과 무관하다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 17 제출 CSV·NPY 저장
VQA_Ensemble_Baseline.py: canonical_test, recipe_candidates, make_submission

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `canonical_test`: 446~462행 | 슬라이드의 근거 함수·자료형 |
| `recipe_candidates`: 727~788행 | 슬라이드의 근거 함수·자료형 |
| `make_submission`: 1020~1029행 | 슬라이드의 근거 함수·자료형 |

### 29. 앙상블 NPY 출력

확률이 있는 후보만 *_probs.npy를 만든다

모든 후보는 IDs와 predictions를 저장하며 validation이면 labels도 저장한다. probability 방식이면 shape(N,4), 행합1, 유한성, argmax와 prediction 일치를 검사하고 float32 probs를 저장한다. hard 방식은 probs를 만들지 않고 vote_counts를 별도로 저장할 수 있다. 앙상블에는 공통 물리적인 token logits가 없으므로 log(mean_probs)를 raw logits처럼 만들어 저장하지 않는다. full valid 출력에는 meta-fit in-sample 부분이 포함될 수 있고 probability_metadata에 meta_in_sample_present를 남긴다. 상위 메타 앙상블 입력으로 full valid를 곧바로 쓰는 것을 loader가 거부한다. OOF와 holdout 출력의 ID 범위도 확인해야 한다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 17 제출 CSV·NPY 저장
VQA_Ensemble_Baseline.py: export_candidate

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `export_candidate`: 997~1018행 | 슬라이드의 근거 함수·자료형 |

### 30. run 폴더와 감사 기록

어떤 입력과 규칙으로 결과를 만들었는지 남긴다

create_run_dir는 기존 run을 덮어쓰지 않는다. config.json/environment.json/inputs.json은 설정, 설치 환경, 입력 경로와 해시를 남긴다. meta_split.json은 fit/holdout IDs와 서명을, fit_parameters.json은 anchor/ranking/temperature/weights/feature 순서 및 fit IDs를, analysis/selection.json은 선택 근거와 config/meta split signatures를 담는다. candidates 아래의 파일은 후보 출력이며 leaderboard에 제출했다고 자동 기록하지 않는다. output prob metadata가 원본 입력과 이후 재활용 방식의 의미를 잇는다. estimator 자체 joblib/pickle을 저장하는 기능은 이 v1에서 구현하지 않았으므로 원본 예측과 환경을 유지해 재실행한다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 16 실행 감사 · 17 저장
VQA_Ensemble_Baseline.py: state_summary, run_ensemble, create_run_dir

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `state_summary`: 708~714행 | 슬라이드의 근거 함수·자료형 |
| `run_ensemble`: 1113~1229행 | 슬라이드의 근거 함수·자료형 |
| `create_run_dir`: 1101~1111행 | 슬라이드의 근거 함수·자료형 |

### 31. test soft만 있는 모델

validation 학습·순위·자동 선택에서 분리한다

validation이 없는 모델은 role=test_only로 등록하고 test_only_recipes에 고정 결합 규칙을 명시한다. soft_mean이면 모든 구성 모델에 actual test_probs가 있어야 하고 hard_vote면 답 투표를 할 수 있다. complete 모델의 메타 학습과 자동 선택에서는 제외하고 test_only/<recipe>/submission_TESTONLY_<name>.csv로 저장한다. 별도 validation 점수는 없다. 과거 qwen_og는 test soft만 확인되었다는 복구 기록이 있으므로 그 상태를 complete로 간주하지 않는다. 원본에 공통 valid 출력이 발견되면 실제 확인 후 role을 변경한다. 전체가 test_only인 작업은 이 베이스에 complete 모델이 최소1개 필요하므로 실행 대상이 아니다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 18 Test-only 선택 분기
VQA_Ensemble_Baseline.py: run_test_only_recipes, load_model

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `run_test_only_recipes`: 1033~1066행 | 슬라이드의 근거 함수·자료형 |
| `load_model`: 257~352행 | 슬라이드의 근거 함수·자료형 |

### 32. 수동 검토와 수정 기록

의견 불일치 문항을 확인하고 수정 CSV를 별도로 남긴다

review_queue는 complete 모델 간 의견 불일치 또는 anchor에서 최종 답이 바뀐 test 문항을 모은다. ground_truth는 unknown이며 queue만으로 정답을 확정할 수 없다. 실제 질문·보기·이미지는 upstream ID로 연결해 사람이 함께 확인한다. 수정 CSV는 id,before,after,reason을 갖고 중복ID, test외ID, before 불일치, 같은 답으로의 변경, 빈 근거를 검사한다. 원 selected CSV를 유지하며 별도 submission_MANUAL_reviewed.csv와 manual_changes_applied/summary를 만든다. 모델 soft 확률은 수동 답 변경에 맞춰 변조하지 않는다. 수동 CSV와 원 probability argmax가 다를 수 있으므로 별도 산출물임을 유지한다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 18 수동 검토 선택 분기
VQA_Ensemble_Baseline.py: review_queue, manual_changes

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `review_queue`: 919~932행 | 슬라이드의 근거 함수·자료형 |
| `manual_changes`: 1068~1097행 | 슬라이드의 근거 함수·자료형 |

### 33. 실행 순서

경로를 채운 뒤 inspect, evaluate, export 순서로 실행한다

노트북20번 셀의 RUN_MODE가 기본 inspect다. inspect는 입력과 공통 validation/test를 검사하고 catalog를 기록하며 앙상블 fitting·CSV 생성을 하지 않는다. evaluate는 valid만으로 meta 분할·학습·OOF·선택·평가를 실행한다. export는 같은 처리와 test 결합·CSV 저장을 수행한다. 각 모드는 새 run을 만들며 기존 run을 덮어쓰지 않는다. evaluate에서 만든 estimator를 export에 직렬화해 이어 쓰는 구조가 아니라 같은 설정/seed/입력으로 fitting을 재실행한다. 고정 manifest_path를 사용하면 meta 분할 ID도 재사용할 수 있다. self_check는 CPU 합성 fixture의 코드 검사이며 실제 대회 성적을 확인하지 않는다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 19 실행기 · 20 실행 설정
VQA_Ensemble_Baseline.py: run_ensemble, self_check, main

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `run_ensemble`: 1113~1229행 | 슬라이드의 근거 함수·자료형 |
| `self_check`: 1231~1267행 | 슬라이드의 근거 함수·자료형 |
| `main`: 1269~1282행 | 슬라이드의 근거 함수·자료형 |

### 34. 과거 RF_v1과 최종 A의 계보

사용자 제공 기록과 새 베이스 구현을 구분한다

사용자 제공 기록상 기존 RF_v1을 기준으로 수동 검토18문항을 거쳐 A에서15개 답을 수정했다. A 파일은 submission_guard_A_RF_repaired15.csv다. B submission_guard_B_core3_plus_RF21.csv는 별도 예비안으로 성적이 좋지 않았으며 정확한 수치는 제공되지 않았다. A 이후 submission_postA_consensus8of9_nonmanual4.csv는 추가4개 답을 바꿔 A보다 낮은 성적을 냈다. 새 노트북의 rf_meta/rf_guard는 이 과거 코드·18문항 검토·15수정 규칙을 그대로 복제한 것이 아니다. 원 RF 학습 입력, 모델 조합, hyperparameter, 변경 IDs와 audit를 직접 대조하지 않았으므로 정확한 재현은 미확인이다.

코드·자료 근거
사용자 제공 프로젝트 설명 및 기존 모델편

### 35. 제출로 확인된 과거 결과

A는 확인된 최고 제출이며 가능한 모든 조합의 최적해는 미확인이다

수치는 사용자 제공 성적 기록이다. A submission_guard_A_RF_repaired15.csv는 Public=.98182,Private=.98302. B의 정확한 점수는 제공되지 않아 미확인으로 남긴다. A 이후 추가4수정 submission_postA_consensus8of9_nonmanual4.csv는 Public=.98153,Private=.98272로 둘 다 A보다 낮다. A를 알려진 제출 중 최고라고 말할 수 있지만 모든 가능한 앙상블의 최적해라고 할 수는 없다. Public과 Private는 test의 서로 다른 숨겨진 subset의 집계다. aggregate 차이만으로 특정 수정 문항의 정답을 확정할 수 없다. Private 결과를 반복 사용해 예외와 규칙을 바꾸면 Private에도 과적합할 수 있다. 새 베이스로 이 점수를 재현하거나 개선했다고 주장하지 않는다.

코드·자료 근거
사용자 제공 프로젝트 설명 및 기존 모델편

### 36. 전체 단계와 공부 순서

모델 출력의 의미부터 RF 선택·최종 감사까지 순서대로 읽는다

01 환경,02 공통 함수,03 설정,04 ID/label/class,05 모델 load,06 공통 alignment,07 이미지 group meta split,08 metrics,09 temperature/weights,10 voting/core/guard,11 RF/Logistic,12 candidates,13 OOF,14 selection,15 compare/review,16 charts/audit,17 output,18 test-only/manual,19 runner/self-check,20 user setup를 연결한 학습 지도다. companion Code_Guide는 슬라이드별 상세 해설과 정확한 함수 위치·핵심 코드 및 실행 옵션을 함께 제공한다. 실제 대회 파일을 입력한 새 베이스의 Public/Private 성적과 과거 RF_v1 원본 설정은 미확인이다. 설명용 차트/숫자는 실제 결과로 사용하지 않는다.

코드·자료 근거
VQA_Ensemble_Baseline_v1.ipynb의 20단계 학습 지도
VQA_Ensemble_Baseline.py: run_ensemble

| 코드 위치 (`VQA_Ensemble_Baseline.py`) | 역할 |
|---|---|
| `run_ensemble`: 1113~1229행 | 슬라이드의 근거 함수·자료형 |

## 설명용 수치의 계산 근거

### soft 평균

모델 A=(0.50,0.40,0.05,0.05), B=(0.10,0.80,0.05,0.05), C=(0.20,0.70,0.05,0.05).

- 균등 평균: (0.266666…, 0.633333…, 0.05, 0.05). PPT 차트는 소수 넷째 자리로 표시한다.
- 가중치 2:1:1: (0.325, 0.575, 0.05, 0.05).
- 둘 다 argmax=b다. 이 값은 설명용이며 실제 모델 run 성능을 나타내지 않는다.

### RF feature 차원

기본 설정에서 soft 모델 하나는 확률4 + log 확률4 + confidence/margin/entropy3 = 11개다. hard-only 모델 하나는 indicator4개다. 전체 투표 비율4개를 더한다. soft3개+hard1개는 3×11+4+4=41개 feature다. optional feature를 끄면 차원이 달라진다.

### Rescue와 Harm

anchor90/100, 후보92/100, rescue5, harm3이면 net=2다. 변경 문항 수는 두 후보가 서로 다른 오답을 고른 경우까지 포함하므로 rescue+harm와 같을 필요가 없다.

## 산출물 위치

```text
ensemble_runs/<new_run>/
  config.json / environment.json / inputs.json
  execution_status.json / meta_split.json / fit_parameters.json
  validation_predictions.csv / validation_summary.json
  test_predictions.csv / submission_selected_<recipe>.csv
  probs/
    valid_ids.json / valid_labels.npy / valid_predictions.npy
    valid_probs.npy                     # probability 후보인 경우만
    test_ids.json / test_predictions.npy / test_probs.npy
    valid_vote_counts.npy / test_vote_counts.npy  # counts가 있는 경우
    meta_fit_oof_valid_*                 # OOF를 수행한 경우
    probability_metadata.json
  analysis/
    input_catalog.csv / selection.json / comparison.csv
    base_model_validation.csv / pairwise_full_validation_diagnostic.csv
    validation_consensus_patterns.csv / selected_holdout_predictions.csv
    selected_fit_oof_predictions.csv / meta_fit_oof_audit.json
    test_review_queue.csv / test_only_recipes.json
    manual_changes_applied.csv / manual_changes_summary.json
  candidates/<recipe>/
  test_only/<recipe>/
  charts/report.html / README.md
```

hard-only 후보는 `*_probs.npy`를 만들지 않는다. 위 일부 파일은 활성 옵션·실행 모드에 따라 존재한다. 앙상블 `*_logits.npy`를 임의로 만들어 내지 않는다. 후보별 CSV는 생성 사실과 실제 대회 제출·채점 사실을 구분해야 한다.

상위 앙상블에 다시 사용할 때 full valid에는 meta-fit in-sample 부분이 섞일 수 있다. `meta_in_sample_present`를 확인하고 일관된 OOF 또는 독립 holdout 출력과 ID를 별도로 준비한다.

## 과거 제출 기록

| 파일 | 사용자 제공 계보 | Public | Private |
|---|---|---:|---:|
| `submission_guard_A_RF_repaired15.csv` | RF_v1 기준 15개 답 수정 | 0.98182 | 0.98302 |
| `submission_guard_B_core3_plus_RF21.csv` | 별도 예비안 | 미확인 | 미확인 |
| `submission_postA_consensus8of9_nonmanual4.csv` | A에서 추가4개 수정 | 0.98153 | 0.98272 |

수동 검토18문항과 최종 A 수정15개는 다른 수치다. A는 확인된 제출 중 최고이고 모든 가능한 앙상블의 최적해는 미확인이다. Public·Private의 집계 차이만으로 특정 문항의 정답을 확정할 수 없다. 숨겨진 성적을 반복 참조해 규칙을 고치면 그 집합에도 과적합할 수 있다.

## 확인이 더 필요한 자료

- 원본 RF_v1 코드·학습 feature·train/holdout 구성·실행 설정과 사용 모델 목록.
- 수동 검토18문항과 A의15개 수정, B, A 이후4개 변경의 실제 ID·before/after·판단 근거.
- 모델별 원본 valid/test soft 파일과 metadata, checkpoint·추론 설정 대응 관계.
- base VLM의 학습 목록·validation 선택 이력·이미지 그룹·근접 중복 검사.
- 새 베이스에 실제 모델 출력들을 입력한 결과 및 실제 대회 제출 성적.

## 원본 코드 발췌

아래는 현재 파일에서 그대로 추출한 함수들이다. 함수 간 의존성이 있으므로 독립 셀로 실행하기보다 원본 노트북/Python 파일과 함께 읽는다. 구현을 변경한 코드는 포함하지 않았다.

### 기본 설정

`default_config` — 원본 88~127행

```python
def default_config():
    return {
        'schema_version': 1, 'classes': CLASSES.copy(), 'seed': 42,
        'models': [], 'validation_csv': None, 'sample_submission_csv': None,
        'validation_columns': {'id': 'id', 'answer': 'answer', 'group': None},
        'validation_is_gold_confirmed': False,
        'sample_submission_columns': {'id': 'id', 'answer': 'answer'},
        'output_root': 'ensemble_runs', 'run_id': None,
        'strict': {'reject_smoke': True, 'reject_onehot_like_soft': True,
                   'allow_unique_id_groups': False,
                   'require_heldout_confirmation': True, 'require_task_confirmation_if_no_rows': True,
                   'probability_sum_tolerance': 0.002},
        'meta_split': {'holdout_fraction': 0.2, 'method': 'StratifiedGroupKFold',
                       'manifest_path': None, 'n_inner_folds': 5},
        'anchor': {'name': None, 'selection': 'best_on_meta_fit'},
        'calibration': {'mode': 'none', 'temperature_bounds': [0.5, 3.0], 'ece_bins': 10},
        'soft': {'fixed_weights': {}, 'use_calibrated': True},
        'hard': {'fixed_weights': {}, 'tie_break': 'anchor_if_tied_then_abcd'},
        'learned_weights': {'enabled': False, 'l2_to_uniform': 0.01, 'maxiter': 300},
        'guard': {'min_supporters_excluding_anchor': 2, 'min_proposal_confidence': 0.8,
                  'min_proposal_margin': 0.15, 'max_anchor_confidence': 0.9,
                  'min_vote_fraction': 0.6, 'proposal': 'soft_fixed'},
        'core': {'names': [], 'auto_size': 3, 'exclude_anchor_from_auto': True, 'unanimity_sizes': [4, 8]},
        'meta_models': {
            'logistic': {'enabled': False, 'C': 0.3, 'max_iter': 1000},
            'rf': {'enabled': False, 'n_estimators': 300, 'max_depth': 6,
                   'min_samples_leaf': 12, 'max_features': 'sqrt', 'n_jobs': 2},
            'features': {'include_soft_log_probs': True, 'include_hard_indicators': True}},
        'selection': {'policy': 'fixed', 'fixed_recipe': 'anchor_soft_guard',
                      'fallback_recipe': 'anchor', 'candidates': [],
                      'min_fit_oof_gain_count': 1, 'max_fit_oof_harm_count': None,
                      'force_inner_oof': False, 'refit_on_all_validation': False},
        'test_only_recipes': [],
        'manual_review': {'enabled': False, 'changes_csv': None, 'require_reason': True,
                          'export_queue': True, 'queue_limit': None},
        'artifacts': {'save_all_candidate_submissions': True, 'write_charts': True,
                      'save_aligned_inputs': False},
        'evidence': {'template_authoring_real_competition_predictions_executed': False,
                     'public_private': None, 'reproduces_historical_RF_v1_or_A': False},
    }
```

### ID·클래스·확률 검증

`class_permutation` — 원본 201~204행

```python
def class_permutation(order):
    if not isinstance(order, list) or len(order) != 4 or set(order) != set(CLASSES):
        raise ValueError('soft 열의 class_order를 명시하세요. a/b/c/d를 한 번씩 포함해야 합니다.')
    return [order.index(c) for c in CLASSES]
```

`align_array` — 원본 206~214행

```python
def align_array(values, source_ids, target_ids, context):
    if len(values) != len(source_ids):
        raise ValueError(f'{context}: ID와 행 수가 다릅니다.')
    if set(source_ids) != set(target_ids):
        missing = sorted(set(target_ids)-set(source_ids))[:5]
        extra = sorted(set(source_ids)-set(target_ids))[:5]
        raise ValueError(f'{context}: 평가 집합이 다릅니다. missing={missing}, extra={extra}; 교집합 자동 비교를 하지 않습니다.')
    lookup = {v: i for i, v in enumerate(source_ids)}
    return np.asarray(values)[[lookup[v] for v in target_ids]]
```

`check_probs` — 원본 216~228행

```python
def check_probs(p, n, context, cfg):
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (n, 4) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1.000001):
        raise ValueError(f'{context}: 유한한 (N,4) 확률이 아닙니다.')
    sums = p.sum(1)
    if np.any(abs(sums - 1) > cfg['strict']['probability_sum_tolerance']):
        raise ValueError(f'{context}: 확률 행 합이 1이 아닙니다. logits는 probs로 등록하지 마세요.')
    if cfg['strict']['reject_onehot_like_soft'] and np.all((p == 0) | (p == 1)):
        raise ValueError(f'{context}: 전체가 one-hot입니다. hard 답을 실제 soft로 등록할 수 없습니다.')
    if np.any(sums <= 0):
        raise ValueError(f'{context}: 0 합 확률')
    # 저장 반올림 오차만 보정하고 원본 해시/정규화 여부를 감사한다.
    return p / sums[:, None]
```

### meta 분할

`grouped_folds` — 원본 473~486행

```python
def grouped_folds(labels,groups,k,seed):
    if len(np.unique(groups))<k:
        raise ValueError(f'{k} folds를 만들 이미지 그룹이 부족합니다.')
    if min(np.bincount(labels,minlength=4))<k:
        raise ValueError(f'각 정답 클래스가 {k}개 이상이어야 합니다. fold 수를 줄이세요.')
    splitter=StratifiedGroupKFold(n_splits=k,shuffle=True,random_state=seed)
    result=[]
    for tr,va in splitter.split(np.zeros(len(labels)),labels,groups):
        if set(groups[tr])&set(groups[va]):
            raise AssertionError('meta group 누수')
        if set(labels[tr])!={0,1,2,3} or set(labels[va])!={0,1,2,3}:
            raise ValueError('어떤 fold에 정답 클래스가 없습니다. 그룹 분포·fold 수를 확인하세요.')
        result.append((tr,va))
    return result
```

`make_meta_split` — 원본 488~520행

```python
def make_meta_split(canonical,cfg):
    if canonical['group_column'] is None and not cfg['strict']['allow_unique_id_groups']:
        raise ValueError('평가에는 이미지 그룹이 필요합니다. 이전 run의 validation_rows.csv를 연결하세요. ID별 그룹 예외는 같은 이미지가 없음을 확인한 경우에만 허용하세요.')
    y,g,ids=canonical['labels'],canonical['groups'],canonical['ids']
    sig=digest({'validation':canonical['signature'],'groups':g.tolist(),'seed':cfg['seed'],
                'fraction':cfg['meta_split']['holdout_fraction']})
    existing=cfg['meta_split']['manifest_path']
    if existing:
        info=read_json(existing)
        if info['signature']!=sig:
            raise ValueError('meta split manifest 서명이 다릅니다.')
        fit_ids,hold_ids=info['fit_ids'],info['holdout_ids']
        if set(fit_ids)&set(hold_ids) or set(fit_ids)|set(hold_ids)!=set(ids):
            raise ValueError('meta split ID 누락/중복')
        fit_ids=check_ids(fit_ids,'meta fit manifest');hold_ids=check_ids(hold_ids,'meta holdout manifest')
        lookup={v:i for i,v in enumerate(ids)}
        fit=np.array([lookup[v] for v in fit_ids],dtype=int);hold=np.array([lookup[v] for v in hold_ids],dtype=int)
    else:
        k=round(1/cfg['meta_split']['holdout_fraction'])
        dist=np.bincount(y,minlength=4)/len(y)
        choices=[]
        for fit,hold in grouped_folds(y,g,k,cfg['seed']):
            d=np.bincount(y[hold],minlength=4)/len(hold)
            loss=abs(len(hold)/len(y)-1/k)+abs(d-dist).mean()
            choices.append((float(loss),fit,hold))
        _,fit,hold=min(choices,key=lambda x:x[0])
        info={'signature':sig,'fit_ids':[ids[i] for i in fit],'holdout_ids':[ids[i] for i in hold],
              'method':'StratifiedGroupKFold_choose_distribution_fold','seed':cfg['seed'],
              'requested_holdout_fraction':cfg['meta_split']['holdout_fraction'],
              'realized_holdout_fraction':len(hold)/len(y),'group_column':canonical['group_column']}
    if set(g[fit])&set(g[hold]):
        raise ValueError('manifest의 fit/holdout 이미지 그룹 누수')
    return fit,hold,info
```

### 지표

`metrics` — 원본 528~545행

```python
def metrics(pred,y,probs=None,bins=10):
    correct=np.asarray(pred)==np.asarray(y)
    out={'n':len(y),'accuracy':float(correct.mean()),'correct':int(correct.sum()),
         'by_class':{c:{'n':int((y==i).sum()),'accuracy':float(correct[y==i].mean()) if (y==i).any() else None}
                     for i,c in enumerate(CLASSES)}}
    if probs is not None:
        clipped=np.clip(probs,1e-15,1)
        out['nll']=float(-np.log(clipped[np.arange(len(y)),y]).mean())
        target=np.eye(4)[y]
        out['brier']=float(np.square(probs-target).sum(1).mean())
        conf=probs.max(1)
        ece=0.0
        for k in range(bins):
            mask=(conf>=k/bins)&(conf<(k+1)/bins if k<bins-1 else conf<=1)
            if mask.any():
                ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
        out['ece']=float(ece)
    return out
```

`rescue_harm` — 원본 547~550행

```python
def rescue_harm(pred,anchor,y):
    a=anchor==y;b=pred==y
    rescue=int((~a&b).sum());harm=int((a&~b).sum())
    return {'changed':int((pred!=anchor).sum()),'rescue':rescue,'harm':harm,'net':rescue-harm}
```

### 온도·학습 가중치

`temperature_probs` — 원본 566~571행

```python
def temperature_probs(p,temperature):
    # log(p)로 만든 보정 입력이지 실제 VLM raw logits가 아니다.
    z=np.log(np.clip(p,1e-15,1))/float(temperature)
    z-=z.max(1,keepdims=True)
    q=np.exp(z)
    return q/q.sum(1,keepdims=True)
```

`fit_temperature` — 원본 573~583행

```python
def fit_temperature(p,y,cfg):
    if cfg['calibration']['mode']=='none':
        return 1.0
    def loss(t):
        q=temperature_probs(p,t)
        return float(-np.log(np.clip(q[np.arange(len(y)),y],1e-15,1)).mean())
    result=minimize_scalar(loss,bounds=cfg['calibration']['temperature_bounds'],method='bounded')
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError('온도 최적화 실패')
    # T=1보다 NLL이 나빠지는 수치 최적화 결과는 쓰지 않는다.
    return float(result.x) if loss(result.x)<loss(1.0) else 1.0
```

`fit_soft_weights` — 원본 594~610행

```python
def fit_soft_weights(stack,y,cfg):
    m=stack.shape[0]
    uniform=np.full(m,1/m)
    if m==1 or not cfg['learned_weights']['enabled']:
        return uniform
    target=stack[:,np.arange(len(y)),y]
    penalty=cfg['learned_weights']['l2_to_uniform']
    def loss(w):
        gold=np.clip(w@target,1e-15,1)
        return float(-np.log(gold).mean()+penalty*np.square(w-uniform).sum())
    result=minimize(loss,uniform,method='SLSQP',bounds=[(0,1)]*m,
                    constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                    options={'maxiter':cfg['learned_weights']['maxiter'],'ftol':1e-10})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('soft 가중치 최적화 실패: '+str(result.message))
    w=np.maximum(result.x,0)
    return w/w.sum()
```

### hard 투표·guard

`vote_counts` — 원본 628~631행

```python
def vote_counts(predictions,weights=None):
    votes=np.asarray(predictions,dtype=int)
    w=np.ones(len(votes)) if weights is None else np.asarray(weights)
    return np.stack([(w[:,None]*(votes==c)).sum(0) for c in range(4)],axis=1)
```

`resolve_tie` — 원본 633~639행

```python
def resolve_tie(counts,anchor):
    maxima=counts.max(1,keepdims=True)
    tied=np.isclose(counts,maxima,atol=1e-12,rtol=0)
    pred=counts.argmax(1)
    choose=tied[np.arange(len(anchor)),anchor]
    pred[choose]=anchor[choose]
    return pred
```

`guard_mask` — 원본 641~652행

```python
def guard_mask(proposal,anchor,soft,anchor_soft,supporters,votes,cfg,with_vote):
    conf,margin=confidence_margin(soft)
    anchor_conf=anchor_soft.max(1)
    support=(supporters==proposal[None,:]).sum(0) if len(supporters) else np.zeros(len(proposal),dtype=int)
    g=cfg['guard']
    mask=(proposal!=anchor)&(support>=g['min_supporters_excluding_anchor'])
    mask&=(conf>=g['min_proposal_confidence'])&(margin>=g['min_proposal_margin'])&(anchor_conf<=g['max_anchor_confidence'])
    if with_vote:
        counts=vote_counts(votes)
        voted=resolve_tie(counts,anchor)
        mask&=(voted==proposal)&(counts.max(1)/len(votes)>=g['min_vote_fraction'])
    return mask
```

### RF feature·모델

`meta_features` — 원본 656~675행

```python
def meta_features(models,split,idx,temperatures,cfg):
    pieces=[];columns=[]
    for m in models:
        p=getattr(m,split+'_probs')
        pred=getattr(m,split+'_pred')[idx]
        if p is not None:
            q=temperature_probs(p[idx],temperatures.get(m.name,1.)) if cfg['soft']['use_calibrated'] else p[idx]
            pieces.append(q);columns.extend([f'{m.name}:real_soft_{c}' for c in CLASSES])
            if cfg['meta_models']['features']['include_soft_log_probs']:
                pieces.append(np.log(np.clip(q,1e-12,1)));columns.extend([f'{m.name}:derived_log_probability_{c}' for c in CLASSES])
            conf,margin=confidence_margin(q)
            entropy=-(q*np.log(np.clip(q,1e-12,1))).sum(1)
            pieces.append(np.stack([conf,margin,entropy],axis=1))
            columns.extend([f'{m.name}:confidence',f'{m.name}:margin',f'{m.name}:entropy'])
        elif cfg['meta_models']['features']['include_hard_indicators']:
            # 범주형 메타 feature. 이 배열을 model probabilities로 저장하지 않는다.
            pieces.append(np.eye(4)[pred]);columns.extend([f'{m.name}:hard_indicator_{c}' for c in CLASSES])
    votes=np.stack([getattr(m,split+'_pred')[idx] for m in models])
    pieces.append(vote_counts(votes)/len(models));columns.extend([f'vote_fraction_feature_{c}' for c in CLASSES])
    return np.concatenate(pieces,axis=1),columns
```

`build_meta_model` — 원본 677~683행

```python
def build_meta_model(kind,cfg,seed):
    if kind=='logistic':
        options=cfg['meta_models']['logistic']
        return make_pipeline(StandardScaler(),LogisticRegression(C=options['C'],max_iter=options['max_iter'],random_state=seed))
    options=cfg['meta_models']['rf']
    return RandomForestClassifier(n_estimators=options['n_estimators'],max_depth=options['max_depth'],
          min_samples_leaf=options['min_samples_leaf'],max_features=options['max_features'],n_jobs=options['n_jobs'],random_state=seed)
```

`model_probabilities` — 원본 685~690행

```python
def model_probabilities(estimator,features):
    q=estimator.predict_proba(features)
    order=list(estimator.classes_)
    if set(order)!={0,1,2,3}:
        raise ValueError('메타 모델 학습 fold에 네 클래스가 모두 필요합니다.')
    return q[:,[order.index(i) for i in range(4)]]
```

`fit_state` — 원본 692~706행

```python
def fit_state(models,idx,y,cfg,anchor,ranking):
    soft=[m for m in models if m.valid_probs is not None]
    temps={m.name:fit_temperature(m.valid_probs[idx],y[idx],cfg) for m in soft}
    qs=[temperature_probs(m.valid_probs[idx],temps[m.name]) if cfg['soft']['use_calibrated'] else m.valid_probs[idx] for m in soft]
    w=fit_soft_weights(np.stack(qs),y[idx],cfg) if soft else None
    state={'anchor':anchor,'ranking':ranking,'temperatures':temps,'soft_names':[m.name for m in soft],
           'learned_weights':w,'estimators':{},'fit_indices':np.asarray(idx,dtype=int),'features':[]}
    if any(cfg['meta_models'][kind]['enabled'] for kind in ['logistic','rf']):
        X,features=meta_features(models,'valid',idx,temps,cfg)
        state['features']=features
        for kind in ['logistic','rf']:
            if cfg['meta_models'][kind]['enabled']:
                estimator=build_meta_model(kind,cfg,cfg['seed'])
                estimator.fit(X,y[idx]);state['estimators'][kind]=estimator
    return state
```

### 후보 생성

`recipe_candidates` — 원본 727~788행

```python
def recipe_candidates(models,split,idx,state,cfg):
    by_name={m.name:m for m in models}
    anchor=by_name[state['anchor']]
    ap=getattr(anchor,split+'_pred')[idx]
    raw_anchor=getattr(anchor,split+'_probs')
    aq=None if raw_anchor is None else raw_anchor[idx]
    candidates={'anchor':Candidate('anchor',ap.copy(),None if aq is None else aq.copy(),semantics='actual anchor final probabilities' if aq is not None else 'anchor hard answer')}
    soft=[by_name[n] for n in state['soft_names']]
    soft_probs=[]
    for m in soft:
        p=getattr(m,split+'_probs')[idx]
        soft_probs.append(temperature_probs(p,state['temperatures'][m.name]) if cfg['soft']['use_calibrated'] else p)
    if soft:
        stack=np.stack(soft_probs)
        for name,w in [('soft_mean',np.full(len(soft),1/len(soft))),
                       ('soft_fixed',fixed_weights(state['soft_names'],cfg['soft']['fixed_weights']))]:
            q=np.tensordot(w,stack,axes=(0,0))
            candidates[name]=Candidate(name,q.argmax(1),q,semantics='weighted mean of actual final probabilities; post-calibration if configured')
        if cfg['learned_weights']['enabled']:
            q=np.tensordot(state['learned_weights'],stack,axes=(0,0))
            candidates['soft_learned']=Candidate('soft_learned',q.argmax(1),q,semantics='meta-fit NLL learned probability mean')
    votes=np.stack([getattr(m,split+'_pred')[idx] for m in models])
    counts=vote_counts(votes)
    candidates['hard_vote']=Candidate('hard_vote',resolve_tie(counts,ap),counts=counts,semantics='integer hard vote counts; not soft probabilities')
    if cfg['hard']['fixed_weights']:
        w=fixed_weights([m.name for m in models],cfg['hard']['fixed_weights'])
        counts_w=vote_counts(votes,w)
        candidates['hard_weighted']=Candidate('hard_weighted',resolve_tie(counts_w,ap),counts=counts_w,semantics='weighted hard vote score; not soft probabilities')
    auto_core=[n for n in state['ranking'] if n!=anchor.name or not cfg['core']['exclude_anchor_from_auto']]
    core=cfg['core']['names'] or auto_core[:cfg['core']['auto_size']]
    if set(core)-set(by_name):
        raise ValueError('core 이름은 complete 모델이어야 합니다.')
    groups=[('core_unanimous',core)]
    groups.extend((f'unanimous_{k}',state['ranking'][:k]) for k in cfg['core']['unanimity_sizes'] if len(models)>=k)
    for name,names in groups:
        if len(names)<2:
            continue
        pv=np.stack([getattr(by_name[n],split+'_pred')[idx] for n in names])
        unanimous=(pv==pv[0]).all(0)
        pred=ap.copy();pred[unanimous]=pv[0,unanimous]
        candidates[name]=Candidate(name,pred,counts=vote_counts(pv),semantics='anchor fallback + declared core unanimous hard answer',changed=pred!=ap)
    proposal_name=cfg['guard']['proposal']
    if (proposal_name in candidates and aq is not None and candidates[proposal_name].probability is not None
            and len(models)-1>=cfg['guard']['min_supporters_excluding_anchor']):
        proposal=candidates[proposal_name]
        supporters=np.stack([getattr(m,split+'_pred')[idx] for m in models if m.name!=anchor.name]) if len(models)>1 else np.empty((0,len(idx)),dtype=int)
        for name,with_vote in [('anchor_soft_guard',False),('soft_hard_guard',True)]:
            mask=guard_mask(proposal.prediction,ap,proposal.probability,aq,supporters,votes,cfg,with_vote)
            q=aq.copy();q[mask]=proposal.probability[mask]
            candidates[name]=Candidate(name,q.argmax(1),q,semantics='row-wise routing of actual anchor/proposal probability vectors',changed=mask)
    for kind,estimator in state['estimators'].items():
        X,columns=meta_features(models,split,idx,state['temperatures'],cfg)
        if columns!=state['features']:
            raise AssertionError('메타 feature 순서 변경')
        q=model_probabilities(estimator,X)
        candidates[kind+'_meta']=Candidate(kind+'_meta',q.argmax(1),q,semantics='trained meta classifier predicted probabilities; not VLM logits/probs')
        if aq is not None and len(models)-1>=cfg['guard']['min_supporters_excluding_anchor']:
            supporters=np.stack([getattr(m,split+'_pred')[idx] for m in models if m.name!=anchor.name]) if len(models)>1 else np.empty((0,len(idx)),dtype=int)
            mask=guard_mask(q.argmax(1),ap,q,aq,supporters,votes,cfg,True)
            guarded=aq.copy();guarded[mask]=q[mask]
            candidates[kind+'_guard']=Candidate(kind+'_guard',guarded.argmax(1),guarded,semantics='anchor/meta-model probability routing with vote guard',changed=mask)
    return candidates
```

### OOF·선택

`build_fit_oof` — 원본 792~829행

```python
def build_fit_oof(models,fit,canonical,cfg):
    y,g=canonical['labels'],canonical['groups']
    folds=grouped_folds(y[fit],g[fit],cfg['meta_split']['n_inner_folds'],cfg['seed']+1)
    predictions={};probabilities={};counts={};seen={};semantics={};audit=[]
    for k,(train_pos,valid_pos) in enumerate(folds):
        train=fit[train_pos];valid=fit[valid_pos]
        fold_anchor=choose_anchor(models,train,y,cfg)
        fold_ranking=rank_models(models,train,y)
        state=fit_state(models,train,y,cfg,fold_anchor,fold_ranking)
        result=recipe_candidates(models,'valid',valid,state,cfg)
        for name,item in result.items():
            if name not in predictions:
                predictions[name]=np.full(len(fit),-1,dtype=int)
                seen[name]=np.zeros(len(fit),dtype=int)
                if item.probability is not None:
                    probabilities[name]=np.full((len(fit),4),np.nan)
                if item.counts is not None:
                    counts[name]=np.full((len(fit),4),np.nan)
                semantics[name]=item.semantics
            predictions[name][valid_pos]=item.prediction;seen[name][valid_pos]+=1
            if item.probability is not None and name in probabilities:
                probabilities[name][valid_pos]=item.probability
            if item.counts is not None and name in counts:
                counts[name][valid_pos]=item.counts
        audit.append({'fold':k,'train_ids':[canonical['ids'][i] for i in train],
                      'oof_ids':[canonical['ids'][i] for i in valid],
                      'state':state_summary(state,canonical['ids'])})
    result={};dropped=[]
    for name,pred in predictions.items():
        if not (seen[name]==1).all():
            dropped.append({'name':name,'reason':'fold별 anchor 유형 변화로 모든 fold에서 실행되지 않은 후보'})
            continue
        p=probabilities.get(name)
        if p is not None and not np.isfinite(p).all():
            dropped.append({'name':name,'reason':'OOF probability 유형 불일치'});continue
        result[name]=Candidate(name,pred,p,counts.get(name),semantics[name])
    return result,{'folds':audit,'dropped_candidates':dropped,'scope':'meta-fit only; no holdout used',
                   'base_vlm_weights_refit_per_fold':False}
```

`select_recipe` — 원본 833~872행

```python
def select_recipe(candidates,oof,fit_labels,cfg):
    fallback=cfg['selection']['fallback_recipe']
    if cfg['selection']['policy']=='fixed':
        desired=cfg['selection']['fixed_recipe']
        name=desired if desired in candidates else fallback
        if name not in candidates:
            raise ValueError('fixed/fallback 레시피를 실행할 수 없습니다.')
        return name,{'policy':'fixed','requested':desired,'selected':name,'holdout_used_for_selection':False,
                     'fallback_used':name!=desired,'reason':'설정에서 고정한 규칙; 미지원이면 명시적 anchor fallback'}
    if not oof or 'anchor' not in oof:
        raise ValueError('fit_oof 선택에는 meta-fit 내부 OOF 결과가 필요합니다.')
    allowed=cfg['selection']['candidates'] or list(oof)
    if set(allowed)-set(oof):
        raise ValueError('선택 후보 일부의 OOF가 없습니다: '+str(sorted(set(allowed)-set(oof))))
    records=[]
    a=oof['anchor'].prediction
    for name in allowed:
        if name not in candidates:
            continue
        item=oof[name]
        score=metrics(item.prediction,fit_labels,item.probability)
        rh=rescue_harm(item.prediction,a,fit_labels)
        eligible=True
        if name!='anchor' and rh['net']<cfg['selection']['min_fit_oof_gain_count']:
            eligible=False
        limit=cfg['selection']['max_fit_oof_harm_count']
        if limit is not None and rh['harm']>limit:
            eligible=False
        records.append({'recipe':name,'accuracy':score['accuracy'],'nll':score.get('nll'),**rh,'eligible':eligible})
    available=[r for r in records if r['eligible']]
    if not available:
        name=fallback
    else:
        # 정확도 우선, 동점은 harm/변경 수를 줄인다. holdout으로 선택하지 않는다.
        row=max(available,key=lambda r:(r['accuracy'],-r['harm'],-r['changed']))
        name=row['recipe']
    if name not in candidates:
        raise ValueError('선택한 레시피의 전체 예측이 없습니다.')
    return name,{'policy':'fit_oof','selected':name,'holdout_used_for_selection':False,
                 'fit_oof_tuning_diagnostics':records,'selection_scores_are_not_independent_final_scores':True}
```

### NPY·CSV 출력

`export_candidate` — 원본 997~1018행

```python
def export_candidate(folder,item,split,ids,labels=None,prefix=None,meta_in_sample=False):
    stem=split if prefix is None else prefix+'_'+split
    probs=folder/'probs'
    save_json(probs/f'{stem}_ids.json',ids)
    save_npy(probs/f'{stem}_predictions.npy',item.prediction.astype(np.int64))
    if labels is not None:
        save_npy(probs/f'{stem}_labels.npy',labels.astype(np.int64))
    if item.probability is not None:
        p=np.asarray(item.probability)
        if p.shape!=(len(ids),4) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-7):
            raise AssertionError('저장할 후보 확률 오류')
        if not np.array_equal(p.argmax(1),item.prediction):
            raise AssertionError('확률 argmax와 최종 답이 다릅니다.')
        save_npy(probs/f'{stem}_probs.npy',p.astype(np.float32))
    if item.counts is not None:
        save_npy(probs/f'{stem}_vote_counts.npy',item.counts)
    return {'shape':[len(ids),4] if item.probability is not None else None,
            'class_order':CLASSES,'original_option_order':True,'hard_onehot_used':False,
            'is_smoke':False,'meta_in_sample_present':meta_in_sample,
            'probs_semantics':item.semantics if item.probability is not None else 'not available; hard answer/counts only',
            'kind':'probability' if item.probability is not None else 'hard_answer_only',
            'raw_logits_available':False,'vote_counts_are_soft':False,'recipe':item.name}
```

`make_submission` — 원본 1020~1029행

```python
def make_submission(folder,name,ids,pred):
    if len(ids)!=len(pred) or len(set(ids))!=len(ids) or not np.isin(pred,[0,1,2,3]).all():
        raise ValueError('제출 ID/행 수/정답 오류')
    frame=pd.DataFrame({'id':ids,'answer':[CLASSES[i] for i in pred]})
    path=folder/f'submission_{name}.csv'
    save_csv(path,frame)
    loaded=read_table(path)
    if list(loaded.columns)!=['id','answer'] or loaded.id.tolist()!=ids or not loaded.answer.isin(CLASSES).all():
        raise AssertionError('제출 CSV 저장 후 검증 실패')
    return path
```

### test-only·수동 수정

`run_test_only_recipes` — 원본 1033~1066행

```python
def run_test_only_recipes(models,ids,cfg,folder):
    by_name={m.name:m for m in models};records=[]
    for recipe in cfg['test_only_recipes']:
        name=recipe['name']
        if not name or any(c in name for c in '/\\\n\r'):
            raise ValueError('test-only recipe 이름 오류')
        names=recipe['models']
        if len(set(names))!=len(names) or not names or set(names)-set(by_name):
            raise ValueError('test-only recipe 모델 이름 오류')
        if not any(by_name[n].role=='test_only' for n in names):
            raise ValueError('test_only_recipes는 validation 미보유 모델을 명시하는 별도 경로입니다.')
        anchor_name=recipe.get('anchor',names[0])
        if anchor_name not in names:
            raise ValueError('test-only tie anchor는 등록한 recipe 모델이어야 합니다.')
        w=fixed_weights(names,recipe.get('weights',{}))
        if recipe['method']=='soft_mean':
            if any(by_name[n].test_probs is None for n in names):
                raise ValueError('test-only soft mean에 hard 모델을 넣을 수 없습니다.')
            q=np.tensordot(w,np.stack([by_name[n].test_probs for n in names]),axes=(0,0))
            item=Candidate(name,q.argmax(1),q,semantics='fixed test-only real soft mean; validation unmeasured')
        elif recipe['method']=='hard_vote':
            votes=np.stack([by_name[n].test_pred for n in names]);counts=vote_counts(votes,w)
            item=Candidate(name,resolve_tie(counts,by_name[anchor_name].test_pred),counts=counts,
                           semantics='fixed test-only hard weighted vote scores; validation unmeasured')
        else:
            raise ValueError('test-only method는 soft_mean / hard_vote입니다.')
        target=folder/'test_only'/name
        target.mkdir(parents=True)
        meta=export_candidate(target,item,'test',ids)
        save_json(target/'probs'/'probability_metadata.json',{'test':meta,'validation_available':False})
        path=make_submission(target,'TESTONLY_'+name,ids,item.prediction)
        records.append({'name':name,'submission':str(path),'validation_measured':False,
                        'selected_by_validation':False,'public_private':None,'recipe':recipe})
    return records
```

`manual_changes` — 원본 1068~1097행

```python
def manual_changes(folder,submission,cfg):
    options=cfg['manual_review']
    if not options['enabled']:
        return None
    if not options['changes_csv']:
        raise ValueError('수동 변경 활성화 시 changes_csv가 필요합니다.')
    changes=read_table(options['changes_csv'])
    needed=['id','before','after','reason']
    if set(needed)-set(changes):
        raise ValueError('수동 수정 CSV에는 id,before,after,reason 열이 필요합니다.')
    check_ids(changes.id.tolist(),'manual changes')
    source=read_table(submission).set_index('id',drop=False)
    if set(changes.id)-set(source.index):
        raise ValueError('수동 수정에 test 외 ID가 있습니다.')
    audit=[]
    for row in changes.to_dict('records'):
        if row['before'] not in CLASSES or row['after'] not in CLASSES or row['before']==row['after']:
            raise ValueError('수동 before/after는 다른 a/b/c/d여야 합니다.')
        if source.loc[row['id'],'answer']!=row['before']:
            raise ValueError('수동 before가 선택한 제출 답과 다릅니다.')
        if options['require_reason'] and not row['reason'].strip():
            raise ValueError('수동 변경 근거를 남기세요.')
        source.loc[row['id'],'answer']=row['after'];audit.append(row)
    path=folder/'submission_MANUAL_reviewed.csv'
    save_csv(path,source.reset_index(drop=True))
    save_csv(folder/'analysis'/'manual_changes_applied.csv',pd.DataFrame(audit))
    save_json(folder/'analysis'/'manual_changes_summary.json',{'n_changed':len(audit),'source':str(submission),
              'changes_sha256':hash_file(options['changes_csv']),'submission':str(path),
              'validation_measured':False,'public_private':None,'soft_probabilities_updated':False})
    return str(path)
```

### 전체 실행

`run_ensemble` — 원본 1113~1229행

```python
def run_ensemble(config,mode='inspect'):
    cfg=deepcopy(config);validate_config(cfg)
    if mode not in {'inspect','evaluate','export'}:
        raise ValueError('run mode는 inspect / evaluate / export입니다.')
    if not cfg['models']:
        raise ValueError('완료된 모델 run을 CFG["models"]에 등록하세요.')
    need_test=mode=='export' or (mode=='inspect' and bool(cfg['sample_submission_csv']))
    models=[load_model(s,cfg,need_test) for s in cfg['models']]
    complete=[m for m in models if m.role=='complete']
    canonical=canonical_validation(models,cfg)
    test=canonical_test(models,cfg) if need_test else None
    folder=create_run_dir(cfg,mode)
    save_json(folder/'config.json',cfg);save_json(folder/'environment.json',environment())
    save_json(folder/'inputs.json',{'models':[{**m.audit,'name':m.name,'validation_signature':m.validation_signature,'spec':m.spec} for m in models],
             'validation_csv':cfg['validation_csv'],'validation_csv_sha256':hash_file(cfg['validation_csv']) if cfg['validation_csv'] else None,
             'canonical_validation_signature':canonical['signature'],'test':test,'warnings':canonical['warnings']})
    save_csv(folder/'analysis'/'input_catalog.csv',pd.DataFrame(input_catalog(models)))
    if mode=='inspect':
        info={'mode':mode,'run_dir':str(folder),'n_valid':len(canonical['ids']),
              'n_test':None if test is None else test['n'],'group_column':canonical['group_column'],
              'n_groups':len(np.unique(canonical['groups'])),'warnings':canonical['warnings'],
              'ensemble_fitted':False,'submission_created':False,'models':input_catalog(models)}
        save_json(folder/'execution_status.json',info)
        return info
    y=canonical['labels'];ids=canonical['ids'];n=len(ids)
    fit,hold,split_info=make_meta_split(canonical,cfg)
    save_json(folder/'meta_split.json',split_info)
    anchor=choose_anchor(complete,fit,y,cfg);ranking=rank_models(complete,fit,y)
    state=fit_state(complete,fit,y,cfg,anchor,ranking)
    valid_candidates=recipe_candidates(complete,'valid',np.arange(n),state,cfg)
    need_oof=(cfg['selection']['policy']=='fit_oof' or cfg['selection']['force_inner_oof']
              or cfg['calibration']['mode']!='none' or cfg['learned_weights']['enabled']
              or bool(state['estimators']))
    oof={};oof_audit={}
    if need_oof:
        oof,oof_audit=build_fit_oof(complete,fit,canonical,cfg)
        save_json(folder/'analysis'/'meta_fit_oof_audit.json',oof_audit)
    selected,selection=select_recipe(valid_candidates,oof,y[fit],cfg)
    # 선택을 저장한 뒤 holdout 점수를 계산한다. 선택 함수에 holdout 정답을 주지 않는다.
    selection['config_signature']=digest(cfg);selection['meta_split_signature']=split_info['signature']
    save_json(folder/'analysis'/'selection.json',selection)
    save_json(folder/'fit_parameters.json',state_summary(state,ids))
    anchor_prediction=next(m for m in complete if m.name==anchor).valid_pred
    tables=[evaluation_table(valid_candidates,fit,y,anchor_prediction,'meta_fit_in_sample_diagnostic',cfg),
            evaluation_table(valid_candidates,hold,y,anchor_prediction,'meta_holdout',cfg),
            evaluation_table(valid_candidates,np.arange(n),y,anchor_prediction,'full_validation_mixed_diagnostic',cfg)]
    if oof:
        tables.append(evaluation_table(oof,np.arange(len(fit)),y[fit],oof['anchor'].prediction,'meta_fit_oof_selection_diagnostic',cfg))
    comparison=pd.concat(tables,ignore_index=True)
    save_csv(folder/'analysis'/'comparison.csv',comparison)
    save_csv(folder/'analysis'/'pairwise_full_validation_diagnostic.csv',pairwise_comparison(complete,y))
    save_csv(folder/'analysis'/'validation_consensus_patterns.csv',consensus_patterns(complete,'valid',ids,y))
    base_rows=[]
    for m in complete:
        score=metrics(m.valid_pred,y,m.valid_probs,cfg['calibration']['ece_bins']);score.pop('by_class')
        base_rows.append({'model':m.name,'scope':'common_base_model_validation',**score})
    save_csv(folder/'analysis'/'base_model_validation.csv',pd.DataFrame(base_rows))
    roles=np.full(n,'meta_fit_in_sample_diagnostic',dtype=object);roles[hold]='meta_holdout'
    item=valid_candidates[selected]
    save_csv(folder/'validation_predictions.csv',prediction_frame(ids,item,y,roles))
    metadata={'valid':export_candidate(folder,item,'valid',ids,y,meta_in_sample=bool(need_oof or cfg['anchor']['name'] is None)),
              'class_order':CLASSES,'producer':'VQA_Ensemble_Baseline','raw_logits_fabricated':False}
    save_csv(folder/'analysis'/'selected_holdout_predictions.csv',prediction_frame([ids[i] for i in hold],
          Candidate(selected,item.prediction[hold],None if item.probability is None else item.probability[hold],
                    None if item.counts is None else item.counts[hold],item.semantics),y[hold]))
    if selected in oof:
        metadata['meta_fit_oof_valid']=export_candidate(folder,oof[selected],'valid',[ids[i] for i in fit],y[fit],prefix='meta_fit_oof')
        save_csv(folder/'analysis'/'selected_fit_oof_predictions.csv',prediction_frame([ids[i] for i in fit],oof[selected],y[fit]))
    hold_prob=None if item.probability is None else item.probability[hold]
    heldout=metrics(item.prediction[hold],y[hold],hold_prob,cfg['calibration']['ece_bins'])
    heldout.update(rescue_harm(item.prediction[hold],anchor_prediction[hold],y[hold]))
    summary={'schema':'ensemble_validation_summary_v1','mode':mode,'selected_recipe':selected,'anchor':anchor,
             'n_valid':n,'n_meta_fit':len(fit),'n_meta_holdout':len(hold),'n_test':None,
             'selected_holdout':heldout,'validation_signature':canonical['signature'],
             'base_validation_signature':canonical['base_validation_signature'],
             'accuracy':float((item.prediction==y).mean()),'accuracy_semantics':'full validation mixed diagnostic, not independent evaluation',
             'is_smoke':False,'public_private_measured':False,'holdout_used_for_selection':False,
             'warnings':canonical['warnings'],'selected_probability_semantics':item.semantics,
             'base_vlm_selection_independence_unverified':True,'training_gpu_used':False,
             'run_dir':str(folder),'submission':None}
    if mode=='export':
        test_ids=test['ids'];nt=len(test_ids)
        test_candidates=recipe_candidates(complete,'test',np.arange(nt),state,cfg)
        if set(valid_candidates)!=set(test_candidates):
            raise AssertionError('validation/test 후보 구성이 다릅니다.')
        chosen=test_candidates[selected]
        metadata['test']=export_candidate(folder,chosen,'test',test_ids)
        save_csv(folder/'test_predictions.csv',prediction_frame(test_ids,chosen))
        path=make_submission(folder,'selected_'+selected,test_ids,chosen.prediction)
        summary.update(n_test=nt,submission=str(path))
        if cfg['artifacts']['save_all_candidate_submissions']:
            for name,candidate in test_candidates.items():
                target=folder/'candidates'/name;target.mkdir(parents=True)
                make_submission(target,name,test_ids,candidate.prediction)
                candidate_meta=export_candidate(target,candidate,'test',test_ids)
                save_json(target/'probs'/'probability_metadata.json',{'test':candidate_meta,'public_private_measured':False})
        if cfg['manual_review']['export_queue']:
            save_csv(folder/'analysis'/'test_review_queue.csv',review_queue(complete,test_ids,chosen,anchor,cfg))
        test_only=run_test_only_recipes(models,test_ids,cfg,folder)
        save_json(folder/'analysis'/'test_only_recipes.json',test_only)
        summary['manual_submission']=manual_changes(folder,path,cfg)
    elif cfg['test_only_recipes'] or cfg['manual_review']['enabled']:
        summary['warnings'].append('test-only/manual 경로는 export에서만 실행합니다.')
    save_json(folder/'probs'/'probability_metadata.json',metadata)
    save_json(folder/'validation_summary.json',summary)
    save_json(folder/'execution_status.json',{'mode':mode,'meta_fitted':True,'gpu_training':False,
              'selected_recipe':selected,'validation_evaluated':True,'test_ensembled':mode=='export',
              'public_private_measured':False,'submission':summary['submission']})
    if cfg['artifacts']['save_aligned_inputs']:
        for m in complete:
            save_npy(folder/'aligned_inputs'/f'{m.name}_valid_hard.npy',m.valid_pred)
            if m.valid_probs is not None:
                save_npy(folder/'aligned_inputs'/f'{m.name}_valid_real_probs.npy',m.valid_probs.astype(np.float32))
    if cfg['artifacts']['write_charts']:
        write_report(folder,comparison,summary)
    write_run_readme(folder,summary)
    return summary
```

