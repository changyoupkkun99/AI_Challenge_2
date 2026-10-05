# 0.97229 참고 솔루션 → 재사용 단일 모델 베이스 초기 설정

작성일: 2026-10-04

목표는 앞으로 비슷한 이미지 4지선다 VQA 프로젝트에서 사용할 단일 모델 베이스를 만드는 것이다. 이번에는 A186 팀의 솔루션 글과 ZIP을 대조해 가져올 기능을 선택하고, Gemma 31B 참고 프리셋을 기본값으로 설정했다. 기존 공통·Qwen·Gemma·Agnes 설정은 삭제하지 않고 별도 선택지로 남겼다. OCR/DINO도 선택 기능으로 보존한다.

산출물은 **초기 설정 파일과 통합 설계 문서**다. 모델 학습·추론·NPY 생성은 아직 실행하지 않았다. GPU와 데이터가 확인된 전체 학습 실행기를 제공했다는 뜻은 아니다. `VQA_Baseline_Config.py`는 표준 라이브러리만으로 설정을 출력하거나 JSON으로 저장하며, 모델을 다운로드하지 않는다.

## 1. 실제로 확인한 두 자료

| 자료 | 확인 내용 | 확인 수준 |
|---|---|---|
| 사용자가 붙여 준 Public 0.97229 솔루션 글 | 모델, LoRA, 1토큰 loss, 프롬프트 반복, 외부→내부 학습, 정제·증강, shift 0/2 | 글에 보고된 방법과 성적 |
| `A186_민구없는_민구팀.zip` | `src/train_gemma4_31b.py`, `README.md`, `docs/solution.md`, `data/README.md`, `requirements.txt`, 분할·실험 코드 | 현재 최종 코드의 동작 확인 |

ZIP에서 `docs/solution.md`는 공유 글과 같은 설명이다. 최종 구현은 `src/train_gemma4_31b.py`를 우선했다. 원본 이미지·대회 CSV, 손 정제 CSV, 실제 학습 로그·최종 adapter·제출 CSV·채점 증빙은 ZIP에 없었다. 따라서 **0.97229는 참고 팀이 보고한 Public 성적**이며, 이 조사에서 직접 재학습하거나 제출 파일과 연결해 검증한 성적은 아니다.

| 대조 항목 | 글 | ZIP의 최종 코드/README | 새 베이스 반영 |
|---|---|---|---|
| 기본 모델 | Unsloth Gemma 4 31B 4bit | 동일 모델 ID | 기본 프리셋 |
| LoRA | r16, alpha32, attention+MLP | language=True, attention=True, MLP=True, vision=False | 그대로 |
| loss | 정답 글자 1토큰 | 다음 답 위치의 **전체 어휘** cross-entropy | 그대로. 4클래스 CE와 구분 |
| 어려운 문항의 보기 반전 | 학습 흐름에 소개 | `experiments/run_transfer_v2.py`에 있고 최종 스크립트에는 없음 | 옵션, 기본 꺼짐 |
| Top-2 검토용 교사 | Qwen3.8-27B | README에는 Qwen3.5-27B. 실제 교사 checkpoint/run 없음 | 모델 ID 미확인, 자동 선택 안 함 |
| 정제 작업 | 손 라벨링 | 최종 구현은 `delete`, `edit_option`. 정답 글자 직접 수정 없음 | 작업 종류를 구분해 감사 기록 |
| validation | 글에 상세 분할 없음 | 원본 train 이미지 그룹 기준 약 9:1, 6043/671 ID 목록 | 새 베이스는 사용자 요구대로 **4:1** |
| 학습 선지 순서 | 위치 편향 완화 소개 | 샘플 index 기반 무작위 셔플 | 새 베이스는 학습 정답 위치 균등 배정 |
| 추론 | shift 0,2 평균 | 각 shift softmax→원래 순서 복원→산술 확률 평균 | 그대로 |

**새 베이스의 4:1 분할·위치 균등화·출력 규격은 의도적인 변경이다.** 이 설정으로 A186 결과를 그대로 재현했다고 표현하지 않는다.

## 2. 가져올 기능과 기본값

| 단계 | 가져올 항목 | 초기 선택 | 이유/연결 |
|---|---|---|---|
| 데이터 정제 | 삭제 목록·선지 수정·수정 전 값 검증 | 적용, 출처 필수 | 데이터 품질 개선을 모델 설정과 함께 보존 |
| dev 처리 | `3`, `3:1`, `3:1:1` 투표 패턴 | dev가 있을 때 적용 | `3`은 3표 만장일치, 글의 3:0 표현에 해당 |
| 검토 대상 선별 | 정답이 교사 확률 Top-2 밖인 문항 표시 | 선택, 기본 꺼짐 | 틀린 라벨로 단정하지 않고 사람 검토 |
| 외부 데이터 | MTVQA 한국어 + KRETA | Stage 1 켜짐 | 한국어 이미지 텍스트와 객관식 입력에 적응 |
| 외부 변환 | MTVQA 주관식→4지선다 | 켜짐 | 같은 이미지의 다른 답, 부족하면 유형·길이 유사 답 |
| 프롬프트 | 질문+4보기+출력 지시 2회 반복 | 2회 | 이미지 자체는 한 번만 입력 |
| 모델 | Unsloth Gemma 4 31B bnb4bit | 기본 | 사용자가 지정한 참고 모델 |
| LoRA | attention+MLP, r16/alpha32 | 기본 | language 학습, vision 고정 |
| loss | 다음 답 글자 1토큰만 감독 | 기본 | 설명·EOS·입력 토큰 loss 제외 |
| 내부 학습 | 외부 adapter를 이어서 학습 | Stage 2 켜짐 | 동일 모델/LoRA, 단계별 optimizer 새로 생성 |
| 이미지 증강 | 원본·JPEG·blur·affine | Stage 2에서 4종 | Stage 1·validation·test는 원본 |
| 추론 | Direct Choice Logit, shift 0/2 | 기본 | 답 위치 편향을 줄이는 같은 모델 추론 |
| Softmax | 선지 4개 logits에 온도 1.0 | 기본 | 각 shift 계산 후 확률 평균 |
| 저장 | 실제 logits·probs·labels·IDs·CSV·summary | 필수 | 미래 앙상블과 실행 감사에 사용 |
| OCR/DINO | 원래 hybrid의 보조 정보 | 옵션, 기본 꺼짐 | 현재 모델에 이득인지 동일 validation에서 확인 |
| 고해상도·TTA | 1024 letterbox, 4-way TTA | 옵션, 기본 꺼짐 | 기존 Agnes 경로 보존. 참고 기본과 섞지 않음 |
| 어려운 문항 반전 | abcd→dcba 및 정답 재매핑 | 옵션, 기본 꺼짐 | 참고 최종 코드에 없었던 실험 옵션 |
| 같은 모델 라우팅 | confidence/margin으로 추가 추론 | 옵션, 기본 꺼짐 | 별도 검증으로 채택. 다중 모델 앙상블과 구분 |

각 옵션의 이득은 아직 이 새 베이스에서 측정하지 않았다. 전부 동시에 켜지 않는다. 고정 validation에서 한 가지씩 비교하고 변경 이유와 결과를 남긴다.

## 3. 실제 초기 설정

### 모델·LoRA·loss

| 항목 | 기본값 | 코드 근거 |
|---|---|---|
| model ID | `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` | `MODEL_NAME`, `load_model()` |
| loader | `unsloth.FastVisionModel` | `LOADER`, `load_model()` |
| 양자화 | 모델의 bitsandbytes 4bit, `load_in_4bit=True` | 모델 설정·로드 |
| 계산 dtype | BF16 | autocast 및 Trainer 설정 |
| context | 4096 | `MAX_SEQ_LEN` |
| gradient checkpointing | 켜짐 | `GRAD_CKPT` |
| LoRA | r=16, alpha=32, dropout=0.0, bias=none | `LORA_*`, `get_peft_model()` |
| 학습 범위 | language attention + MLP | `finetune_language_layers`, attention/MLP flags |
| vision | 고정 | `finetune_vision_layers=False` |
| 감독 토큰 | 정답 a/b/c/d 한 토큰 | `LastTokenTrainer.compute_loss()` |
| loss | 마지막 답 예측 위치의 전체 어휘 CE | `last_logits()`→`F.cross_entropy()` |
| padding | left | processor/tokenizer·마지막 위치 정렬 |
| thinking | 비활성 | `enable_thinking=False` |
| 답 경계 | 필요한 경우 빈 thought 블록 추가 | `render()`, `ANSWER_PREFIX` |

학습 loss에서 정답 글자만 감독하는 것과, loss의 분모를 a/b/c/d 4개로 제한하는 것은 다르다. 참고 코드는 **전체 어휘 logits**로 CE를 계산한다. 추론에서만 a/b/c/d의 4개 logits를 선택해 softmax한다.

실행 시 tokenizer에서 네 글자가 각각 단일 토큰인지 확인한다. 채팅 템플릿의 마지막 다음 토큰이 답 글자인지도 확인한다. `ANSWER_PREFIX`를 다른 모델에 그대로 적용하지 않는다. Gemma 프리셋의 중복 prefix 방지와 실제 토큰 확인을 함께 사용한다.

### 두 학습 단계

| 항목 | Stage 1: external | Stage 2: target |
|---|---|---|
| 입력 | MTVQA 한국어 + KRETA | 정제한 train 80% + 선별한 dev |
| epoch | 1 | 1 |
| learning rate | 1e-4 | 1e-4 |
| scheduler | cosine | cosine, 새로 시작 |
| warmup | 0.05 | 0.05 |
| optimizer | adamw_8bit | adamw_8bit, 새로 생성 |
| weight decay / grad clip | 0.01 / 1.0 | 0.01 / 1.0 |
| 이미지 증강 | 없음, 원본 1종 | orig/compression/blur/affine 4종 |
| adapter | 새 LoRA→`adapter_external` | 그 LoRA를 이어 학습→`adapter` |
| checkpoint 선택 | 단계 마지막 | 최종 단계 마지막 |

이미지 4종은 한 원본 문항에 4개의 학습 변형을 만드는 것이다. Stage 2의 1 epoch에서 변형된 전체 학습 샘플을 돈다. 원본만 쓰는 1 epoch와 학습량이 같다고 해석하면 안 된다. validation/test를 증강해 학습에 넣지 않는다.

기본 체크포인트 선택은 참고 코드의 마지막 adapter 방식이다. validation best 선택은 별도 옵션으로 구현할 수 있지만, 바꾸면 실험 조건에 기록한다. 외부 단계 이후 LoRA를 새로 초기화하면 2단계 전이 학습이 사라지므로 같은 adapter를 유지한다.

### 이미지·프롬프트·추론

| 항목 | 값 |
|---|---|
| 이미지 입력 | EXIF 방향 보정→RGB→종횡비 유지 |
| 긴 변 상한 | 2048 |
| Gemma vision token budget | `max_soft_tokens=1120` |
| 기본 padding | 별도 1024 letterbox 사용 안 함 |
| JPEG compression | quality 70 |
| blur | Gaussian radius 1.0 |
| affine | 이동 ±3%, **shear** ±5°, 흰색 빈 영역 |
| 프롬프트 | 한국어 system, 질문·보기·지시 2회, 사이에 빈 줄 |
| 추론 | shift 0, shift 2 |
| logits | 답 위치 a/b/c/d 4개 |
| Softmax | temperature 1.0, shift마다 계산 |
| 최종 확률 | 원래 보기 순서로 복원 후 산술 평균 |

참고 코드의 affine은 기울임(shear)이며 단순한 회전 5°와 동일하지 않다. 또한 Gemma의 vision token budget 1120과 이미지 픽셀 크기 1024는 다른 설정이다. 참고 프리셋은 2048 상한/1120 토큰을 사용하고, 기존 1024+4-way 경로는 선택 실험으로 남긴다.

### GPU·라이브러리

| 프로필 | train batch | grad accumulation | 1 GPU 유효 batch | inference batch | 근거 |
|---|---:|---:|---:|---:|---|
| 기본 `reference_b200` | 16 | 1 | 16 | 32 | 최종 소스의 B200 192GB 기준 주석과 기본값 |
| 선택 `portable` | 4 | 4 | 16 | 4 | 새 베이스 시작값, 실측 필요 |

기본값은 참고 소스 그대로 보존했다. B200에서 실제 성적을 낸 로그는 ZIP에 없으므로 주석만으로 사용 GPU를 확정하지 않는다. A100/다른 GPU에서 메모리가 부족하면 `portable`을 선택하고 먼저 backward와 20-step benchmark를 실행한다. 유효 batch가 같아도 실행 결과가 완전히 같음을 보장하지 않는다. 다중 GPU에서는 world size까지 곱해 유효 batch를 기록한다.

ZIP의 `requirements.txt`에는 Python 3.11/CUDA 12.8, torch 2.11.0, unsloth 2026.9.7, transformers 5.5.0, peft 0.21.0, bitsandbytes 0.50.2가 적혀 있다. **실제 설치 로그·호환성을 확인한 버전 목록은 아니다.** 설치 후 smoke/backward를 통과한 환경을 freeze하고 GPU·CUDA·resolved model/dataset revision을 실행 폴더에 저장한다.

## 4. 4:1 분할과 학습 정답 위치 25%

기본 순서는 다음과 같다.

1. 원본 CSV·이미지·정답 매핑 검사.
2. 사람 정제의 삭제·선지 수정과 변경 전 텍스트 일치 여부 확인.
3. 정답이 있는 원본 train을 이미지 SHA 그룹으로 묶어 정답 분포를 고려한 80:20 분할.
4. validation ID·원본 보기·정답·이미지 hash를 manifest로 고정.
5. dev는 투표 패턴과 정제 결과로 선별하고 **학습에만** 추가.
6. dev·외부 데이터에 validation과 같은 이미지/ID가 있으면 학습에서 제외.
7. 학습용 문항·증강 샘플에 정답 위치를 균등 배정하고 보기와 답을 함께 재매핑.

분할 비율 4:1은 train/validation의 크기이며, 답 위치 25%는 학습 입력의 위치 균형이다. 원본 정답 분포를 층화해 분할하는 것만으로 25%가 되지는 않는다. 새 기본은 보기 내용을 바꾸거나 샘플을 버리지 않고 정답 위치를 옮긴다. 학습 샘플 수가 4의 배수가 아니면 각 위치 개수에 최대 1개 차이를 허용한다.

보기 위치를 직접 참조하는 질문, ‘위 보기’, ‘a와 b 모두’ 같은 문항은 재배치가 안전한지 확인하고 제외 사유를 기록한다. 예외가 있으면 전체 입력이 정확한 25%인지 실제 분포로 보고한다. validation/test는 원본 순서를 보존하고, 추론 시 shift한 결과만 원래 순서로 복원한다.

SHA-256은 같은 파일 내용의 중복을 찾는다. 재압축·crop 등 다른 바이트로 된 같은 이미지까지 막으려면 pHash/추가 검토를 옵션으로 사용한다. 그룹 제약 때문에 80:20이나 클래스 비율이 정확히 맞지 않을 수 있으므로 실제 값과 그룹 목록을 남긴다. 한 프로젝트에서는 manifest를 재사용한다.

현재 기본은 사람 정답 train에서 validation을 만들고 dev pseudo-label은 학습에만 넣는다. 기존 clean7169 manifest처럼 dev도 validation에 포함하는 평가를 사용할 때는 별도 프로필로 명시하고 출처별 성적을 나눠 보고한다. 671·672·1343·1434 평가를 하나의 validation 점수로 비교하지 않는다.

## 5. 정제와 외부 데이터에서 꼭 필요한 것

참고 코드의 내부 데이터 정제를 그대로 적용하려면 `train_changes.csv`, `train_delete_list.csv`, `dev_changes.csv`, `dev_delete_list.csv`가 필요하다. ZIP에는 그 네 파일이 없다. 현재 `manual_csv` 모드에서 경로를 비워 두었으며, 이 자료를 임의로 생성하거나 빈 파일로 대체하지 않았다.

이미 정제된 공통 데이터가 있으면 `precleaned_manifest` 모드로 바꾸고 생성 규칙·정제 감사 기록·데이터 hash를 연결할 수 있다. 그 데이터가 참고 팀의 손 정제 데이터와 같다고 주장하지 않는다. `none` 모드는 정제 없는 비교 실험이다. Top-2 밖 문항은 검토 대상이며 자동 삭제·자동 오답 판정으로 쓰지 않는다.

MTVQA는 한국어 KO/KR에 한정하고 주관식 답을 4지선다로 바꾼다. 같은 이미지의 다른 답을 먼저 오답으로 사용하고, 부족하면 숫자 포함 유형·길이 유사 답에서 채운다. 답의 중복·정답 포함 문자열을 걸러 4개가 안 되면 제외 사유와 수를 기록한다. KRETA는 A/B/C/D 및 정답을 a/b/c/d로 정규화하고 누락·중복 보기·잘못된 정답을 검사한다.

참고 코드 기본은 MTVQA의 train/test와 KRETA의 test를 **학습용 외부 데이터**로 사용한다. 이는 SSAFY test 정답을 학습에 넣는다는 뜻이 아니다. 외부 벤치마크의 원래 split 이름과 실제 학습 사용을 기록하고 같은 외부 샘플에 대한 평가를 별도 성능 증명으로 쓰지 않는다. 외부 데이터 허용 조건은 적용할 대회의 규칙으로 확인한다.

## 6. Direct Choice Logit·복원·Softmax·NPY

원래 보기 순서가 a/b/c/d일 때 shift 2 입력은 c/d/a/b이다. 모델이 출력한 네 위치의 logits·확률은 **입력 위치 기준**이므로 먼저 원래 a/b/c/d 위치로 되돌린다. 그 후 두 shift의 확률을 평균한다.

\[
p^{(s)} = \operatorname{restore}_s(\operatorname{softmax}(z^{(s)} / T)),\qquad
p_{\mathrm{final}} = \frac{p^{(0)} + p^{(2)}}{2},\quad T=1.
\]

학습은 전체 어휘 CE, 추론 확률은 4개 답 토큰에 대한 조건부 정규화다. 이 확률을 이미 검증된 정답 확률/교정된 confidence로 표현하지 않는다. 모델 hard 답을 one-hot으로 바꿔 실제 soft라고 기록하지 않는다.

| 파일 | shape | 의미 |
|---|---|---|
| `probs/valid_logits.npy` | `(N_valid,4)` | **실제로 계산한 shift 0 logits**, 원래 순서로 복원 |
| `probs/valid_probs.npy` | `(N_valid,4)` | shift 0/2 확률 평균인 최종 soft |
| `probs/valid_labels.npy` | `(N_valid,)` | 원래 순서의 정답 0/1/2/3 |
| `probs/valid_ids.json` | N_valid IDs | validation manifest 순서 |
| `probs/test_logits.npy` | `(N_test,4)` | 실제 shift 0 logits |
| `probs/test_probs.npy` | `(N_test,4)` | 최종 soft |
| `probs/test_ids.json` | N_test IDs | 원본 test 순서 |
| `probs/*_shift_logits.npy` | `(K,N,4)` | 실제 추론한 각 shift logits, 원래 순서로 복원 |
| `probs/*_shift_probs.npy` | `(K,N,4)` | 각 shift softmax 결과, 원래 순서로 복원 |
| `probs/*_reference_probs.npy` | `(N,4)` | 기준 shift 0 logits에 대응하는 softmax |

**TTA 평균 확률과 기준 shift logits는 같은 변환 쌍이 아니다.** `softmax(valid_logits)`는 `valid_reference_probs`에 대응하며, `valid_probs`에는 일반적으로 대응하지 않는다. 평균 확률의 log를 모델의 실제 logits라고 저장하지 않는다. 둘 다 필요한 이유와 `logits_reference_shift=0`, shift 순서 `[0,2]`, temperature·평균 방식·모델/adapter hash를 `probability_metadata.json`에 기록한다.

차후 앙상블에는 `*_probs.npy`와 IDs·class order를 함께 읽는다. logit 기반 방법을 비교하려면 각 shift의 실제 logits와 메타데이터를 사용한다. shape·ID 유일성·클래스 순서·유한값·행 확률 합·완료 행 수를 검사하고 test 누락을 조용히 임의 답으로 채우지 않는다.

이번 대회에서 해당 manifest를 선택하면 N_valid=1434, N_test=6714일 수 있다. 새 4:1 분할과 미래 프로젝트에서는 실제 개수를 계산하며 **1434/6714를 상수로 고정하지 않는다.** 현재 파일은 설정을 준비한 것이므로 이 NPY가 이미 생성됐다는 의미는 아니다.

## 7. 각 run에 남길 결과

| 영역 | 파일/결과 |
|---|---|
| 실행 증거 | config, model/dataset resolved revision, 환경 버전, seed, 데이터/분할/정제 hash |
| 학습 자료 | train/external rows, 실제 정답 위치 분포, 출처·증강 종류, 누수 제외 기록 |
| 학습 | loss/LR/step/VRAM, stage별 checkpoints, adapter_external, 최종 adapter |
| 평가 | `validation_predictions.csv`, `validation_summary.json`, 실제 logits/probs/labels/IDs |
| shift 분석 | `analysis/shift_calibration.csv`, `analysis/selected_shift.json` |
| test | 원본 ID 순서 예측, 중간 결과·재개 기록, `submission_{run_id}.csv` |
| 그림 | 학습 전후 비교, loss/LR/VRAM, 클래스·출처별 정확도, 혼동행렬, shift/TTA 비교 |
| 비교 | 동일 validation ID·정답 hash에 대한 run 비교표와 변경 설정 |

기본 recipe의 shift 0/2를 유지하면서 0/1/2/3을 진단할 수 있다. `selected_shift.json`에는 실제 선택이 단일 shift인지 `[0,2]` TTA인지와 선택 사유를 적는다. 점수를 보고 자동으로 여러 설정을 반복 탐색하는 모드는 기본 꺼짐이다. 검증용 calibration 분할 또는 별도 고정 holdout을 쓰면 선택에 쓴 부분과 최종 보고 부분을 구분한다.

차트에는 실제로 계산된 값만 넣는다. 참고 팀의 Public 0.97229, 기존 notebook validation 점수, 새 실험 validation을 한 종류의 성적으로 섞지 않는다. Public/Private은 제출 영수증·스크린샷·파일 hash 등 확인 수준을 기록한다. 목표 0.97을 실측 막대로 그리지 않는다.

## 8. 기존 설정과 선택 기능 보존

| 프로필/옵션 | 남긴 내용 | 제한 |
|---|---|---|
| `reference_gemma31` | 새 기본 Unsloth 4bit, r16/alpha32, 2 stages | 참고 recipe 기반, 데이터 분할 등 변경 |
| `team_common` | Qwen2.5-VL-3B, BF16/none, r16/alpha32, LR2e-4, context4096 | 공통 노트북의 현재 핵심 기본값 |
| `qwen27_only` | Qwen3.8-27B GPTQ4bit, r8/alpha16, LR5e-5, context4096 | GPTQ backward/backend 구현 필요 |
| `gemma31_legacy` | Google Gemma 31B BF16, r8/alpha16, LR5e-5, context2048 | 현재 노트북에 `5e-5,,` 오타 있음. 실행 기록 확인 필요 |
| `agnes_legacy` | Agnes BF16, r8/alpha16, LR5e-5, context2048 | attention+delta target 탐색 연결 필요 |
| OCR / DINO / 둘 다 | hybrid의 confidence·개수 기준과 cache 설계 | 정확한 object query와 backend 실행 연결 필요 |
| 1024 padding / 4-way TTA | 기존 경로를 선택 설정으로 보존 | 적용 모델의 processor budget과 실제 입력 크기 검증 |
| 기존 photometric aug | 기능을 삭제하지 않고 옵션 자리 보존 | hybrid 실제 파라미터 연결은 다음 구현 단계 |

원본 네 업로드 노트북은 수정하지 않았다. 설정 파일 상단에는 이전 `ACTIVE_PRESET` 선택 줄을 주석으로 남겼다. 옛 프로필은 핵심 모델/LoRA/LR 값을 보존하며 새 분할·저장 규격을 사용하는 시작점이다. **기존 점수를 낸 모든 실행 조건을 복원한 프로필이나 전체 실행기라고 주장하지 않는다.** 템플릿·loss·processor·특수 target module·manifest는 모델별 연결이 더 필요하다.

OCR/DINO가 122B에서 손해, 27B에서 이득이었다는 사용자 보고를 모든 모델에 일반화하지 않는다. Gemma 새 기본은 참고 recipe와 비교할 수 있도록 보조 정보를 끈다. OCR만/DINO만/둘 다는 동일 데이터·checkpoint·validation에서 비교하고 train·valid·test에 적용한 경로를 기록한다.

## 9. 설정 사용

```bash
# 새 기본 설정 출력
python VQA_Baseline_Config.py

# JSON으로 고정
python VQA_Baseline_Config.py --output VQA_Default_Config.json

# GPU에 맞춘 배치 시작값 선택
python VQA_Baseline_Config.py --hardware portable --output portable_config.json

# 기존 Qwen 핵심 프로필 + 선택 OCR/DINO
python VQA_Baseline_Config.py --preset qwen27_only --hardware portable --extra-context ocr_dino
```

파이썬에서 `build_config()`로 읽어 새 학습 실행기에 전달한다. 모델 로드·LoRA 함수에 쓸 핵심 인자는 다음처럼 연결한다. 이 예시는 로드 연결이며 전체 학습 실행 코드는 아니다.

```python
from VQA_Baseline_Config import build_config

cfg = build_config("reference_gemma31", "portable")

# 환경변수는 unsloth/torch import 전에 설정
import os
os.environ.update(cfg["model"]["environment_before_import"])
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    cfg["model"]["model_id"],
    load_in_4bit=cfg["model"]["load_in_4bit"],
    max_seq_length=cfg["model"]["max_seq_length"],
    use_gradient_checkpointing=cfg["model"]["gradient_checkpointing"],
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
    lora_dropout=cfg["lora"]["dropout"], bias=cfg["lora"]["bias"],
    random_state=cfg["run"]["seed"],
    use_gradient_checkpointing=cfg["model"]["gradient_checkpointing"],
)
```

실제 실행기는 여기에서 데이터 변환·분할·샘플/콜레이터·답 경계·1토큰 CE·각 단계 Trainer·확률 저장까지 연결해야 한다. 설정의 `None` 경로는 작업할 데이터와 manifest를 확인한 후 채운다. source의 `prepare-only`도 import 단계에서 CUDA를 요구하므로, 새 베이스에서는 CPU 데이터 검증과 GPU 점검을 분리한다.

## 10. 0.97 목표의 판단과 남은 확인

사용자가 지정한 모델·LoRA·loss·epoch에 **데이터 정제, 외부→내부 2단계, 프롬프트 2회, Stage 2의 4종 이미지, shift 0/2 평균**까지 반영하는 것이 현재 근거에 맞는 초기 설정이다. 네 가지 모델 설정만 바꾸고 데이터 정제를 생략하면 참고 방법을 적용한 것과 다르다.

현재 확인이 필요한 것은 다음과 같다.

- 참고 팀의 손 정제 CSV 네 개 및 원본 train/dev/image: 없으면 동일 데이터 재현 불가.
- 실제 제출 CSV·score 증빙·학습 로그·adapter: 없으면 0.97229 실행과 소스 대응 미확인.
- Top-2 검토용 교사의 정확한 model ID/revision·확률 파일: 글/README가 서로 다름.
- 새 프로젝트 데이터 경로·열 구조·이미지 접근·정제 출처: 기본 설정 경로는 미지정.
- 사용할 GPU와 설치한 실제 환경: 기본 B200 주석만으로 적합성을 확정하지 않음.
- 외부 데이터 사용 가능 여부와 revision, validation/test 이미지 중복 여부.
- 이전 모델별 processor/template/GPTQ/Agnes delta 경로 연결 및 원본 실행 로그 확인.

0.97은 목표이며 보장된 초기 성적이 아니다. 특히 사용자 요구대로 분할·학습 위치 균형을 바꾸었으므로 참고 결과와 다른 실험이다. 고정 validation에서 학습 전후, 프롬프트 1/2회, 증강 on/off, shift0/2 및 OCR/DINO를 비교하고, 채택한 결과·설정·실제 soft 파일을 남긴다. 같은 validation에 많은 옵션을 반복 맞추면 평가에도 과적합할 수 있으므로 선택용 부분과 보고용 부분을 구분한다.

## 11. 읽는 순서와 출처

1. 이 문서: 두 자료의 대조와 가져올 기능·기본값.
2. `VQA_Baseline_Checklist.md`: 환경부터 학습·추론·차트·NPY까지 전체 24단계.
3. `VQA_Baseline_Config.py`: 활성 프리셋과 주석으로 남긴 이전 설정.
4. `VQA_Default_Config.json`: 이번 기본 선택을 펼쳐 저장한 설정.
5. A186 ZIP의 `src/train_gemma4_31b.py`: 원래 loader, render, loss, stages, TTA 구현.

코드 확인 위치: `load_change_lists()`/`apply_edits()`, `build_dev_part()`, 외부 데이터 구성부, `load_model()`, `TrainDataset`, `build_messages()`/`render()`, `last_logits()`, `predict_probs()`, `LastTokenTrainer`, `train_stage()`.

외부 링크: [참고 모델](https://huggingface.co/unsloth/gemma-4-31B-it-unsloth-bnb-4bit), [MTVQA](https://huggingface.co/datasets/ByteDance/MTVQA), [KRETA](https://huggingface.co/datasets/tabtoyou/KRETA), [현재 프로젝트 GitHub](https://github.com/changyoupkkun99/AI_Challenge_2).
