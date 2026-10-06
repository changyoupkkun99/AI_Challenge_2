# 단일 모델 VQA 파이프라인 — 슬라이드 해설과 코드 근거

모델편 35장: 본문 31장, 추가 파인튜닝 없는 깡 실행 별첨 4장. 기존 모델·앙상블 노트북을 수정하지 않았습니다.

자료 기준: `VQA_Gemma31B_Reusable_Baseline_v1.ipynb`의 현재 코드. 실제 GPU 학습·추론과 새 Public/Private 성적은 아직 측정하지 않았습니다. 참고 팀 Public 0.97229는 제공받은 솔루션의 보고값입니다. PPT의 합성 이미지와 확률·비교 숫자는 설명용입니다.

원본 학습용 노트북: [VQA_Gemma31B_Reusable_Baseline_v1.ipynb](sandbox:/workspace/scratch/b15306f8f1f9/output/VQA_Gemma31B_Reusable_Baseline_v1.ipynb)

## 공부 순서와 코드 위치

| 단계 | 내용 | PPT 슬라이드 | 주요 함수·설정 |
| --- | --- | --- | --- |
| 01 | 환경·라이브러리 | 4 | 설치 스위치, 환경변수, 실제 버전 기록 |
| 02 | 데이터 파일 준비 | 5 | Drive/ZIP, CSV·이미지 경로 |
| 03 | 프로젝트 설정 | 6 | build_config, RUN_MODE, PRESET, HARDWARE |
| 04 | 정제·출처 | 7–8 | normalize_csv, apply_manual_cleaning, provenance |
| 05 | 4:1 그룹 분할 | 9 | add_group_keys, split_gold, vote_dev, remove_leakage |
| 06 | 이미지 처리 | 10 | load_image, augment_image, permutation_safe |
| 07 | OCR/DINO | 11 | auxiliary_context |
| 08 | 프롬프트 | 12 | messages_for, render_prompt |
| 09 | 모델 로드 | 13 | load_vqa_model, find_lora_targets |
| 10 | LoRA 감사 | 14 | audit_lora |
| 11 | 학습 샘플·loss | 15–16 | training_plan, ChoiceDataset, audit_training_plan |
| 12 | 프로세서·Collator | 17 | call_processor, collate_choices, last_logits |
| 13 | GPU 사전 점검 | 18 | gpu_preflight |
| 14 | 외부·내부 학습 | 19–20 | build_mtvqa_mc, prepare_stages, ChoiceTrainer, train_one_stage |
| 15 | 생성 답 파싱 | 21 | parse_answer, generation_diagnostic |
| 16 | Direct Choice·Softmax | 22 | softmax_numpy, restore_choice_order |
| 17 | Option Permutation | 23 | forward_shift, evaluate_in_memory |
| 18 | 1024/TTA4 선택 진단 | 24 | highres_diagnostic, route_same_model |
| 19 | 학습 전후 평가 | 25 | execute_training |
| 20 | 제출 CSV | 26 | make_submission |
| 21 | NPY·재개 | 27–28 | predict_and_store, shift_analysis |
| 22 | 실제 결과 차트 | 29 | plot_results |
| 23 | run 비교 | 30 | compare_runs |
| 24 | 감사·Top-2 검토 | 31 | export_top2_review, write_run_readme |

## 슬라이드별 상세 해설

### 01. 단일 모델 VQA 파이프라인

이 자료는 재사용 가능한 단일 모델 베이스라인의 24단계를 설명합니다. 기존 모델 노트북의 구현을 근거로 환경 준비, 정제, 분할, LoRA 학습, 추론, 저장과 비교를 연결합니다. 마지막 별첨은 추가 파인튜닝 없이 사전학습 모델을 쓰는 별도 실행 경로입니다. 앙상블 실행은 별도 노트북의 범위입니다. 새 베이스의 실제 GPU 학습과 제출 점수는 아직 확인하지 않았습니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
모델편

### 02. 한 문항으로 보는 VQA

실제 대회 이미지 대신 합성한 영업시간 표지판을 사용합니다. 질문은 영업 시작 시간이고 정답 내용은 09:00입니다. 원래 보기에서는 b입니다. 이후 보기 재배열을 설명할 때 내용 09:00은 그대로이고 표시 글자만 바뀐다는 점을 기억하면 됩니다. CSV에는 정답 글자만 저장하지만 soft 파일에는 a,b,c,d 네 확률을 원래 보기 순서로 저장합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
도입 · 설명용 예시

이 슬라이드의 이미지·수치·ID는 설명용 예시이며 실제 대회 실행 결과가 아닙니다.

### 03. 전체 실행 구조

공통 데이터 정제와 출처 확인 후 Gold의 4:1 manifest를 고정합니다. 본문 경로는 외부·내부 데이터로 같은 LoRA를 이어 학습합니다. 별첨 깡 경로는 사전학습 본체만 불러오고 LoRA 생성과 adapter 로드, 추가 학습을 생략합니다. 두 경로는 이미지와 프롬프트 구성 원칙, 모델별 추론, 확률 순서 원복, 평가와 저장 규격을 공유할 수 있습니다.

현재 노트북의 infer는 학습된 adapter 추론이므로 깡 실행의 진입점으로 사용할 수 없습니다. 깡 loader는 별도로 연결합니다. 앙상블은 모델이 남긴 출력물을 읽는 별도 단계입니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
도입 · 파이프라인 지도

### 04. 환경과 라이브러리

노트북의 설치 셀은 옵션으로 실행합니다. reference 환경의 버전은 A186 ZIP requirements와 주석에서 가져온 값이며 이 조합을 새 베이스에서 실제 GPU 검증한 기록은 없습니다. Unsloth는 Transformers보다 먼저 import하고 환경변수는 import 전에 적용합니다. 실제 실행에서는 Python, CUDA, Torch, GPU명, VRAM, 라이브러리 버전, pip freeze를 남깁니다. 양자화 가중치 메모리와 학습 activation 메모리는 다르므로 B200 주석만으로 다른 GPU의 실행 가능성을 확정하지 않습니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
01 환경 준비
셀 0 · INSTALL 스위치, environment_reference
A186_민구없는_민구팀.zip · requirements.txt, src/train_gemma4_31b.py

### 05. 데이터 파일과 경로

Drive 연결과 ZIP 압축 해제는 선택 사항입니다. 안전한 ZIP 해제는 외부 경로로 빠지는 항목을 차단합니다. CSV 경로와 image_root는 설정에서 실제 파일에 맞게 지정합니다. 자동 발견이 모호하면 명시적으로 입력하고 이미지 파일 존재 여부를 검사합니다. runs 아래에는 config, manifest, 환경, adapter, probs, analysis와 제출 CSV를 저장합니다. 디렉터리 예시는 새 프로젝트에서 사용할 구조이며 원본 대회의 실제 디렉터리를 그대로 단정하는 지도가 아닙니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
02 데이터 준비
셀 1 · 데이터 연결과 안전한 압축 해제
셀 3 · 사용자 경로 설정
셀 4 · resolve_image, run_context

### 06. 기본 설정과 실행 모드

기본 모델 ID는 unsloth/gemma-4-31B-it-unsloth-bnb-4bit입니다. 참고 팀이 공개한 Public 0.97229는 전달된 솔루션 주장으로 표시하며 새 베이스의 결과와 동일시하지 않습니다. 새 베이스는 원본 ZIP의 9:1 대신 Gold 4:1 그룹 분할과 정답 위치 균등화를 추가했습니다. 기존 team_common, qwen27_only, gemma31_legacy, agnes_legacy 선택지는 핵심 설정을 보존한 시작 프로필이며 과거 제출 실행을 완전 재현하는 설정이라고 확정하지 않습니다. 현재 기본 RUN_MODE는 prepare입니다. smoke는 소량 2step 실행이고 full은 전체 학습입니다. infer에는 완성 adapter가 필요합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
03 프로젝트 설정
셀 2 · build_config, PRESETS
셀 3 · RUN_MODE, PRESET, HARDWARE
VQA_Reference_Integration.md · 코드와 공개 글의 차이

### 07. CSV 정규화와 수동 정제

normalize_csv는 id, image path, question, a,b,c,d, answer를 공통 스키마로 바꾸고 a,b,c,d를 0,1,2,3에 연결합니다. 이미지 경로는 실제 파일로 해결하고 SHA256과 출처를 기록합니다. apply_manual_cleaning은 train_changes.csv, dev_changes.csv의 보기 문구 수정과 train_delete_list.csv, dev_delete_list.csv의 삭제를 반영합니다. before 텍스트가 현재 데이터와 일치하는지 검사합니다. 이 기본 경로는 정답 글자를 임의 수정하지 않습니다. 이미 정제된 파일을 쓰면 정제 출처 manifest를 제공해야 합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
04 데이터 정제
셀 4 · normalize_csv, resolve_image, apply_manual_cleaning
셀 2 · cleaning 설정

### 08. Gold와 Dev 투표 라벨

Gold는 정답이 제공된 train 출처입니다. Dev는 사람 또는 모델의 투표 열을 읽어 다수결 pseudo label로 사용합니다. 허용 패턴은 3:0에 해당하는 3, 3:1, 3:1:1입니다. 숫자 투표 1,2,3,4는 a,b,c,d로 바꿉니다. 애매한 패턴은 제외하고 validation에 pseudo label을 섞지 않습니다. train과 dev가 합쳐진 CSV라면 _src 열 또는 경로 접두어로 출처를 명시해야 합니다. 합쳐진 데이터 전체를 확인 없이 Gold라고 선언하면 validation 의미가 바뀝니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
04 출처와 라벨 품질
셀 3 · provenance 설정
셀 5 · vote_dev, remove_leakage

이 슬라이드의 이미지·수치·ID는 설명용 예시이며 실제 대회 실행 결과가 아닙니다.

### 09. 그룹 기준 4:1 데이터 분할

Gold의 image SHA256 그룹을 만들고 StratifiedGroupKFold의 5개 후보 중 크기와 원래 클래스 분포에 가까운 분할을 고릅니다. 같은 이미지 그룹을 쪼개지 않으므로 실제 비율은 정확히 80:20과 다를 수 있습니다. seed, data signature, train IDs, valid IDs를 immutable manifest로 저장하고 비교 실험에서 공유합니다. 정확한 hash는 동일 파일을 찾지만 모든 유사 이미지를 잡지는 못합니다. 선택 pHash도 현재 구현은 같은 pHash를 묶는 방식입니다. validation과 학습 소스의 ID, 경로, hash 중복은 제외하고 train-test 중복은 감사 파일에 기록합니다. 과거 672, 1343, 1434 validation은 동일한 평가 집합이라고 가정하지 않습니다. 4:1 분할은 원래 클래스 비율을 보존하려고 하며 25% 정답 위치는 이후 학습 보기 재배열에서 조정합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
05 분할과 누수 검사
셀 5 · add_group_keys, split_gold, remove_leakage
manifests · train_ids/valid_ids, data signature

### 10. 이미지 처리와 학습 증강

이미지는 EXIF 방향 보정 후 RGB로 바꾸고 기본 native_aspect 경로에서 긴 변 2048 cap을 적용합니다. Gemma 이미지 토큰 예산은 max_soft_tokens=1120입니다. 별도 letterbox는 비율을 유지해 1024 정사각형 안에 넣고 흰색으로 padding합니다. 1024라는 이름만으로 native cap 2048보다 모든 정보가 많다고 단정하지 않습니다. 내부 학습은 원본, JPEG quality70, Gaussian blur radius1.0, affine 이동 ±3%·기울임 ±5°의 네 변형을 사용하고 외부 학습, validation, test는 기본 orig입니다. 증강은 이미지마다 loss 표본 수를 늘리지만 고유 이미지 수가 4배가 되는 것은 아닙니다. 기존 photometric 증강도 별도 선택 기능으로 남깁니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
06 이미지 전처리
셀 6 · load_image, augment_image
셀 2 · image 설정

### 11. OCR와 DINO 선택 분기

AUX_MODE는 none, ocr, dino, ocr_dino 중 하나입니다. EasyOCR는 ko/en, cache confidence .25, prompt confidence .40, 최대15개 텍스트를 사용합니다. Grounding DINO tiny는 object queries로 객체를 찾고 threshold .30, 최대12개를 참고 텍스트로 전달합니다. 캐시는 이미지 hash, 전처리 설정, 도구 revision에 묶습니다. OCR/DINO가 말한 내용을 정답으로 강제하지 않고 이미지와 질문을 유지한 프롬프트에 보조 정보로 넣습니다. 사용자가 보고한 Qwen122B에서의 하락과 27B에서의 이득은 모델별 검증 필요성을 보여주지만 새 베이스의 실측으로 표시하지 않습니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
07 추가 정보 처리
셀 7 · auxiliary_context
셀 8 · messages_for
셀 3 · AUX_MODE

### 12. 프롬프트와 채팅 템플릿

messages_for는 system과 user 메시지를 만듭니다. user에는 이미지 한 번과 반복 텍스트를 넣습니다. 반복 텍스트는 질문, 표시 순서의 네 보기, 소문자 한 글자로 출력하라는 지시를 포함합니다. 추가 OCR/DINO는 반복하지 않고 끝에 한 번 붙입니다. processor.apply_chat_template의 실제 모델 템플릿을 쓰고 기본 Gemma 경로는 enable_thinking=False를 전달합니다. 빈 thought prefix가 이미 있으면 중복으로 넣지 않습니다. Direct Choice Logit에서는 마지막 prompt 토큰의 다음 위치가 실제 정답 글자 위치인지 검증해야 합니다. Prompt Repetition 논문은 반복 아이디어의 출처이며 이 VQA 데이터에서 같은 향상을 보장하는 근거는 아닙니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
08 입력 샘플 구성
셀 8 · messages_for, render_prompt
https://arxiv.org/abs/2512.14982
https://huggingface.co/unsloth/gemma-4-31B-it-unsloth-bnb-4bit

### 13. 모델·프로세서·양자화

기본 backend는 Unsloth FastVisionModel이며 bnb4bit, bfloat16 compute, max_seq_length4096을 사용합니다. resolved revision과 tokenizer choice IDs를 기록합니다. 다른 프로필은 Transformers+PEFT 또는 GPTQ+PEFT 경로이며 GPTQ 학습은 실제 backward가 가능한 kernel이 필요합니다. Auto trainable backend 선택과 legacy patch는 해당 모델·라이브러리 조합에서 검증해야 합니다. 4bit는 가중치 저장 메모리를 줄이며 LoRA와 activation, image token, KV cache는 별도 메모리를 차지합니다. 모델 별 실제 프로세서 입력과 answer prefix를 같다고 가정하지 않습니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
09 모델 로드
셀 9 · load_vqa_model, find_lora_targets
셀 2 · model 설정
https://huggingface.co/unsloth/gemma-4-31B-it-unsloth-bnb-4bit

### 14. LoRA 설정과 학습 대상

기본 LoRA는 r16, alpha32, dropout0, bias none입니다. finetune_vision_layers=False, language=True, attention=True, mlp=True로 생성합니다. LoRA의 delta는 alpha/r 배율을 곱한 BA입니다. A,B가 학습되고 기존 W는 고정되므로 전체31B를 모두 업데이트하는 full fine-tuning과 다릅니다. audit_lora는 실제 trainable parameter 이름과 수를 기록하고 vision parameter가 의도와 달리 학습되는지 점검합니다. 설정 플래그만 읽어 학습 대상을 확정하지 않고 실제 resolved module과 requires_grad를 확인합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
10 LoRA 검증
셀 9 · get_peft_model 호출
셀 10 · audit_lora
https://unsloth.ai/docs/basics/vision-fine-tuning

### 15. 학습 정답 위치 25% 균등화

Gold 4:1 그룹 분할의 계층화는 원래 클래스 분포를 가급적 유지합니다. 학습의 정답 위치 a,b,c,d 각 25% 목표는 별도 training_plan의 보기 재배열로 구현합니다. 정답 내용 09:00을 b에서 d로 옮기면 target index도 1에서 3으로 바꿉니다.

현재 permutation_safe는 보기 a, a와 b, 첫 번째 보기 같은 일부 위치 참조 표현을 정규식으로 탐지하고 재배열에서 제외합니다. 모든 의미적 위치 참조를 탐지하는 검사는 아니므로 예외를 추가 검토해야 합니다. 제외 문항과 전체 표본 수 때문에 실제 균형이 완벽히 25%가 아닐 수 있습니다. validation과 test는 원래 보기 순서로 평가하고, TTA 결과도 원래 순서로 복원합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
11 학습 샘플 계획
셀 6 · permutation_safe
셀 11 · training_plan, ChoiceDataset, audit_training_plan

이 슬라이드의 이미지·수치·ID는 설명용 예시이며 실제 대회 실행 결과가 아닙니다.

### 16. 정답 글자 1토큰 loss

현재 ChoiceTrainer는 prompt 끝의 full vocabulary logits를 얻고 정답 글자의 token ID를 target으로 CrossEntropy를 계산합니다. 이는 전체 vocabulary에서 정답 token의 확률을 높이는 loss입니다. 추론에서 a,b,c,d만 골라4개 Softmax를 계산하는 것과 구분합니다. 질문, 이미지 토큰, 반복 프롬프트, reasoning, EOS와 padding은 target loss를 받지 않습니다. 각 답이 실제 tokenizer에서 정확히1개 token인지 assert하고 실제 답 경계의 generation 진단을 확인합니다. 정답 글자가2토큰이거나 thought prefix가 잘못 맞으면 이 설계는 그대로 사용할 수 없습니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
11 학습 목표
셀 9 · choice token 인코딩 검사
셀 12 · last_logits
셀 16 · ChoiceTrainer.compute_loss

### 17. 프로세서와 배치 Collator

collate_choices는 row마다 이미지 변형과 보기 순서를 적용하고 render_prompt로 만든 문자열을 processor에 전달합니다. 모델 프로세서가 요구하는 flat 또는 nested images 형태를 확인합니다. tokenizer padding_side는 left이며 full vocabulary 출력의 마지막 prompt 위치가 마지막 유효 token이 되게 합니다. input_ids, attention_mask, pixel_values 등은 모델에 전달하고 target_ids는 별도로 loss에 사용합니다. max context를 넘긴 입력을 조용히 자르지 않고 실패시켜 이미지와 정답 경계가 잘리지 않게 합니다. left padding만으로 모든 모델의 position 처리 안전성을 단정하지 않고 preflight로 검사합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
12 학습 입력 텐서
셀 12 · call_processor, collate_choices, to_device, last_logits

### 18. GPU 사전 점검과 VRAM

gpu_preflight는 실제 모델, 실제 샘플, 실제 processor로 forward와 backward를 수행합니다. loss와 gradient가 유한한지, LoRA gradient가 생기는지, token 경계가 맞는지, peak VRAM을 기록합니다. 점검은 optimizer.step을 하지 않습니다. reference_b200 배치는 A186 소스의 B200192GB 주석과 기본값16×1/eval32를 가져온 것이며 실제 실행 로그는 제공받지 못했습니다. portable4×4/eval4는 시작값이고 모든GPU에서 통과를 보장하지 않습니다. 선택 benchmark20steps는 adapter와 RNG를 복구하도록 구현되어 있습니다. GPU 점검 실패를 prepare 성공으로 대체하지 않습니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
13 실제 학습 전 검증
셀 13 · gpu_preflight
셀 16 · benchmark_training
셀 2 · HARDWARE_PROFILES

### 19. 외부 VQA 데이터의 4지선다 변환

Stage1은 ByteDance/MTVQA 한국어 부분과 tabtoyou/KRETA를 사용합니다. MTVQA는 자유 답변을4지선다로 바꾸며 같은 이미지의 다른 답을 우선 오답으로 쓰고 부족하면 숫자 여부와 길이가 유사한 답 풀에서 채웁니다. 중복과 gold substring 포함 오답을 제외합니다. KRETA는 원래 객관식 A,B,C,D를 공통 a,b,c,d로 정규화합니다. 현재 외부 dataset의 train/test라는 split 이름은 외부 벤치마크의 이름이며 대회 test.csv와 다릅니다. 외부 split을 학습에 사용했다는 사실과 revision을 기록하고 해당 프로젝트의 외부 데이터 허용 범위에 맞춰 설정합니다. validation 이미지와 겹치는 외부 학습 문항은 제외합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
14 Stage 1 데이터
셀 15 · load_kreta, load_mtvqa_qa, build_mtvqa_mc
셀 16 · prepare_stages
A186 src/train_gemma4_31b.py

이 슬라이드의 이미지·수치·ID는 설명용 예시이며 실제 대회 실행 결과가 아닙니다.

### 20. 외부·내부 2단계 학습

Stage1 external1epoch LR1e-4 후 같은 LoRA 가중치를 Stage2 target1epoch LR1e-4로 이어 학습합니다. Stage1은 orig만 쓰고 Stage2는 원본과3종 증강을 사용합니다. optimizer와 scheduler 상태는 단계마다 새로 시작합니다. 기본 AdamW8bit, cosine, warmup ratio.05, weight decay.01, grad clip1입니다. save_steps200/save_total_limit2로 주기 체크포인트를 저장하고 기본 선택은 last_stage_final입니다. best_retained_checkpoint 옵션은 남겨진 마지막2개와 최종 상태를 평가하는 것이며 전체 모든 과거 체크포인트 중 최적을 보장하지 않습니다. 여러 후보를 validation으로 선택하면 selection bias를 기록합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
14 학습과 체크포인트
셀 16 · prepare_stages, training_arguments, train_one_stage
셀 21 · execute_training

### 21. 생성 답 파싱과 진단

기본 추론은 Direct Choice Logit이지만 generation_diagnostic은 소량8개를 generate하여 토큰 경계와 답 형식을 확인합니다. parse_answer는 한 글자 또는 정답: b 같은 짧은 형식만 허용합니다. 정답은 b입니다처럼 긴 설명이나 여러 답은 invalid로 기록합니다. 파싱 실패를 임의 정답으로 채우거나 one-hot을 실제 soft라고 저장하지 않습니다. 모델이 빈 thought block, 설명, 공백 또는 답 이외 문자를 생성한다면 프롬프트/템플릿/answer prefix를 점검합니다. parser의 성공이 Direct Choice Logit 경계의 정확함을 자동 입증하는 것은 아니므로 token ID와 실제 generation을 대조합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
15 Generate 진단
셀 17 · parse_answer, generation_diagnostic

### 22. Direct Choice Logit과 Softmax

forward 결과의 마지막 prompt 위치에서 tokenizer choice token IDs의4개 logits를 추출합니다. Softmax는 temperature T로 나누고 최대값을 빼서 exp 계산의 안정성을 확보합니다. 기본 T1입니다. 그림의 logits [0,2,0,0]는 설명용 숫자로 확률은 약[.0963,.7112,.0963,.0963]이고 argmax는 b입니다. 이것은 a,b,c,d 사이로 조건부 정규화한 확률입니다. 모델 전체 vocabulary의 모든 답문자 외 토큰 확률까지 합친 값이나 자동으로 calibration된 정답 신뢰도로 해석하지 않습니다. valid/test 파일에는 이런 실제soft를 저장합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
16 확률 계산
셀 12 · last_logits
셀 18 · softmax_numpy
셀 19 · forward_shift

이 슬라이드의 이미지·수치·ID는 설명용 예시이며 실제 대회 실행 결과가 아닙니다.

### 23. 보기 순환 이동과 확률 원복

shift0의 order는[0,1,2,3]이고 shift2는[2,3,0,1]입니다. 따라서 원래a,b,c,d의 내용은 표시c,d,a,b에 대응합니다. 원래b인09:00은 shift2에서 표시d입니다. 표시 확률[.08,.07,.05,.80]은 원래순서로[.05,.80,.08,.07]에 복원합니다. shift0 원복확률[.10,.70,.10,.10]과 평균하면[.075,.75,.09,.085]가 됩니다. 기본은 shift0와2의 확률 산술평균이며 logits 평균과 다릅니다. 위치를 참조하는 unsafe 문항은 재배열하지 않고 실제 적용 순서를 기록합니다. OOM은 batch를 줄여 재시도하고 최소batch에서도 실패하면 오류를 기록하며 다른 오류를 정답으로 덮지 않습니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
17 Option Permutation 검증
셀 18 · restore_choice_order
셀 19 · forward_shift, evaluate_in_memory
셀 23 · predict_and_store

이 슬라이드의 이미지·수치·ID는 설명용 예시이며 실제 대회 실행 결과가 아닙니다.

### 24. 1024 입력과 4-way TTA

기본 학습과 추론은 native aspect cap2048/maxsoft1120이고 shift0,2입니다. RUN_HIGHRES_TTA4=True는1024 letterbox와shift0,1,2,3을 적용하는 별도 validation 진단입니다. 1024는 모드 이름이며 기본2048cap보다 언제나 고해상도라는 뜻은 아닙니다. same_model_routing은 confidence .80 또는 margin .20 미만 문항에 shift1,3을 추가하는 선택 경로이며 기본off입니다. confidence/margin 기준은 validation으로 검증하고 Public/Private로 문항별 정답을 역추정해 맞추지 않습니다. 단일 모델의 여러 입력을 평균하는 TTA와 서로 다른 모델의 앙상블은 구분합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
18 선택 실험
셀 3 · RUN_HIGHRES_TTA4, routing
셀 20 · highres_diagnostic, route_same_model
셀 23 · final probs 저장

### 25. 학습 전후 validation 실행

execute_training은 학습 전에 같은 validation의 확률을 저장하고, 준비된 외부·내부 Stage를 순서대로 학습합니다. smoke는 일부 데이터와 2 step으로 연결을 점검하는 실행입니다. full은 전체 샘플과 epoch 설정을 사용합니다. infer는 완성된 adapter를 로드한 뒤 학습을 생략합니다.

학습 전후 비교에는 같은 ID, Gold 정답, 질문, 보기와 이미지가 필요합니다. 새 LoRA의 초기 상태를 before로 평가한다면 실제 초기화도 기록합니다. 현재 새 베이스의 GPU 학습·추론과 제출 점수는 측정하지 않았습니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
19 학습 실행과 평가
셀 21 · execute_training
셀 3 · RUN_MODE
셀 9 · infer adapter 필수 조건

### 26. Test 추론과 제출 CSV

make_submission은 final probs의 argmax를 a,b,c,d로 바꾸고 sample submission의 ID 집합과 순서에 맞춥니다. 누락·중복·추가 ID를 검사하고 id,answer 두 열만 저장합니다. 슬라이드의 세 ID는 설명용입니다.

과거 대회의 test 6,714개는 해당 데이터의 값입니다. 재사용 베이스는 입력에서 실제 개수를 확인합니다. smoke 출력은 일부 행만 포함하므로 SMOKE 파일로 표시합니다. 후속 앙상블을 위해 CSV와 별도로 soft NPY, ID와 metadata도 보존합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
20 제출 파일
셀 22 · make_submission
셀 23 · predict_and_store

이 슬라이드의 이미지·수치·ID는 설명용 예시이며 실제 대회 실행 결과가 아닙니다.

### 27. NPY와 ID 저장 규격

valid와 test 각각 logits, probs, ID를 저장하고 valid에는 Gold label index도 저장합니다. logits와 probs는 (N,4), labels는 (N,)이며 ID JSON의 순서가 NPY의 행 순서입니다. 클래스 순서는 원래 [a,b,c,d]입니다.

과거 공통 validation 1,434개와 test 6,714개를 사용한 실행에서는 그 크기가 나오지만 새 데이터에서는 manifest의 개수로 결정합니다. shift별 logits와 probs는 (K,N,4)로 저장합니다. hard 제출 답의 one-hot은 실제 soft 확률과 구분합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
21 확률 파일 저장
셀 23 · predict_and_store
셀 2 · artifacts 설정

### 28. logits·TTA 평균과 재개 기록

logits.npy는 기준 shift 0의 실제 원복 logits이고 reference_probs.npy는 여기에 Softmax를 적용한 결과입니다. 최종 probs.npy는 shift별 원복 확률의 산술평균이며 선택 routing 결과를 반영할 수 있습니다. 따라서 softmax(logits.npy)가 final probs.npy와 일반적으로 일치하지 않습니다. 평균 확률의 log를 raw logits처럼 저장하지 않습니다.

probability_metadata.json에는 클래스 순서, shape, temperature, shift, 기준 shift, 의미, adapter hash와 hard_onehot=False를 기록합니다. 미완성 memmap 행은 NaN이며 config·data·adapter·ID·shift signature가 같을 때만 재개합니다. shift_calibration과 selected_shift는 validation 진단 및 고정 레시피 기록입니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
21 파일 의미와 재현성
셀 23 · predict_and_store, shift_analysis
probs/probability_metadata.json
analysis/shift_calibration.csv, selected_shift.json

### 29. 실제 실행 결과 차트

plot_results는 실제 logs/*_history.csv에서 step별 loss, learning rate, grad norm을 그립니다. training_summary.json의 stage별 peak allocated VRAM도 별도 차트로 만듭니다. 같은 validation의 학습 전후 정확도, 클래스별 정확도, confusion matrix, 출처별 정확도와 shift/TTA 비교를 HTML로 저장합니다.

실행 로그가 없으면 실제 학습 그래프나 점수를 채워 넣지 않습니다. 이 PPT의 확률과 rescue/harm 숫자는 설명용입니다. 참고 팀 Public 0.97229는 제공받은 솔루션의 보고값이며 현재 새 베이스의 검증된 성적이 아닙니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
22 결과 시각화
셀 24 · plot_results
logs/*_history.csv, training_summary.json, validation_summary.json
VQA_Reference_Integration.md

### 30. 실행 결과 비교와 rescue / harm

compare_runs는 validation_signature가 같은 실행만 직접 비교합니다. signature에는 ID, Gold 정답, 질문, 보기와 이미지 hash가 들어갑니다. 기준 run이 틀리고 새 run이 맞힌 문항은 rescue, 기준 run이 맞고 새 run이 틀린 문항은 harm이며 net=rescue-harm입니다.

설명용 100문항에서 기준은 90개를 맞히고 새 run은 92개를 맞힙니다. rescue 5개, harm 3개라 net은 +2입니다. 과거 validation 672·1343·1434개는 같은 집합이라고 가정하지 않습니다. Public/Private는 별도 테스트 분할 결과입니다. 같은 validation으로 많은 설정을 선택한 성적에는 선택 편향이 생길 수 있습니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
23 같은 평가 집합에서 비교
셀 25 · compare_runs
validation_summary.json · validation_signature

이 슬라이드의 이미지·수치·ID는 설명용 예시이며 실제 대회 실행 결과가 아닙니다.

### 31. 실행 감사와 Top-2 검토

export_top2_review는 Gold 답이 모델 확률 상위 두 개에 없는 문항을 수동 검토 CSV로 내보냅니다. 이 조건만으로 라벨 오류를 확정하지 않습니다. 사람이 이미지와 질문·보기를 확인하고 수정·삭제·유지 근거를 별도 기록합니다.

run README와 감사 파일은 config, 환경, model revision, 정제 이력, split manifest, adapter hash, checkpoint 선택, 실제 평가와 출력 규격을 연결합니다. prepare로 파일과 설정을 확인하고 smoke에서 사전 forward/backward와 저장 연결을 점검한 뒤 full을 수행합니다. 완성된 valid/test soft, ID, label, metadata는 이후 별도 앙상블 노트북에 전달합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
24 재현 기록과 수동 검토
셀 26 · export_top2_review, write_run_readme
VQA_Baseline_Checklist.md · 24단계 대응

### 32. 별첨: 깡 모델 실행 경로

깡 실행은 이번 프로젝트의 추가 파인튜닝 없이 사전학습된 instruction 모델을 사용하는 경로입니다. LoRA를 생성하거나 대회 adapter를 로드하지 않고 Trainer와 backward도 실행하지 않습니다. 모델 자체가 이전에 사전학습·instruction tuning된 사실과 이번 프로젝트에서 추가 학습하는지는 구분합니다.

이미지 처리, 프롬프트, OCR/DINO와 TTA는 추론 입력 옵션입니다. 현재 노트북의 infer는 완성 adapter를 요구하므로 깡 실행에 그대로 사용할 수 없습니다. 별도 base-only loader 진입점을 만들고 공통 입력·평가·저장 함수를 연결합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
별첨 A · 추가 파인튜닝 없는 추론
셀 9 · load_vqa_model의 infer 분기
셀 21 · execute_training의 infer 분기

### 33. 깡 실행: 본체만 로드

이 loader는 Unsloth 공식 FastVision 사용법과 현재 기본 모델 ID를 연결한 설명용 코드입니다. GPU 실행은 확인하지 않았습니다. from_pretrained로 본체와 processor를 얻고 for_inference와 eval로 추론 상태를 설정합니다. get_peft_model, load_adapter, Trainer와 backward는 호출하지 않습니다.

실제 환경과 resolved revision을 기록하고 답 token ID, 채팅 템플릿, 이미지 입력과 답 경계를 본문과 같은 방법으로 확인합니다. 4bit는 로드 메모리 설정이며 추가 학습 여부를 뜻하지 않습니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
별첨 B · 모델 로드 예시
https://unsloth.ai/docs/basics/vision-fine-tuning
https://huggingface.co/unsloth/gemma-4-31B-it-unsloth-bnb-4bit
현재 노트북 셀 9의 from_pretrained 로드 패턴

### 34. 깡 실행: 프롬프트와 선택 확률

이 코드는 inputs와 choice_token_ids가 올바르게 준비됐다는 전제의 추론 예시입니다. 이미지·텍스트 형식을 모델 processor에 맞추고, a,b,c,d가 각 1 token이며 마지막 prompt 위치 다음이 실제 답 위치인지 확인합니다. Left padding 배치에서 마지막 위치의 full vocabulary logits를 읽고 네 답 token만 뽑아 Softmax를 적용합니다.

reasoning 모델이 다음 token에서 답 글자를 내지 않으면 적절한 answer prefix와 경계를 검증하거나 generate와 parser로 별도 평가합니다. 생성 hard 답만 있으면 hard로 저장합니다. 이를 one-hot으로 변환해 실제 soft라고 표시하지 않습니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
별첨 C · 추론 예시
셀 8 · render_prompt
셀 12 · call_processor, last_logits
셀 18 · softmax_numpy
https://unsloth.ai/docs/basics/vision-fine-tuning

### 35. 깡 실행: 옵션과 결과 저장

깡 실행도 같은 Gold validation manifest로 평가합니다. 기본 이미지와 프롬프트를 먼저 실행하고 OCR/DINO, 반복, 1024/TTA4를 별도 실험으로 비교합니다. TTA에서는 표시 보기 확률을 원래 순서로 복원한 뒤 평균합니다.

실제 Direct Choice 확률을 얻었다면 NPY로 보존하고 hard 답만 있다면 hard라는 성격을 기록합니다. metadata에는 base-only 실행, model revision과 adapter 없음 상태를 남깁니다. 모델 결과를 만들고 저장하는 단계에서 끝내며 서로 다른 모델의 후속 앙상블은 별도 노트북에서 진행합니다.

코드 근거
VQA_Gemma31B_Reusable_Baseline_v1.ipynb
별첨 D · 동일 평가와 저장 규격
셀 18/19 · 확률 원복과 추론
셀 22/23 · 공통 저장 규격
VQA_Ensemble_README.md · 모델 실행과 별도 사용

## 현재 노트북의 핵심 코드 발췌

아래 함수는 현재 노트북에서 그대로 추출했습니다. `CFG`, `processor`, `model`, `CHOICES` 등의 원본 노트북 전역 상태를 사용하므로 이 문서 자체가 독립 실행 스크립트는 아닙니다. 전체 실행은 학습용 노트북에서 합니다.

### messages_for — code cell 09 (1부터 계산)

```python
def messages_for(row, image, order):
    body = str(row["question"])+"\n"+"\n".join(f"({c}) {row[CHOICES[j]]}" for c,j in zip(CHOICES,order))
    body += "\n\n"+CFG["prompt"]["instruction"]
    repeated = CFG["prompt"]["repeat_separator"].join([body]*CFG["prompt"]["repeat"])
    aux = auxiliary_context(row)
    if aux:
        repeated += "\n\n이미지가 가장 중요한 근거입니다. 아래 참고 정보가 틀리거나 이미지와 충돌하면 무시하세요.\n"+aux
    return [{"role":"system","content":CFG["prompt"]["system"]},
            {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":repeated}]}]
```

### render_prompt — code cell 09 (1부터 계산)

```python
def render_prompt(row,image,order):
    text = processor.apply_chat_template(messages_for(row,image,order), tokenize=False,
                add_generation_prompt=True,**CFG["prompt"]["template_kwargs"])
    prefix=CFG["prompt"]["answer_prefix"]
    if prefix and not text.endswith(prefix):
        text += prefix
    return text
```

### collate_choices — code cell 13 (1부터 계산)

```python
def collate_choices(features):
    encoded=call_processor([f["text"] for f in features],[f["image"] for f in features])
    encoded["target_ids"]=torch.tensor([f["target"] for f in features],dtype=torch.long)
    return encoded
```

### last_logits — code cell 13 (1부터 계산)

```python
def last_logits(batch,current_model=None):
    global _logits_to_keep
    current_model=current_model if current_model is not None else model
    if _logits_to_keep is not False:
        try:
            output=current_model(**batch,logits_to_keep=1,use_cache=False)
            _logits_to_keep=True
            return output.logits[:,-1,:]
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            _logits_to_keep=False
    return current_model(**batch,use_cache=False).logits[:,-1,:]
```

### parse_answer — code cell 18 (1부터 계산)

```python
def parse_answer(text):
    text=str(text).strip()
    if re.fullmatch(r"[abcd]",text,re.I):
        return text.lower()
    match=re.fullmatch(r"(?:정답\s*[:：]?\s*)?[\(\[]?([abcd])[\)\].]?",text,re.I)
    return match.group(1).lower() if match else None
```

### softmax_numpy — code cell 19 (1부터 계산)

```python
def softmax_numpy(logits,temperature=1.0):
    if temperature<=0:
        raise ValueError("temperature는 양수여야 합니다.")
    z=np.asarray(logits,dtype=np.float32)/temperature
    z=z-z.max(axis=-1,keepdims=True)
    e=np.exp(z)
    return e/e.sum(axis=-1,keepdims=True)
```

### restore_choice_order — code cell 19 (1부터 계산)

```python
def restore_choice_order(shown,order):
    restored=np.empty(4,dtype=np.float32)
    restored[np.asarray(order,dtype=np.int64)]=shown
    return restored
```

### check_probabilities — code cell 19 (1부터 계산)

```python
def check_probabilities(probs,n):
    p=np.asarray(probs)
    if p.shape!=(n,4) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1,atol=1e-5):
        raise ValueError(f"확률 shape/유한값/합 검증 실패: {p.shape}")
```

### forward_shift — code cell 20 (1부터 계산)

```python
def forward_shift(rows,shift,image_policy=None):
    orders=[[(j+shift)%4 for j in range(4)] if permutation_safe(r) else list(range(4)) for r in rows]
    images=[load_image(r["_img"],image_policy) for r in rows]
    texts=[render_prompt(r,img,order) for r,img,order in zip(rows,images,orders)]
    batch=to_device(call_processor(texts,images))
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        shown=last_logits(batch)[:,CHOICE_IDS].float().cpu().numpy()
    restored=np.stack([restore_choice_order(z,order) for z,order in zip(shown,orders)])
    return restored,softmax_numpy(restored,CFG["inference"]["temperature"])
```

### evaluate_in_memory — code cell 20 (1부터 계산)

```python
def evaluate_in_memory(frame,shifts,image_policy=None):
    set_model_mode(False)
    logits=np.empty((len(shifts),len(frame),4),dtype=np.float32)
    probs=np.empty_like(logits)
    size=CFG["training"]["per_device_eval_batch_size"]
    cursor=0
    while cursor<len(frame):
        end=min(cursor+size,len(frame))
        rows=frame.iloc[cursor:end].to_dict("records")
        try:
            for k,shift in enumerate(shifts):
                z,p=forward_shift(rows,shift,image_policy)
                logits[k,cursor:end]=z
                probs[k,cursor:end]=p
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if size==1:
                raise
            size=max(1,size//2)
            continue
        cursor=end
    return logits,probs
```

### route_same_model — code cell 21 (1부터 계산)

```python
def route_same_model(frame,base_probs,shifts):
    setting=CFG["inference"]["same_model_routing"]
    if not setting["enabled"]:
        return base_probs,None
    values=np.sort(base_probs,axis=1)
    flags=np.flatnonzero((values[:,-1]<setting["confidence_threshold"]) |
                         ((values[:,-1]-values[:,-2])<setting["margin_threshold"]))
    extra=[s for s in setting["extra_shifts"] if s not in shifts]
    if not extra:
        raise ValueError("routing 추가 shift가 기본 shift와 겹칩니다.")
    out=base_probs.copy()
    if len(flags):
        z,p=evaluate_in_memory(frame.iloc[flags],extra)
        out[flags]=(base_probs[flags]*len(shifts)+p.sum(0))/(len(shifts)+len(extra))
    else:
        z=np.empty((len(extra),0,4),dtype=np.float32)
        p=z.copy()
    detail={"indices":flags,"ids":frame.iloc[flags].id.tolist(),"shifts":extra,"logits":z,"probs":p}
    return out,detail
```

### highres_diagnostic — code cell 21 (1부터 계산)

```python
def highres_diagnostic():
    policy=deepcopy(CFG["image"])
    policy["resize_mode"]="letterbox"
    policy["letterbox"].update(enabled=True,side=1024)
    shifts=[0,1,2,3]
    logits,probs=evaluate_in_memory(valid_df,shifts,policy)
    labels=valid_df.answer.map(CHOICES.index).to_numpy()
    score=float((probs.mean(0).argmax(1)==labels).mean())
    for name,array in [("logits",logits),("probs",probs)]:
        save_npy(RUN_DIR/"analysis"/f"valid_highres1024_shift_{name}.npy",array)
    save_json(RUN_DIR/"analysis"/"highres1024_tta4.json",{"accuracy":score,
        "shifts":shifts,"image_policy":policy,"ids":valid_df.id.tolist(),
        "comparison_scope":"same validation; separate diagnostic; not automatically applied to test"})
    return score
```

### make_submission — code cell 23 (1부터 계산)

```python
def make_submission(frame,probs):
    check_probabilities(probs,len(frame))
    result=pd.DataFrame({"id":frame.id.tolist(),"answer":[CHOICES[i] for i in probs.argmax(1)]})
    sample=CFG["data"]["sample_submission_csv"]
    if sample and RUN_MODE!="smoke":
        template=pd.read_csv(sample,dtype=str,keep_default_na=False)
        if "id" not in template or template.id.duplicated().any() or set(template.id)!=set(result.id):
            raise ValueError("test와 sample_submission ID 집합이 다릅니다.")
        result=result.set_index("id").loc[template.id].reset_index()
    if result.id.duplicated().any() or not result.answer.isin(CHOICES).all():
        raise ValueError("제출 ID/답 검증 실패")
    prefix="submission_SMOKE_" if RUN_MODE=="smoke" else "submission_"
    path=RUN_DIR/(prefix+RUN_DIR.name+".csv")
    result.to_csv(path,index=False,encoding="utf-8")
    pd.DataFrame({"id":frame.id.tolist(),**{c:probs[:,j] for j,c in enumerate(CHOICES)}}).to_csv(
        RUN_DIR/"test_soft.csv",index=False,encoding="utf-8")
    return path
```

### compare_runs — code cell 26 (1부터 계산)

```python
def compare_runs(paths):
    records=[]
    prediction_sets={}
    expected=None
    for folder in map(Path,paths):
        file=folder/"validation_summary.json"
        if not file.exists():
            records.append({"run":str(folder),"comparable":False,"reason":"summary 없음"})
            continue
        info=json.loads(file.read_text())
        sig=info["validation_signature"]
        if expected is None:
            expected=sig
        if sig!=expected:
            records.append({"run":str(folder),"comparable":False,"reason":"validation ID/정답/문항 hash 다름"})
            continue
        prediction_sets[str(folder)]=pd.read_csv(folder/"validation_predictions.csv",dtype={"id":str}).set_index("id")
        records.append({"run":str(folder),"comparable":True,"n":info["n"],"accuracy":info["accuracy"],
            "before_accuracy":info.get("before_accuracy"),"model":info["model"],"shifts":str(info["shifts"]),
            "adapter_hash":info["adapter_hash"],"is_smoke":info["is_smoke"]})
    if prediction_sets:
        anchor_name,anchor=next(iter(prediction_sets.items()))
        a=anchor.pred.eq(anchor.answer)
        for row in records:
            if row.get("comparable"):
                other=prediction_sets[row["run"]].loc[anchor.index]
                b=other.pred.eq(other.answer)
                row.update(anchor=anchor_name,rescue=int((~a&b).sum()),harm=int((a&~b).sum()),
                           net=int((~a&b).sum()-(a&~b).sum()))
    return pd.DataFrame(records)
```

### export_top2_review — code cell 27 (1부터 계산)

```python
def export_top2_review(frame,ids,probs,path):
    if set(frame.id)!=set(map(str,ids)) or len(set(ids))!=len(ids):
        raise ValueError("교사 확률 ID와 정답 데이터가 다릅니다.")
    order={str(v):i for i,v in enumerate(ids)}
    p=np.asarray(probs)[[order[v] for v in frame.id]]
    check_probabilities(p,len(frame))
    gold=frame.answer.map(CHOICES.index).to_numpy()
    top2=np.argsort(p,axis=1)[:,-2:]
    outside=~(top2==gold[:,None]).any(1)
    review=frame.loc[outside,["id","path","question",*CHOICES,"answer"]].copy()
    review["teacher_top2"]=["/".join(CHOICES[j] for j in row[::-1]) for row in top2[outside]]
    review["review_status"]="pending_human_review"
    review.to_csv(path,index=False,encoding="utf-8-sig")
    return review
```

## 깡 실행 별첨의 설명용 코드

현재 학습용 노트북의 `RUN_MODE="infer"`는 학습된 adapter 추론입니다. 깡 실행은 별도 base-only loader에서 LoRA 생성과 adapter 로드를 생략해야 합니다. 아래는 공식 API와 본문 구조를 연결한 설명용 코드이며 GPU 실행은 확인하지 않았습니다.

```python
from unsloth import FastVisionModel
import torch

MODEL_ID = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
model, processor = FastVisionModel.from_pretrained(
    model_name=MODEL_ID, load_in_4bit=True, max_seq_length=4096,
)
FastVisionModel.for_inference(model)
model.eval()

# No get_peft_model, no load_adapter, no Trainer/backward.
# Prepare inputs with the correct image processor and chat template.
# Validate left padding, one-token answer letters, and answer boundary.
# choice_token_ids must be verified for this exact tokenizer.
with torch.inference_mode():
    logits = model(**inputs).logits[:, -1, :]
    choice_logits = logits[:, choice_token_ids]
    probs = torch.softmax(choice_logits.float(), dim=-1)
```

입력 준비에는 본문의 `messages_for`·`render_prompt`·`call_processor` 원칙을 적용하고, 보기 순서를 바꾸면 `restore_choice_order`로 원래 순서를 복원합니다. reasoning 답 경계가 맞지 않는 모델에는 이 Direct Choice 코드 조각을 그대로 적용하지 않습니다. 생성 hard 답만 있을 때는 hard로 표시합니다.

## 재현 전에 필요한 실제 자료

- CSV와 원본 이미지, sample submission, 정확한 출처 태그
- 수동 정제 CSV 4개 또는 정제본 provenance manifest
- 모델·외부 dataset revision과 실제 GPU 환경
- 같은 split manifest, adapter hash와 체크포인트 선택 기록
- full 실행 로그와 실제 valid/test 출력; Public/Private 제출 점수는 별도 근거

## 주요 출처

- 현재 학습용 노트북과 VQA_Reference_Integration.md
- A186_민구없는_민구팀.zip/src/train_gemma4_31b.py 및 requirements.txt
- 사용자 제공 단일 모델 Public 0.97229 솔루션 글
- https://huggingface.co/unsloth/gemma-4-31B-it-unsloth-bnb-4bit
- https://unsloth.ai/docs/basics/vision-fine-tuning
- https://arxiv.org/abs/2512.14982

원본 노트북 SHA256: `a4ff9b8a843818518e935186277acb322ce0848a41c0a1a35fe7f17645590a7f`
