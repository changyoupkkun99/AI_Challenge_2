# 재사용 가능한 이미지 VQA 베이스라인 — 전체 단계·선택 기능 체크리스트

작성일: 2026-10-04
문서 상태: 설계·구현 체크리스트 및 초기 설정 작성. 전체 학습 실행기 구현이나 GPU 실행 완료를 뜻하지 않는다.

현재 기본은 `VQA_Baseline_Config.py`의 `reference_gemma31` 프리셋이다. 참고 솔루션 글과 A186 ZIP을 대조한 차이·가져올 기능·값은 `VQA_Reference_Integration.md`, 펼친 기본 설정은 `VQA_Default_Config.json`에 정리했다. 아래 체크박스는 전체 실행기에서 확인할 작업이며, 초기 설정만 작성한 항목을 GPU 검증 완료로 표시하지 않는다.

목표: 비슷한 한국어 이미지 4지선다 VQA 프로젝트에서 데이터·모델·실험 설정을 바꿔 실행할 수 있는 단일 모델 베이스라인을 만든다. SSAFY AI Challenge 2는 출처를 확인할 수 있는 실행 사례로 연결한다. 7,169 / 5,735 / 1,434 / 6,714 같은 개수나 특정 모델·GPU를 범용 코드의 상수로 고정하지 않는다.

필수: 데이터 검증, 분할·누수 검사, 입력·정답 매핑, 실행 설정 기록, 오류 기록, 실제 결과 저장.
선택: 성능에 영향을 주는 증강, OCR/DINO, 프롬프트 반복, 양자화, LoRA 범위, 선지 재배치, 고해상도, TTA, 같은 모델의 추론 결과 라우팅.
선택 기능은 구현에서 제거하지 않고 설정으로 켜고 끈다. 동시에 전부 켜는 것을 기본값으로 삼지 않는다. 아직 실제 코드에 없던 기능은 확장 설계로 표시한다.

## 01. 환경 준비

- [ ] 실행 위치 설정: Colab / 로컬 / GPU 서버.
- [ ] GPU 이름·VRAM·CUDA capability·가용 메모리 확인.
- [ ] Python, CUDA, PyTorch, torchvision 등 실제 버전 기록.
- [ ] 모델·GPU에 맞는 의존성 설치 프로필 선택.
- [ ] transformers, accelerate, peft, bitsandbytes, GPTQModel, Unsloth 등은 사용하는 backend에 맞게 설치.
- [ ] OCR/DINO·증강·차트 패키지는 해당 기능 선택 시 설치.
- [ ] 버전 충돌·설치 후 런타임 재시작 여부 점검.
- [ ] 모델 다운로드 인증은 필요할 때 환경변수로 전달하고 토큰을 설정 파일이나 로그에 저장하지 않음.
- [ ] 디스크 용량, 모델·데이터 캐시, 출력 경로 확인.
- [ ] 환경 스냅샷 및 설치 버전 파일 저장.

환경 준비는 패키지 설치와 런타임 확인이다. 03단계의 라이브러리 import·데이터 로드·실험 설정과 구분한다.

## 02. 데이터 준비

- [ ] 데이터 입력 방식 선택: 로컬 폴더 / ZIP / Drive / 데이터셋 로더.
- [ ] 필요한 경우 Drive 마운트 및 ZIP의 VM 디스크 복사.
- [ ] 압축 해제 완료 여부·원본 ZIP 해시·CSV 위치 확인.
- [ ] 경로가 안전한지, 이미지가 실제 존재하는지 검사.
- [ ] train, 선택적 dev, 선택적 외부 데이터, test, sample_submission의 역할 분리.
- [ ] 원본과 정제본, 생성된 분할, 캐시, 학습 결과의 경로 구분.
- [ ] 암호화 ZIP은 제공된 비밀번호를 입력할 수 있게 하고 로그에는 기록하지 않음.
- [ ] 데이터별 실제 행 수와 이미지 수를 계산해 저장.

## 03. 라이브러리·데이터·설정

- [ ] 기본 import: pathlib, os, json, csv, random, time, hashlib, logging 등.
- [ ] 데이터·이미지 import: pandas, numpy, PIL/ImageOps 등.
- [ ] 학습 import: torch, 모델별 transformers API, peft, optimizer/scheduler 등.
- [ ] 선택 import: OCR, Grounding DINO, albumentations, Plotly 등.
- [ ] 실행 모드 설정: train / evaluate / infer, smoke / full.
- [ ] 모델 ID·revision·backend·dtype·양자화·adapter·resume checkpoint 설정.
- [ ] CSV 열 매핑, 데이터 경로, 클래스 목록, seed, split 비율 설정.
- [ ] 이미지 처리, 증강, OCR/DINO, 프롬프트, loss, LoRA 설정.
- [ ] epoch, learning rate, batch, gradient accumulation, warmup, scheduler, 저장 간격 설정.
- [ ] 평가 방식, 선지 permutation, 고해상도, TTA, 라우팅 및 저장 설정.
- [ ] run ID, config.json, 출력 디렉터리 생성.
- [ ] 불가능하거나 모순되는 설정 조합을 시작 전에 검사.

## 04. 데이터 정제·라벨·출처 검증

- [ ] 공통 입력 형식: id, image/path, question, choices, answer.
- [ ] 이번 사례의 a/b/c/d 열을 공통 choices 구조로 변환하는 연결 코드.
- [ ] 빈 ID·중복 ID·잘못된 정답·깨진 이미지·누락 보기 확인.
- [ ] a/b/c/d와 0/1/2/3 매핑을 기록하고 클래스 순서를 고정.
- [ ] 원본 답·정제 후 답·선지 수정·삭제 이유를 감사 기록으로 남김.
- [ ] 원본 라벨, 사람이 검토한 라벨, dev pseudo-label 등 출처 보존.
- [ ] dev 투표 필터나 모델 기반 검토 대상 선별은 옵션으로 제공.
- [ ] 모델 Top-2 밖에 있다는 이유만으로 정답을 자동 수정하거나 삭제하지 않음.
- [ ] 이번에 받은 clean7169 데이터 생성 정책은 실제 원본 CSV·manifest 확인 후 확정.
- [ ] 외부 데이터는 출처·변환 규칙·학습 사용 여부를 기록.

## 05. 4:1 분할·중복 방지·정답 위치 균형

- [ ] 기본 train:validation = 80:20. 입력 데이터 규모에 따라 실제 개수 계산.
- [ ] 동일 이미지의 여러 문항을 한 이미지 그룹으로 묶음.
- [ ] 기본 SHA-256 그룹, 필요 시 pHash 기반 추가 그룹을 선택 옵션으로 제공.
- [ ] 이미지 그룹을 유지하면서 정답 비율을 가능한 한 유지하는 층화 분할.
- [ ] 그룹 제약으로 분할 비율·클래스 비율에 오차가 생기면 실제 분포 기록.
- [ ] seed·data signature·train IDs·validation IDs·이미지 그룹을 split manifest에 저장.
- [ ] 같은 프로젝트에서는 분할 재사용, 데이터가 바뀌면 새로운 manifest 생성.
- [ ] train/validation 이미지 누수 및 dev·외부 데이터와 validation의 중복 검사.
- [ ] 선지 위치 균형 모드: 원본 유지 / 무작위 셔플 / 균등 위치 재배치.
- [ ] 균등 모드에서는 학습 정답 위치를 a/b/c/d에 균등 배정하고 보기 순서와 답을 함께 재매핑.
- [ ] 학습 문항 수가 4의 배수가 아니면 클래스 간 1개 정도의 차이 허용.
- [ ] 무작위 셔플은 기댓값의 균형이며 매번 정확한 25%를 보장하지 않음을 구분.
- [ ] 보기를 직접 참조하는 질문·보기는 재배치 가능 여부 검사.
- [ ] validation의 원래 보기·정답은 보존. 추론 permutation은 원래 순서로 복원해 채점.

4:1 분할과 각 정답 위치 25%는 다른 규칙이다. 층화 분할만으로 원본의 불균형이 25%씩으로 바뀌지 않는다. 균등화 목적으로 문제를 버리거나 정답 내용 자체를 바꾸지 않는다.

## 06. 비율 유지 Padding + 학습용 Augmentation

- [ ] EXIF 방향 보정과 RGB 변환.
- [ ] 이미지 정책 선택: 모델 native 처리 / 종횡비 유지·픽셀 수 상한 / 종횡비 유지 letterbox padding.
- [ ] 기존 직접 정사각 resize도 비교용 옵션으로 보존하되 왜곡 여부 기록.
- [ ] 해상도·픽셀 예산·padding 색상·resampling 설정.
- [ ] 학습용 증강 on/off 및 강도·확률·seed 설정.
- [ ] 기존 Hybrid 기능: 밝기·대비, 색상, 노이즈, JPEG 압축.
- [ ] 참고 솔루션 확장 기능: JPEG q70, Gaussian blur r1.0, affine 이동·shear.
- [ ] 기존 Hybrid 증강과 참고 솔루션의 4배 증강을 별개 preset으로 구분.
- [ ] train에는 선택한 증강 적용, validation/test는 정해진 평가용 전처리 적용.
- [ ] 증강 전 원본 ID·원본 이미지와 연결, 모든 증강본은 원본과 같은 분할에 배치.
- [ ] 원본 수·확장 후 학습 샘플 수·epoch별 노출 수 기록.
- [ ] 모델 processor가 추가로 만든 실제 이미지 크기·토큰 수 점검.

참고 솔루션의 affine은 단순 회전이 아니라 이동과 shear이다. 증강 이름만 같다고 구현과 강도가 같다고 가정하지 않는다.

## 07. Local OCR + Grounding DINO 1회 캐시

- [ ] aux_mode = none / ocr / dino / both.
- [ ] 선택한 기능만 로드·실행, VLM 학습 전에 불필요한 보조 모델 메모리 해제.
- [ ] OCR backend·언어·confidence threshold·최대 항목 수·입력 길이 설정.
- [ ] DINO 모델·탐지 후보 문구·score threshold·최대 객체 수 설정.
- [ ] 원본 이미지에서 text/confidence/bbox, label/score/box를 저장.
- [ ] 안정적인 이미지 해시·출처·캐시 버전으로 캐시 키 관리.
- [ ] 기존 절대경로 키 캐시는 새 데이터 경로와 매칭을 검증한 후 재사용.
- [ ] OCR 문자열 정제·중복 제거·confidence 순 선택.
- [ ] DINO 객체 이름 정제·중복 제거·score 순 선택.
- [ ] 캐시 누락·빈 결과·실행 오류를 구분해 기록.
- [ ] 중간 저장·재개 및 이미지 수 대비 coverage 검사.
- [ ] 기본적으로 학습/validation/test에 동일한 aux_mode 적용.
- [ ] 학습 후 참고자료를 끄거나 바꾸는 실험은 별도 inference 실험으로 명시.
- [ ] 정답·다른 모델의 제출 답을 OCR/DINO 입력이나 캐시에 섞지 않음.

기존 Hybrid의 확인된 prompt 필터: OCR confidence >= 0.40, 최대 15개 / DINO score >= 0.30, 최대 12개. OCR 원본 캐시에는 confidence >= 0.25인 결과도 남지만 prompt 필터와는 다르다. 현재 prompt에는 객체 이름을 넣는다. box 자체를 VLM에 입력하거나 crop을 추가하는 기능은 별도 확장이다.

## 08. 프롬프트 템플릿·채팅 템플릿

- [ ] 이미지·질문·보기·참고자료를 공통 입력 생성 함수로 조합.
- [ ] OCR/DINO가 꺼져 있으면 해당 참고자료 블록을 넣지 않음.
- [ ] 이미지와 보조자료가 충돌할 때 이미지 판단을 우선하는 기존 지시문 보존.
- [ ] 정답은 a/b/c/d 한 글자 출력 형식으로 설정.
- [ ] system/user 메시지 구성과 모델별 apply_chat_template 연결.
- [ ] thinking on/off를 지원 범위 내에서 선택.
- [ ] 프롬프트 반복 횟수 설정: 기본 기존 방식, 2회 반복은 참고 솔루션 확장 옵션.
- [ ] answer generation boundary, 빈 thinking 블록, 종료 토큰 처리 확인.
- [ ] BOS 중복 여부·context 길이·이미지 토큰 길이 점검.
- [ ] 길이 초과를 무조건 잘라 이미지·정답 경계를 손상시키지 않음.
- [ ] 모델·template revision과 실제 prompt 예시·template hash 저장.

## 09. 모델·Processor·양자화 로드

- [ ] 모델 ID와 revision을 고정 또는 최초 조회 후 기록.
- [ ] backend 선택: Hugging Face / GPTQModel / Unsloth 등 모델별 연결.
- [ ] BF16 / FP16 / NF4 / GPTQ / AWQ 등 지원되는 조합 선택.
- [ ] 해당 backend가 fine-tuning을 지원하는지 확인.
- [ ] processor·tokenizer를 모델과 같은 revision에 연결.
- [ ] native 이미지 처리 설정, context, padding side, special tokens 확인.
- [ ] 필요한 모델별 호환성 패치만 선택적으로 적용.
- [ ] 기존 GPTQ trainable kernel 및 backward patch는 해당 프로필에 보존.
- [ ] 모델 로드 후 VRAM·quantized module·trainable module 기록.
- [ ] 학습/평가 모드의 use_cache·gradient checkpointing 전환 확인.
- [ ] 다른 GPU나 backend로 바뀌었다고 이전 실험의 성공을 자동 보장하지 않음.

## 10. LoRA 설정

- [ ] 미학습 평가 / LoRA fine-tuning / 저장된 adapter 추가 학습 중 실행 경로 선택.
- [ ] LoRA rank r, alpha, dropout, bias 설정.
- [ ] 범위 선택: 언어 attention / 언어 attention+MLP / 지정 module.
- [ ] 비전 encoder·projector 학습 여부는 지원되는 모델에서 명시적 옵션.
- [ ] embedding·LM head·비전 부분의 동결 여부 기록.
- [ ] Qwen/Agnes hybrid attention의 global attention과 delta attention projection 처리 보존.
- [ ] 자동 module 탐색 결과·실제 trainable parameter 수·대상 목록 저장.
- [ ] k-bit 학습 준비와 gradient checkpointing 설정.
- [ ] adapter load/save 및 base model revision 일치 검사.
- [ ] 새 rank·target 설정과 기존 adapter가 호환되는지 확인.

모든 모델에 q_proj/k_proj/v_proj/o_proj 이름만 있다고 가정하지 않는다. 언어 attention-only와 MLP 추가 학습을 각각 실험 가능한 옵션으로 남긴다.

## 11. Custom Dataset·Collator·Loss

- [ ] 학습 Dataset: 이미지 처리 → 선택적 증강 → 선지 재배치 → 참고자료 → prompt → 정답.
- [ ] 검증/test Dataset: 선택한 평가 이미지 처리와 prompt, 정답은 입력에 포함하지 않음.
- [ ] 선지 재배치 전후 정답 내용이 같은지 검사.
- [ ] 입력 부분 label=-100, 답변 부분만 supervision.
- [ ] loss 모드 구분: assistant 답변 토큰(종료 토큰 포함 가능) / 정답 글자 1토큰.
- [ ] 모델 chat template의 prompt prefix와 실제 답변 경계 검증.
- [ ] input_ids, attention_mask, mm_token_type_ids 등 모델별 필드 처리.
- [ ] padding·attention mask·정답 토큰 위치 검사.
- [ ] batch 제한과 지원되는 collator 방식 기록.
- [ ] 몇 개 학습 예시의 decode 및 loss 대상 토큰 확인.

기존 공통/Gemma/Agnes의 answer-only supervision은 종료 토큰을 포함할 수 있다. Qwen Only와 참고 솔루션의 1글자 loss와 동일하다고 설명하지 않는다.

## 12. DataLoader

- [ ] per-device batch size, gradient accumulation, 실효 batch 기록.
- [ ] train shuffle, seed, worker, pin memory 등 설정.
- [ ] validation/test ID 순서는 저장된 기준 순서 유지.
- [ ] epoch별 선지 재배치·증강 재현 규칙 정의.
- [ ] 소량 smoke subset과 full dataset 구분.
- [ ] 마지막 불완전 batch/gradient accumulation 처리 검증.

## 13. Preflight·Smoke Test·Benchmark

- [ ] 데이터-only 점검이 GPU 모델 로드 없이 실행 가능하도록 분리.
- [ ] 모델 로드·이미지 encode·chat template smoke.
- [ ] a/b/c/d가 단일 토큰인지 검사. 다중 토큰 모델은 sequence scoring 연결이 필요.
- [ ] 추론 경계의 다음 토큰이 실제로 정답 위치인지 확인.
- [ ] 실제 forward + backward 및 trainable LoRA gradient 검사.
- [ ] 점검으로 본 학습이 오염되지 않도록 optimizer step·RNG·모델 상태 처리.
- [ ] 소량 학습·저장·재로드·재개 동작 확인.
- [ ] 속도·peak VRAM·예상 처리 시간 기록.

## 14. 파인튜닝

- [ ] 선택한 단일 모델에 대해 train 실행.
- [ ] AdamW/8bit optimizer 등 지원되는 optimizer 선택.
- [ ] learning rate, epoch, scheduler, warmup, weight decay, gradient clipping 설정.
- [ ] gradient accumulation, mixed precision, gradient checkpointing 설정.
- [ ] loss, LR, step, elapsed time, GPU memory 로그 저장.
- [ ] checkpoint 저장 간격·보관 개수·final adapter 저장.
- [ ] 선택 기준: final / best validation loss / best validation accuracy를 명시.
- [ ] resume 시 optimizer·scheduler·step·학습 순서·RNG 상태 확인.
- [ ] 학습 전과 후의 동일 프로토콜 평가를 선택적으로 실행.
- [ ] 외부→내부 2-Stage는 후속 확장 슬롯으로 보존. 현재 단계에서 자동 적용하지 않음.
- [ ] 별도 학습 단계 간 adapter 전달·optimizer 초기화·단계별 데이터 수 기록.

## 15. 기본 Validation 평가·응답 파싱

- [ ] 전체 validation accuracy 및 라벨 출처별 accuracy.
- [ ] generate 방식 평가: greedy/sampling 설정과 max_new_tokens 명시.
- [ ] 원문 응답·파싱 결과·정답·format failure·실행 오류 저장.
- [ ] 파싱 실패를 특정 답으로 몰래 대체하지 않음.
- [ ] 학습 전/후의 평가 prompt·이미지 정책·ID 집합 맞춤.
- [ ] generation 평가와 direct-choice 평가 결과를 별도로 기록.
- [ ] 학습에서 본 문제 또는 test 정답을 validation에 사용하지 않음.

## 16. Direct Choice Logit / Softmax 검증

- [ ] 정답 생성 위치에서 a/b/c/d의 실제 logits 추출.
- [ ] 토큰 ID·토큰 개수·다음 토큰 위치·thinking 종료 경계 검증.
- [ ] 4개 logits에 numerically stable Softmax 적용.
- [ ] 확률 shape, finite 여부, 행별 합=1, 클래스 순서 확인.
- [ ] argmax 답·최대 확률·상위 두 확률의 margin 저장.
- [ ] generate 답과 direct-choice argmax의 일치 여부·정확도 비교.
- [ ] 선택적 temperature calibration은 validation의 calibration 부분에만 fit하고 설정 기록.
- [ ] hard 답을 one-hot으로 변환해 실제 모델 soft 출력으로 저장하지 않음.

4개 선지 Softmax는 a/b/c/d 후보 사이의 정규화 확률이다. 전체 어휘에서 해당 글자들이 차지하는 확률 질량이나 완전히 보정된 정답 확률과 같다고 가정하지 않는다.

## 17. Option Permutation 검증

- [ ] inference permutation 모드: 원래 순서 / single shift / 여러 shift / 사용자 지정.
- [ ] shift0=abcd, shift1=bcda, shift2=cdab, shift3=dabc.
- [ ] shown index와 original index 사이의 매핑 보존.
- [ ] logits·확률·예측을 원래 a/b/c/d 순서로 복원한 후 채점.
- [ ] shift별 accuracy·클래스별 accuracy·source별 accuracy 비교.
- [ ] 원래 순서 대비 changed / rescue / harm / net 계산.
- [ ] shift 선택용 calibration subset ID와 나머지 평가 ID를 구분.
- [ ] 선택에 사용한 문항을 포함한 전체 점수라면 그 사실도 기록.
- [ ] shift calibration CSV와 selected_shift JSON 저장.
- [ ] 기존 Qwen Only의 shift0/shift3, 최소 +1%p 선택 규칙은 사례 preset으로 보존. 범용 기본값으로 강제하지 않음.
- [ ] 그 외 24가지 전체 permutation 실험은 별도 확장 옵션.

## 18. High-Resolution 1024 + 4-way TTA

- [ ] 해상도/픽셀 예산/vision token budget을 모델별로 설정.
- [ ] 기존 해상도와 high-resolution 설정을 validation에서 비교.
- [ ] 해상도별 실제 입력 크기·이미지 토큰 수·시간·메모리 기록.
- [ ] TTA 모드: off / single shift / 2-way / 4-way / 사용자 지정.
- [ ] 4-way는 같은 모델·이미지에 shift0/1/2/3 적용.
- [ ] 매 shift의 실제 logits·확률을 원래 클래스 순서로 복원.
- [ ] arithmetic probability mean / geometric mean / 지정 weights를 실험 옵션으로 보존.
- [ ] 최종 확률에서 argmax 선택.
- [ ] single shift 대비 rescue/harm, source별 정확도, 추가 비용 비교.
- [ ] 선택된 해상도·TTA 설정을 test에도 동일 적용.
- [ ] 이미지 변환 TTA와 선지 순서 TTA는 별개 옵션으로 구분. 현재 확인된 핵심은 선지 순서 TTA.

1024는 모든 모델에서 같은 의미의 processor 파라미터가 아니다. 실제 설정과 처리 결과를 기록해야 한다. TTA는 한 모델을 여러 입력 변형으로 실행하는 기능이며 여러 모델을 학습할 필요가 없다.

## 19. 같은 모델의 추론 결과 라우팅·결합

- [ ] 기존 Hybrid의 Smart Routing 기능을 선택 옵션으로 보존.
- [ ] single shift, 다른 shift, TTA 등의 확률 중 선택하는 규칙 설정.
- [ ] confidence/margin 기반 routing threshold 기록.
- [ ] 같은 모델의 결과 결합과 다른 모델의 ensemble을 구분.
- [ ] weight/threshold 선택은 validation에서만 수행.
- [ ] anchor 대비 rescue/harm과 최종 규칙을 기록.
- [ ] 미래의 여러 모델 ensemble은 출력 파일을 이용한 별도 후속 작업으로 둠.

## 20. Test 추론·제출 CSV

- [ ] 최종 checkpoint와 선택한 입력·해상도·permutation·TTA 설정 고정.
- [ ] test.csv와 sample_submission의 ID 집합·순서 검증.
- [ ] test 문항 수를 실제 데이터에서 계산.
- [ ] 중간 logits/확률, 완료 mask, ID 목록, 실행 signature를 함께 저장.
- [ ] 동일 checkpoint·설정·ID일 때만 partial resume.
- [ ] 오류·재시도·OOM 처리 내역 저장. precision/prompt 등을 바꾸면 별도 run.
- [ ] 완료 전 partial 결과와 최종 결과를 구분.
- [ ] 제출 CSV에는 id,answer만 저장하고 값·행 수·중복·순서 검사.
- [ ] 실제 제출 점수는 파일명·해시·제출 기록과 연결될 때만 기록.

## 21. NPY·JSON·CSV 및 재현 기록 저장

각 run의 필수 출력:

| 경로 | 내용 |
|---|---|
| config.json | 실제 실행 설정·모델·환경·선택 기능 |
| dataset_manifest.json | 데이터 출처·해시·실제 개수·정제 정책 |
| split_manifest.json | seed·비율·이미지 그룹·train/validation IDs |
| model_metadata.json | 모델 revision·processor/template·backend·LoRA |
| checkpoints/ | 중간 checkpoint와 학습 재개 상태 |
| adapter/ 또는 checkpoint-final/ | 선택한 최종 checkpoint |
| probs/valid_logits.npy | 원래 클래스 순서의 실제 4개 logits, (N_valid,4) |
| probs/valid_probs.npy | 실제 logits에서 계산 또는 기록된 TTA 방식으로 결합한 최종 확률, (N_valid,4) |
| probs/valid_labels.npy | a=0,b=1,c=2,d=3 기준 라벨, (N_valid,) |
| probs/valid_ids.json | NPY 행과 같은 ID 순서 |
| probs/test_logits.npy | 원래 클래스 순서의 실제 4개 logits, (N_test,4) |
| probs/test_probs.npy | 최종 test 확률, (N_test,4) |
| probs/test_ids.json | test NPY와 같은 ID 순서 |
| validation_predictions.csv | 답·예측·정오·confidence·margin·라벨 출처 |
| validation_summary.json | 전체·출처별 성능·실패·평가 규칙 |
| analysis/shift_calibration.csv | shift별 비교 결과 |
| analysis/selected_shift.json | 선택한 shift·기준·calibration ID |
| test/partial_* 및 done mask | test 중간 저장·재개 |
| submission_<model>_<run>.csv | 검증한 최종 제출 형식 |
| train_log.* / training_summary.json | 학습 과정·checkpoint 선택·소요 시간 |
| charts/ | 데이터·학습·평가 차트 |
| comparison/ | 실험 간 성능·rescue/harm·설정 차이 |

- [ ] 단일 shift 모드에서는 logits와 probs가 해당 shift에서 직접 대응.
- [ ] TTA 모드에서는 valid/test_shift_logits.npy와 valid/test_shift_probs.npy를 추가로 저장. 권장 shape=(K,N,4).
- [ ] 기본 (N,4) logits는 명시한 reference shift의 실제 값을 보존하고 logits_reference_shift를 기록.
- [ ] 최종 probs가 여러 shift의 probability mean이면 그 규칙을 기록. reference logits에 Softmax를 적용한 값과 같다고 설명하지 않음.
- [ ] 평균 확률의 log를 실제 모델 raw logits라고 저장하지 않음.
- [ ] 필요 시 계산한 mean logits는 별도 이름으로 저장하고 계산 방식을 명시.
- [ ] 선택적 학습 전 validation 결과는 before/after로 구분해 저장.
- [ ] model revision·checkpoint hash·data/split hash·class order·ID order·image/aux/TTA 설정을 prediction manifest에 기록.
- [ ] 아직 관측되지 않은 점수는 미확인으로 남김.

## 22. 실제 결과 차트

- [ ] 데이터 출처·분할·정답 위치 분포.
- [ ] 이미지 크기·종횡비·전처리 결과.
- [ ] train/validation loss, learning rate, step.
- [ ] 학습 전/후 accuracy와 라벨 출처별 accuracy.
- [ ] 클래스별 성능·confusion matrix.
- [ ] confidence·margin 분포와 오답 분석.
- [ ] OCR/DINO coverage·오류·참고자료 길이.
- [ ] 해상도·shift·TTA별 정확도·rescue/harm.
- [ ] GPU memory·시간·처리량.
- [ ] Plotly HTML 등 재확인 가능한 차트 저장.
- [ ] 실제 저장된 결과를 차트 입력으로 사용. 예시 점수로 실제 차트를 만들지 않음.

## 23. 실행 결과 비교

- [ ] 같은 데이터·분할·ID·정답 버전인지 먼저 검사.
- [ ] 모델 revision·checkpoint·prompt·image·aux·loss·학습 설정의 차이 표시.
- [ ] OCR/DINO 비교는 다른 조건을 같게 유지한 실험으로 구분.
- [ ] 학습 때 aux_mode를 바꾼 비교와 같은 checkpoint의 inference-only 비교를 구분.
- [ ] LoRA on/off, attention-only/MLP 추가, rank/LR/epoch 비교.
- [ ] 양자화 비교와 파인튜닝 효과 비교를 구분.
- [ ] generate/direct-choice, 해상도, single shift/TTA/routing 비교.
- [ ] accuracy뿐 아니라 changed/rescue/harm/net, 출처별 성능, 시간·VRAM 비교.
- [ ] 선택적으로 paired 비교 및 McNemar 분석.
- [ ] 공통 validation을 사용하지 않은 과거 점수는 직접 순위 비교에서 분리.
- [ ] 선택·조정에 사용한 validation 점수와 독립 평가 점수를 구분.
- [ ] best run 선택 기준과 선택에 사용한 데이터 기록.

## 24. 프로젝트 설명·최종 감사 기록

- [ ] README: 설치→데이터→설정→smoke→학습→평가→test→저장 순서.
- [ ] 옵션 표, 필요한 데이터 파일, 실제 실행 예시.
- [ ] 처음 보는 사람이 단계별로 읽을 수 있는 셀 제목·역할·입출력.
- [ ] 원본·정제·분할·학습·checkpoint·예측·제출의 대응 관계.
- [ ] 현재 사례의 확인된 사실 / 설명과 코드의 차이 / 미확인 자료 구분.
- [ ] 실패와 재개, 제한사항, 모델별 필요한 연결 코드 문서화.
- [ ] 코드 구현 완료·GPU smoke 통과·전체 학습·test 완료 상태를 따로 표시.

## 옵션 설정표 — 제안하는 공통 설정 이름

아래 이름은 새 베이스라인에 구현할 설정 규격이다. 기존 모든 노트북에 이미 같은 이름으로 구현돼 있다는 뜻은 아니다.

| 설정 | 선택 값·내용 | 기본 설계 원칙 |
|---|---|---|
| task | train / evaluate / infer | 실행 목적 명시 |
| profile | smoke / full | 점검과 본 실행 분리 |
| train_fraction | 기본 0.8 | 데이터 개수 고정 없음 |
| group_policy | sha256 / sha256+phash | 이미지 그룹 보존 |
| train_option_policy | original / random / balanced | 사용자가 요청한 균등 모드 제공 |
| augmentation | none / hybrid / reference_4x / custom | 기능 유지, preset 구분 |
| image_policy | native / aspect_cap / letterbox / resize_square | 모델별 비교 |
| image_resolution | 사용자 지정 | 실제 processor 결과 기록 |
| max_image_pixels | 사용자 지정 | context/VRAM과 함께 확인 |
| aux_mode | none / ocr / dino / both | 켠 모듈만 실행 |
| ocr_confidence | 사례 값 0.40 | 프로젝트별 변경 가능 |
| max_ocr_items | 사례 값 15 | 길이 제한과 함께 설정 |
| dino_score | 사례 값 0.30 | 프로젝트별 변경 가능 |
| max_object_items | 사례 값 12 | 중복 제거 |
| prompt_repeat | 1 / 2 / 사용자 지정 | Gemma 참고 프리셋은 2회 선택 |
| enable_thinking | false / true(지원 모델) | 템플릿 경계 확인 |
| backend | hf / gptqmodel / unsloth 등 | 실제 지원 조합 |
| precision | bf16 / fp16 / 지원 양자화 | 모델·GPU 프로필별 |
| finetune_mode | none / lora / adapter_continue | 단일 모델 경로 |
| lora_scope | language_attention / language_attention_mlp / custom | 실제 대상 목록 기록 |
| lora_r / alpha / dropout | 사용자 지정 | 사례 값을 범용 상수로 고정하지 않음 |
| loss_mode | answer_tokens / one_letter | 종료 토큰 포함 여부 구분 |
| batch_size / grad_accum | 사용자 지정 | 실효 batch 기록 |
| epochs / lr / scheduler | 사용자 지정 | 실제 실행 설정 저장 |
| checkpoint_selection | final / best_val_loss / best_val_accuracy | 선택 기준 명시 |
| eval_mode | generate / direct_choice / compare | 프로토콜별 점수 구분 |
| temperature | 기본 1.0 / calibration 값 | calibration 자료 기록 |
| inference_shifts | [0] / [3] / [0,2] / [0,1,2,3] / custom | 원래 클래스 순서 복원 |
| tta_aggregation | arithmetic / geometric / weighted | raw shift 출력 보존 |
| routing_mode | none / confidence_margin | validation 근거 기록 |
| training_stages | internal / external_then_internal | Gemma 참고 프리셋은 외부→내부, 각 1 epoch |
| save_raw_logits | true | 실제 모델 출력 |
| save_probabilities | true | hard와 분리 |
| save_shift_outputs | TTA/shift 비교 시 true | 미래 분석·결합 가능 |
| save_charts / compare_runs | true 또는 선택 실행 | 관측 결과만 사용 |

## 출처별 기능 대응

| 자료 | 확인한 주요 기능 | 새 베이스라인에 연결할 위치 |
|---|---|---|
| 팀_공통_베이스라인.ipynb | 환경/설정, SHA-256 이미지 그룹 80/20, 답변 mask, 학습 전후 평가, Plotly, run 비교 | 01~05, 11~15, 22~23 |
| (260908)_baseline_colab_hybrid_qwen38_v2 (1)(1)(1).ipynb | 비율 유지 padding, 학습 증강, EasyOCR/DINO cache·참고 prompt, GPTQ trainable/LoRA, 실제 backward, checkpoint·resume, direct logits, shift/TTA, Smart Routing | 06~14, 16~21 |
| gemma4_31b_TRAINING(1).ipynb | clean7169 검증·고정 split, BF16 LoRA, 모델별 template/loss, 전후 평가·차트·비교, test 경로 | 02~05, 08~15, 20, 22~23 |
| Agnes-3.0-Flash_clean7169_full_v3(2)(1).ipynb | clean7169·BF16 LoRA, hybrid attention target, direct-choice/permutation, high-resolution·4-way TTA, test resume·확률 기록 | 02~05, 08~23 |
| GitHub Qwen38_27B_clean7169_AgnesStyle_A100.ipynb | GPTQ/LoRA backward, 1토큰 loss, aspect cap, shift calibration, logits/probs/labels/IDs·CSV 저장 | 08~21 |
| A186_민구없는_민구팀.zip | 외부→내부 2-Stage, prompt repetition, 4종 이미지 증강, 선지 셔플, shift0/2 확률 평균, 검증 예측 비교 | 현재 초기 프리셋: 04, 06, 08, 14, 18, 23. 전체 실행기 연결은 남음 |

## 확인된 사례와 아직 남은 확인

- 공통 베이스라인 출력: 원본 train 기준 5,371/1,343.
- Hybrid의 분할 코드와 출력: 원본 train의 10%, validation 672. 그룹 분할과 같다고 가정하지 않음.
- Gemma/Agnes의 정제본 출력: train 출처 5,527 + dev 출처 1,642 = 7,169, split 5,735/1,434.
- 저장된 차트의 학습 정답 개수: a=1,444, b=1,344, c=1,464, d=1,483. 정확히 25%씩은 아님.
- validation 정답 개수: a=350, b=357, c=360, d=367.
- 제공된 Agnes에는 OCR/DINO 사용이 없다고 명시돼 있다. 결합본이라는 설명만으로 OCR/DINO 입력을 포함했다고 기록하지 않음.
- Qwen Only 코드에는 요청한 NPY/JSON 저장이 이미 구현돼 있다. 조회한 GitHub 노트북에는 실행 출력이 없어서 그 코드만으로 실제 점수를 확정하지 않음.
- Hybrid와 Qwen Only는 OCR뿐 아니라 이미지 처리·학습 설정·평가셋도 다르므로 과거 점수 차이를 OCR 단독 효과로 확정하지 않음.
- A186 README는 Public 0.97229를 명시한다. ZIP에 그 최종 제출 CSV·학습 로그·checkpoint는 없어 독립 점수 검증은 아직 못 함.
- A186의 hard 문제 역순 증강은 experiments/run_transfer_v2.py에 있고 최종 src/train_gemma4_31b.py에는 없음.
- A186의 validation ID 파일은 671개. 우리 1,434개와 동일 평가셋으로 비교하지 않음.
- A186 문서의 검토용 Qwen 모델 표기가 3.5/3.8로 서로 다르며 실제 검토 실행 자료로 확정해야 함.
- clean_dataset_7169.zip, split JSON, dataset_manifest 및 정제 원본 목록은 이번 노트북 첨부에 없어서 해시·정제 정책을 독립 재계산하지 못 함.
- 제공 Gemma 설정 셀에는 learning_rate=5e-5,, 구문이 남아 있어 실행 가능한 새 코드에서는 정리해야 한다. 저장 출력은 현재 셀과 다른 시점의 기록일 수 있음.
- 초기 설정·JSON 내보내기는 작성했다. 새 학습 실행기와 GPU 검증은 남아 있다. 지원 여부·정확한 실행 설정·결과는 단계별로 확인하고 기록한다.

## 현재 적용한 참고 프리셋

- 모델: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`, Unsloth FastVisionModel, bitsandbytes 4bit.
- LoRA: r16/alpha32/dropout0, language attention+MLP, vision 고정.
- loss: 다음 정답 글자 1토큰에만 **전체 어휘 CE**. 추론에서의 4개 선지 softmax와 구분.
- Stage 1: MTVQA 한국어+KRETA, 원본 이미지, LR1e-4, 1 epoch.
- Stage 2: 정제 train+선별 dev, orig/JPEG70/blur1.0/affine(이동3%·shear5°), LR1e-4, 1 epoch.
- 프롬프트: 질문·보기·출력 지시 2회, 이미지 1회, thinking 비활성, 답 경계 검사.
- 이미지: 종횡비 유지·긴 변 상한2048·Gemma max_soft_tokens1120. 1024 padding은 옵션.
- 추론: Direct Choice Logit→각 shift softmax→원래 클래스 순서 복원→shift0/2 산술 확률 평균.
- 새 베이스 분할: 이미지 그룹 기준 4:1. 학습 정답 위치 균등화, validation 원본 순서 보존.
- OCR/DINO·hard 반전·1024/4-way TTA·같은 모델 라우팅: 옵션, 현재 기본 꺼짐.
- 실제 logits/probs/labels/IDs·각 shift 출력·비교 CSV/summary·차트 저장: 실행기에서 생성할 필수 규격.
- `*_logits.npy`는 실제 기준 shift0 logits, `*_probs.npy`는 최종 TTA 평균. 서로 다른 의미를 metadata로 명시하고 기준 shift 확률도 별도 저장.
- 기존 프로필: 공통/Qwen Only/Google Gemma/Agnes 핵심값을 남겼으며 원본 업로드는 수정하지 않음.
- 참고 Public0.97229는 보고된 성적, 목표0.97은 새 실행에서 아직 확인되지 않음.
