# lcy_정리

SSAFY 이미지 4지선다 VQA 모델·공통 자료의 정리본입니다. 기존 원본은 보존하고 이 폴더에 사본을 모았습니다.

## 모델별 public 점수

| 담당 | 모델 폴더 | public 점수 |
|---|---|---:|
| 창엽 | [Qwen3.8-27B_OCR-DINO](https://drive.google.com/drive/folders/11zDbIjOZCn3aNTfGoQlP2LRq9GS25UgB) | 0.95918 |
| 창엽 | [Qwen3.8-27B_Only](https://drive.google.com/drive/folders/1NK8Z6FWxe-yeqnfDLq5HUekSdLay2inU) | 0.95621 |
| 창엽 | [Qwen3.5-122B-A10B-GPTQ-Int4](https://drive.google.com/drive/folders/1QjkGwT9ySPqjWK5toaItBfyUknWmUUND) | 0.96097 |
| 창엽 | [Agnes-3.0-Flash](https://drive.google.com/drive/folders/1_R9Qg1PyoTxwytK58apOJOCTC8elEXQt) | 0.96157 |
| 창엽 | [GLM-4.6V_AWQ-4bit](https://drive.google.com/drive/folders/1HaSSn5IuVcTQAn01R2NWOdO_sFIgM9zK) | 0.78194 |
| 창엽 | [Ling-3.0-flash](https://drive.google.com/drive/folders/1FM7W3QijX5dZ4rI2aW1tyEaEd83Ghgdn) | 0.94638 |
| 종서 | [Qwen3-VL-4B-Instruct](https://drive.google.com/drive/folders/1lJDdUCHsKMOe9s1rz9SJ2Rze6s7gmOEY) | 0.94399 |
| 종서 | [Qwen3-VL-8B-Instruct](https://drive.google.com/drive/folders/1NPvXThZQhSiyS2r6ysb_fde4DI-97YCm) | 0.95174 |
| 종서 | [Gemma-4-26B-A4B-it](https://drive.google.com/drive/folders/1FSlX3zI_0hUQf0s-Ll-d8Tm6pbSapPtR) | 0.96216 |
| 종서 | [Gemma-4-31B-it](https://drive.google.com/drive/folders/1hTh6Z4d48nNbetGI5CeL9Y1txCfiGh-w) | 0.96544 |
| 종서 | [Qwen2.5-VL-7B-Instruct](https://drive.google.com/drive/folders/1B386mkzj-Ny6vOAk9CTKgVrLcmEme2eS) | 0.89603 |

점수는 사용자가 제공한 public leaderboard 값이며 private 성능과 다릅니다. 모델별 validation 점수는 데이터 범위(1,343/1,434 등)를 명시해 별도로 기록합니다.

## 폴더

- `공통/`: 대회 solution ZIP, 팀 공통 베이스라인 노트북, 이미지 제외 `cleaning_data`, 원본 데이터 위치.
- `발표자료/`: 1차 발표 PDF와 [종서의 발표 근거 문서·그림](https://drive.google.com/drive/folders/1ysVz3XjvMyzPqAfDF2S7gh7w5Fcqy3P-).
- `창엽/`, `종서/`: 담당자별 `00_모델별_안내.md`와 모델별 노트북·제출 CSV·환경. 종서의 네 모델에는 대표 실행 어댑터·설정 자료도 있습니다.
- `종서/실험_기록/`: [전체 실행 결과표, 대표 실행별 지표 안내, 기존 안내 원본](https://drive.google.com/drive/folders/1-zzr8d0oF4gHp3Llmju17kxkWi6dyEsz).
- `앙상블_결과/`: 최종 제출 후보와 검토 기록.
- `파일_대응표.csv`: 모델·점수·노트북·제출 CSV 연결.

## 재현과 데이터

- 원본 이미지 파일은 정리 폴더에 중복 저장하지 않았습니다. `공통/데이터_위치.md`의 원본 Drive 위치를 참조하세요.
- 테스트 데이터/정제 데이터 경로는 실행 노트북마다 다릅니다. 원본 `data.zip` 또는 압축 해제된 경로를 우선 확인합니다.
- `valid_probs.npy`/`test_probs.npy`는 원본 ID·class 순서가 확인된 실제 4선지 확률만 인정합니다. 없는 모델을 one-hot으로 채우지 않습니다.
- 일부 노트북은 여러 실험 셀이 섞여 있습니다. 파일 대응표의 확인 메모를 보고 해당 점수와 실행본을 구분하세요.
- Qwen3-VL-4B 대표 실행은 RTX 5060 Ti의 BF16 LoRA입니다. 실행 이름의 `full`은 전체 데이터 프로필이며 전체 파라미터 미세조정이 아닙니다.
