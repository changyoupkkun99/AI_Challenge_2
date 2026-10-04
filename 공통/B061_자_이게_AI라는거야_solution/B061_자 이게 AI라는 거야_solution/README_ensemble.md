## 각각의 이름 폴더

각각의 팀원 이름으로 된 폴더는 개개인의 모델을 연구한 노트북 + 로그 등이 들어가 있습니다.

-> 함깨 첨부하고 싶었지만 용량이 너무커 공식 계정에 구글 드라이브 종서 숙제 폴더 안에 정리 해 놨습니다

## ensemble_merge 실행

이 폴더에서 Python 3.9 이상으로 실행합니다. GPU나 별도 패키지가 필요 없습니다.

```bash
python3 ensemble_merge/merge.py
```

`submission_ensemble_merge.csv`와 문항별 투표 내역, 검증 결과를 생성합니다. 다른 위치에 생성하려면 `--output-dir 경로`를 추가합니다. 내용이 다른 기존 출력 파일은 덮어쓰지 않습니다.


## (가장 최신)rf-버전 ensemble_rf 실행
이 폴더에서 실행하세요!

rf 학습 부터 csv 생성
```bash
python3 -m pip install -r ensemble_rf/requirements.txt
python3 ensemble_rf/run.py --retrain
```
저장해 놓은 최종 버전 재현
```bash
python3 ensemble_rf/run.py
```
이게 앙상블을 하면서 여러번 바뀐 폴더라 구조가 복잡할 수 있습니다ㅠㅠ