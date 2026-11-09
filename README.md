# Personalized Travel Destination Recommendation for the Seoul Capital Area

Machine Learning Term Project — Team 8

여행 로그 데이터를 학습해 사용자의 여행 스타일, 동반자, 여행 미션·동기, 유명 관광지 선호도, 이번 여행에서 다녀온 곳을 반영해 개인 맞춤형 수도권(서울·경기·인천) 여행지를 추천하는 시스템

<br>

## 1. 개요

| | |
|---|---|
| 분야 | 여행지 추천 |
| 추천 대상 | 수도권의 관광지 (자연관광지, 문화시설 등) 1,516곳 |
| 대상 사용자 | 수도권으로 당일 또는 단기 여행을 계획하는 사용자 |
| 필터링 기법 | 콘텐츠 기반 필터링 (CBF) + 모델 기반 협업 필터링 (방문 MF, 직접 구현) |
| 추천 방식 | 전환형 하이브리드: 설문만 입력하면 CBF, 이번 여행에서 다녀온 곳까지 입력하면 방문 MF |
| 확장 기능 (예정) | 다음 장소·코스 추천 (마르코프 체인, 연관 규칙, DBSCAN) |

<br>

## 2. 데이터셋

AI Hub 데이터는 이용 조건상 승인받지 않은 사람에게 제공할 수 없고, 국외 반출도 별도 합의가 필요해서 레포에 **올리지 않아요.** 각자 AI Hub에서 직접 내려받아야 해요.
TourAPI 장소 목록·장소 설명(이용허락범위 제한 없음)과 ASOS 날씨 자료(공공누리 제1유형)는 재배포할 수 있어서 **레포에 포함해 뒀어요.**

| 데이터 | 출처 | 받는 방식 | 위치 |
|---|---|---|---|
| 국내 여행로그 데이터(수도권) (2023) | [AI Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71776) | 각자 파일 다운로드 (zip) | `data/raw/TL/`, `data/raw/VL/` |
| 관광정보 서비스 (TourAPI) | [공공데이터포털](https://www.data.go.kr/data/15101578/openapi.do) | **레포에 포함** | `data/raw/tourapi/area_list.csv`, `overview.csv` |
| 종관기상관측(ASOS) 일자료 | [기상자료개방포털](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36) | **레포에 포함** | `data/raw/asos/asos_daily.csv` |

### 2-1. 여행 로그 데이터 받기

위 AI Hub 페이지에서 `TL_csv.zip`과 `VL_csv.zip`을 내려받아서 압축을 푼 후, CSV 파일을 `data/raw/TL/`, `data/raw/VL/`에 넣어요.

| 항목 | Training (TL) | Validation (VL) | 합계 |
|---|---|---|---|
| 여행객 (1명당 여행 1번) | 2,560명 | 320명 | **2,880명** |
| 방문 기록 | 21,384건 | 2,770건 | 24,154건 |
| 방문 기간 | | | **2023-04-28 ~ 2023-09-18** |

> 공식 Validation 320명만 테스트로 쓰면 평가할 정답이 너무 적어서, 두 폴더를 합쳐(여행객 2,880명) 우리가 직접 train / validation / test로 나눠요.

### 2-2. 관광정보(TourAPI) 데이터

`data/raw/tourapi/area_list.csv`가 레포에 들어 있어서 따로 받지 않아도 돼요.

**받은 조건**

| 항목 | 값 |
|---|---|
| 수집일 | 2026-10-28 (TourAPI는 실시간으로 바뀌어서, 다시 받으면 내용이 조금 달라질 수 있어요) |
| 기능 | 지역기반 관광정보조회 (`areaBasedList2`, KorService2) |
| 지역 (`lDongRegnCd`) | 11 서울, 28 인천, 41 경기 |
| 관광 타입 (`contentTypeId`) | 12 관광지, 14 문화시설, 15 축제공연행사, 28 레포츠 |
| 개수 | 5,389곳 (서울 1,660 · 인천 733 · 경기 2,996) |
| 형식 | CSV (UTF-8) |

**다시 수집하려면** (인증키가 필요한 경우는 이때뿐이에요)

1. 공공데이터포털에 로그인하고 위 페이지에서 **활용신청**을 눌러 인증키를 받아요. 개발계정은 기능별로 하루 1,000회까지 호출할 수 있어요.
2. `.env.example`을 복사해 이름을 `.env`로 바꾸고, 포털의 **일반 인증키**를 적어요. `.env`는 git에 올라가지 않아요.
3. 프로젝트 최상위 폴더에서 수집 코드를 실행해요.

   ```bash
   python -m src.preprocessing.collect_tourapi
   ```

**장소 설명 (`overview.csv`)**

CBF의 텍스트 점수(TF-IDF)에 쓰는 장소 설명이에요. 여행 로그 장소와 매칭된 TourAPI 관광지 834곳의 설명(`overview`)을 받아 뒀어요. 그중 4곳은 설명이 비어 있어요. 다시 받으려면 전처리를 먼저 돌린 뒤 실행해요. 이미 받은 장소는 건너뛰어요.

```bash
python -m src.preprocessing.collect_overview
```

### 2-3. 날씨(ASOS) 데이터

`data/raw/asos/asos_daily.csv`가 레포에 들어 있어서 따로 받지 않아도 돼요.

**받은 조건** (다시 받을 때 같은 조건으로)

| 항목 | 값 |
|---|---|
| 자료형태 | 일자료 |
| 기간 | 2023-04-01 ~ 2023-09-30 (여행 로그 기간 2023-04-28 ~ 2023-09-18을 포함) |
| 지점 | 서울·인천·경기 9곳: 동두천(98), 파주(99), 백령도(102), 서울(108), 인천(112), 수원(119), 강화(201), 양평(202), 이천(203) |
| 요소 | 평균기온, 최저기온, 최고기온, 일강수량 |
| 형식 | CSV (CP949 인코딩, `pd.read_csv(..., encoding="cp949")`로 읽기) |

<br>

## 3. 실행하기

모든 명령은 프로젝트 최상위 폴더에서 실행해요.

```bash
pip install -r requirements.txt
```

### 3-1. 전처리

`data/raw/TL/`, `data/raw/VL/`에 여행 로그 CSV를 넣은 뒤 실행해요. 결과는 `data/processed/`에 생기고 (git에 안 올라가요), 요약 숫자는 `results/metrics/preprocess_summary.txt`에 남아요. 시드가 고정돼 있어서 누가 돌려도 결과가 같아요.

```bash
python -m src.preprocessing.preprocess
```

| 파일 | 내용 | 주로 쓰는 곳 |
|---|---|---|
| `visits.csv` | 여행지 방문 (여행객 x 장소 방문 1행), `ITEM_ID`는 같은 장소를 합친 ID | 전체 |
| `items.csv` | 장소 속성 (유형, 위치, train 통계, TourAPI 매칭, 비 온 날 방문 비율) | CBF, DBSCAN, 보정 |
| `users.csv` | 여행객 속성 (동반자 그룹, 여행 스타일, 미션·동기), `SPLIT` | CBF, 보정 |
| `item_features.csv`, `user_features.csv` | 0~1 숫자 특성 행렬 | CBF |
| `ratings_train.csv` | train 여행객 평점 (MF용 장소만) | MF |
| `ratings_{val,test}_known.csv` | val·test 여행객의 앞부분 평점 (학습에 같이 넣음) | MF |
| `ratings_{val,test}_hidden.csv` | val·test 여행객의 여행 뒤쪽 30% (정답) | 평가 |
| `item_index.csv`, `user_index.csv` | MF 행렬 인덱스 | MF |
| `holdout_by_order.csv` | 여행객 x 장소 방문 순서, 여행 뒤쪽 30%면 `HOLDOUT=1` | 평가 (cold·warm), 방문 MF |
| `user_folds.csv` | 전체 여행객 5-fold 분할 | (현재 미사용) |
| `sequences.csv`, `transitions.csv` | 여행별 방문 순서, 장소·유형 전이 확률 (train) | 마르코프 체인 |
| `transactions.csv` | 여행별 장소·유형 집합 | 연관 규칙 |
| `place_legs.csv`, `place_pairs.csv` | 여행지 -> 다음 여행지 거리(`DIST_KM`)·이동수단 | 코스, 이동 보정 |
| `weather.csv` | 방문별 날짜, 가장 가까운 ASOS 관측소의 기온·강수량 | 날씨 보정 |


### 3-2. 모델 학습과 평가

전처리를 먼저 돌린 뒤 실행해요. 각 모델이 val로 설정을 비교하고, 결과 표를 `results/metrics/`에 저장해요. 학습된 모델은 따로 저장하지 않아요.

```bash
python -m src.modeling.cbf     # 콘텐츠 기반 필터링 -> cbf_val_*.csv
python -m src.modeling.mf      # 평점 MF, 방문 MF -> mf_val_*.csv
```

- `--check`: 여러 사람을 한 번에 채점하는 빠른 평가가 한 사람씩 채점한 결과와 같은지 확인해요. 점수 계산을 고친 뒤 실행해요.
- `--test`: 고른 설정을 test로 채점해요. **모든 모델이 끝난 뒤 마지막에 한 번만** 써요.

**평가 방법** (`src/modeling/evaluate.py`, 모든 모델이 같은 코드로 채점)

| | cold | warm |
|---|---|---|
| 상황 | 설문만 입력 | 설문 + 이번 여행에서 다녀온 곳 |
| 입력 | 없음 | 여행 앞쪽 70% 장소 |
| 정답 | 그 사람이 간 장소 전부 | 여행 뒤쪽 30% 장소 |

- 후보: train 여행객이 한 번이라도 간 수도권 장소 중 지역 축제를 뺀 1,516곳
- 지표: Recall@K, NDCG@K, Hit@K (K = 5, 10, 20). 설정을 고르는 기준은 cold·warm NDCG@10 평균이에요.

**모델 요약**

| 모델 | 파일 | 내용 |
|---|---|---|
| CBF | `cbf.py`, `text.py` | 장소 속성 27칸(유형, TourAPI 분류, 활동 비율, 체류 시간, 지역)과 사용자 프로필(설문 릿지 회귀 + 다녀온 곳)의 가중 자카드 유사도에 인기도·동반자 보정과 장소 설명 TF-IDF 점수를 더함 |
| 방문 MF | `mf.py` | 간 곳 1 / 안 간 곳 0에 가중치를 준 행렬 분해를 ALS로 학습. 설문으로 새 사용자 임베딩의 시작점을 만들어 cold에도 추천 |
| 평점 MF | `mf.py` | 만족도 1~5점 예측 (SGD, RMSE). 평점이 4~5점에 몰려 있어 추천에는 쓰지 않고 비교용으로 둠 |

**val 결과** (NDCG@10)

| 모델 | cold | warm |
|---|---|---|
| 무작위 (20번 평균) | 0.004 | 0.003 |
| 인기순 | 0.092 | 0.063 |
| CBF | **0.155** | 0.177 |
| 방문 MF | 0.130 | **0.296** |

cold는 CBF, warm은 방문 MF가 좋아서 터미널 추천은 두 모델을 상황에 따라 골라 써요. warm에서 방문 MF가 강한 이유는 같은 여행에서 함께 가는 근처 장소를 잘 맞히기 때문이에요.

**그래프** (cbf.py, mf.py를 먼저 실행한 뒤)

```bash
python -m src.plots      # results/figures/val_*.png, 그래프 숫자는 results/metrics/plots_val_summary.csv
```

| 파일 | 내용 |
|---|---|
| `val_1_models.png` | 모델 비교 (무작위, 인기순, 평점 MF, CBF, 방문 MF), 95% 신뢰구간 |
| `val_2_cbf_steps.png` | CBF 단계별 성능과 직전 단계 대비 p값 |
| `val_3_warm_region.png` | warm을 같은 시·군·구 정답 유무로 나눈 비교 |
| `val_4_tuning.png` | CBF 텍스트 세기, 방문 MF의 k·alpha 설정 곡선 |

### 3-3. 터미널 추천

```bash
python -m src.recommend            # 추천 10곳
```

설문 8문항(성별, 연령대, 동반자, 여행 스타일 3문항, 여행 미션·동기, 원하는 지역)에 답하고, 마지막에 이번 여행에서 이미 다녀온 곳을 선택해서 입력해요.

- 다녀온 곳을 입력하지 않으면 **CBF**로 추천해요. 학습과 형태소 분석기 로딩으로 30초 정도 걸려요.
- 다녀온 곳을 입력하면 **방문 MF**로 추천해요. 2초 안팎이에요.
- 수도권 장소만 추천하고, 다녀온 곳은 다시 추천하지 않아요.

<br>

## 4. 폴더 구조

```
travel-destination-recsys/
├── README.md
├── references.md          # 참고한 코드 출처, 수업 외 라이브러리 설명
├── requirements.txt
├── .gitignore             # data/ (tourapi/, asos/ 제외), .env, __pycache__/
├── data/                  # !! tourapi/, asos/ 말고는 커밋 금지
│   ├── raw/               #   원본, 수정 금지
│   │   ├── TL/, VL/       #     AI Hub 여행 로그 (각자 다운로드)
│   │   ├── tourapi/       #     TourAPI 장소 목록, 장소 설명 (레포에 포함)
│   │   └── asos/          #     ASOS 날씨 일자료 (레포에 포함)
│   └── processed/         #   train / val / test, items, users, visits
├── src/
│   ├── __init__.py
│   ├── config.py          # 공통 경로·시드·상수
│   ├── recommend.py       # 터미널 추천 (설문 -> CBF 또는 방문 MF)
│   ├── plots.py           # 발표용 성능 그래프
│   ├── preprocessing/     # 전처리 파트
│   │   ├── __init__.py
│   │   ├── collect_tourapi.py   # TourAPI 장소 목록 수집
│   │   ├── collect_overview.py  # TourAPI 장소 설명 수집
│   │   ├── load_data.py   # TL+VL 읽어 합치기
│   │   ├── places.py      # 같은 장소 합치기 (장소 ID 통일), 사진 GPS로 좌표 채우기
│   │   ├── external.py    # TourAPI 매칭, ASOS 날씨 붙이기
│   │   ├── preprocess.py  # 전처리 전체 실행: 필터, 분할, 평점, 코스용 데이터
│   │   └── features.py    # 카테고리 원-핫, 장소·사용자 특성 행렬
│   └── modeling/          # 모델링 파트
│       ├── __init__.py
│       ├── evaluate.py    # 공용 평가: 후보 장소, cold·warm 분할, Recall·NDCG·Hit
│       ├── cbf.py         # 콘텐츠 기반 필터링 (사용자 프로필, 유사도, 점수 보정)
│       ├── text.py        # 장소 설명 TF-IDF (Kiwi 명사 추출)
│       └── mf.py          # 평점 MF, 방문 MF (직접 구현)
└── results/               # 실행 결과 (커밋 O)
    ├── figures/           # 발표용 그래프: val_1_models.png … val_4_tuning.png (plots.py)
    └── metrics/           # 실행 결과 표·로그
        ├── preprocess_summary.txt  # 전처리 요약 숫자 (preprocess.py)
        ├── cbf_val_*.csv          # CBF 설정·보정·텍스트 세기 비교 (cbf.py)
        ├── mf_val_*.csv           # 평점 MF RMSE, 방문 MF 그리드 (mf.py)
        └── plots_val_summary.csv  # 그래프에 쓴 평균·95% 신뢰구간 (plots.py)
```

<br>

## 5. 브랜치 관리 규칙

- `main` 하나와 작업 브랜치로 관리해요. **`main`에 직접 커밋하지 않고**, 작업 브랜치에서 Pull Request로 합쳐요.
- 작업 기간이 짧아서 Merge는 기본적으로 작성자가 바로 하고, 대신 PR 내용은 꼭 적어요. 다 쓴 브랜치는 삭제해요.
- 단, `src/config.py`를 바꾸는 경우 팀원 1명 이상 확인받은 후 Merge해요.
- Merge한 뒤에 문제를 발견하면 PR에 댓글을 남기고, 필요하면 GitHub의 **Revert** 버튼으로 PR 단위로 되돌려요.

| 브랜치 | 용도 |
|---|---|
| `main` | 항상 실행되는 최신 버전. 직접 커밋 금지 |
| `파트/작업내용` | 작업 하나를 하는 브랜치. 합친 뒤 삭제 |

### 브랜치 이름

`파트/작업내용` 형식으로 만들어요. 모두 소문자로 쓰고, 띄어쓰기 대신 하이픈(`-`)을 써요.

| 파트 | 언제 | 주로 고치는 폴더 | 예시 |
|---|---|---|---|
| `prepro/` | 전처리, EDA | `src/preprocessing/` | `prepro/split` |
| `model/` | 모델링, 하이퍼파라미터 튜닝, 평가 | `src/modeling/` | `model/tuning-mf` |
| `common/` | 두 파트가 같이 쓰는 파일 | README, `references.md`, `src/config.py` 등 | `common/readme` |

### 커밋 메시지

`타입: 짧은 설명` 형식으로 써요.

| 타입 | 언제 | 예시 |
|---|---|---|
| `feat` | 새 기능, 새 모델 | `feat: MF 학습 루프 구현` |
| `fix` | 오류 수정 | `fix: items.csv에 방문 수 적은 장소 유지` |
| `eda` | 데이터 탐색, 분석 그래프 | `eda: 장소별 방문 수 분포 확인` |
| `docs` | README, `references.md` 등 문서 | `docs: TF-IDF 파라미터 설명 추가` |
| `refactor` | 동작은 같고 코드 구조만 개선 | `refactor: 분할 함수를 split.py로 분리` |
| `chore` | 설정, 패키지, 폴더 정리 | `chore: requirements에 requests 추가` |

### Pull Request 작성

- **제목**: 커밋 메시지와 같은 형식으로 써요. 예: `feat: MF 학습 및 검증 RMSE 계산`
- **내용**: 무엇을 수정했는지 간단히 요약해요.

<br>

## 6. 협업 규칙

- 경로와 공통 상수(시드, 관광 유형 코드 등)는 코드에 직접 쓰지 말고 `src/config.py`에서 가져와요.
- `data/raw/`의 원본은 수정하지 않아요. 가공한 결과는 `data/processed/`에 저장해요.
- 발표에 쓸 그래프는 `results/figures/`, 실행 결과 표·로그는 `results/metrics/`에 저장해요.
- 인터넷(블로그, Kaggle, GitHub 등)의 코드를 참고했으면 코드 옆 주석과 [references.md](references.md)에 **출처를 남겨요.**
- 수업에서 다루지 않은 모듈·클래스·라이브러리·함수는 [references.md](references.md)에 파라미터까지 설명해요.

<br>

## 7. 팀

| 이름 | 역할 |
|---|---|
| 노다연 | 전처리 |
| 박소윤 | 모델링 및 평가 |
| 지승찬 | 전처리 |
| 황다영 | 모델링 및 평가 |
