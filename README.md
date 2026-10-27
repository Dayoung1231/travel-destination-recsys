# Personalized Travel Destination Recommendation for the Seoul Capital Area

Machine Learning Term Project — Team 8

여행 로그 데이터를 학습해 사용자의 여행 스타일, 동반자, 이동 거리·수단, 날씨, 유명 관광지 선호도까지 반영한 개인 맞춤형 수도권(서울·경기·인천) 여행지를 추천하는 시스템

<br>

## 1. 개요

| | |
|---|---|
| 분야 | 여행지 추천 |
| 추천 대상 | 수도권의 관광지 (자연관광지, 문화시설 등) |
| 대상 사용자 | 수도권으로 당일 또는 단기 여행을 계획하는 사용자 |
| 필터링 기법 | 콘텐츠 기반 필터링 + 모델 기반 협업 필터링 (MF, 직접 구현) |
| 확장 기능 | 다음 장소·코스 추천 (마르코프 체인, 연관 규칙, DBSCAN) |

<br>

## 2. 데이터셋

AI Hub 데이터는 재배포가 제한되어 있어서 레포에 **올리지 않아요.** 각자 직접 내려받아야 해요.
TourAPI 장소 목록(이용허락범위 제한 없음)과 ASOS 날씨 자료(공공누리 제1유형)는 재배포할 수 있어서 **레포에 포함해 뒀어요.**

| 데이터 | 출처 | 받는 방식 | 위치 |
|---|---|---|---|
| 국내 여행로그 데이터(수도권) | [AI Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71581) | 각자 파일 다운로드 (zip) | `data/raw/TL/`, `data/raw/VL/` |
| 관광정보 서비스 (TourAPI) | [공공데이터포털](https://www.data.go.kr/data/15101578/openapi.do) | **레포에 포함** | `data/raw/tourapi/area_list.csv` |
| 종관기상관측(ASOS) 일자료 | [기상자료개방포털](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36) | **레포에 포함** | `data/raw/asos/asos_daily.csv` |

### 2-1. 여행 로그 데이터 받기

위 AI Hub 페이지에서 `TL_csv.zip`과 `VL_csv.zip`을 내려받아서 압축을 푼 후, CSV 파일을 `data/raw/TL/`, `data/raw/VL/`에 넣어요.

> AI Hub의 Training/Validation 구분은 테이블마다 다르게 나뉘어 있어요. 그래서 두 폴더를 합쳐(여행자 3,600명) 우리가 직접 train / validation / test로 나눠요.

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

### 2-3. 날씨(ASOS) 데이터

`data/raw/asos/asos_daily.csv`가 레포에 들어 있어서 따로 받지 않아도 돼요.

**받은 조건** (다시 받을 때 같은 조건으로)

| 항목 | 값 |
|---|---|
| 자료형태 | 일자료 |
| 기간 | 2022-08-01 ~ 2022-11-30 (여행 로그 기간 2022-08-04 ~ 2022-11-13을 포함) |
| 지점 | 서울·인천·경기 9곳: 동두천(98), 파주(99), 백령도(102), 서울(108), 인천(112), 수원(119), 강화(201), 양평(202), 이천(203) |
| 요소 | 평균기온, 최저기온, 최고기온, 일강수량 |
| 형식 | CSV (CP949 인코딩, `pd.read_csv(..., encoding="cp949")`로 읽기) |

<br>

## 3. 시작하기

모든 명령은 프로젝트 최상위 폴더에서 실행해요.

```bash
pip install -r requirements.txt
```

> 전처리와 추천 실행 방법은 코드가 완성되면 추가해요.

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
│   │   ├── tourapi/       #     TourAPI 장소 목록 (레포에 포함)
│   │   └── asos/          #     ASOS 날씨 일자료 (레포에 포함)
│   └── processed/         #   train / val / test, items, users, visits
├── notebooks/             # 탐색용 (01_eda.ipynb …)
├── src/
│   ├── __init__.py
│   ├── config.py          # 공통 경로·시드·상수
│   ├── preprocessing/     # 전처리 파트
│   │   ├── __init__.py
│   │   ├── collect_tourapi.py  # TourAPI 장소 목록 수집
│   │   ├── load_data.py   # TL+VL 합치기, 중복 제거 등
│   │   ├── preprocess.py  # 장소 ID 통일, 필터, 분할 등
│   │   └── features.py    # TF-IDF, 카테고리 원-핫, 사용자 프로필 등
│   ├── modeling/          # 모델링 파트
│   │   ├── __init__.py
│   │   ├── content_based.py
│   │   ├── mf.py          # 직접 구현
│   │   └── course.py      # (확장) 마르코프 체인, 연관 규칙, DBSCAN
│   ├── adjust.py          # 동반자·날씨·인기도 점수 보정
│   ├── evaluate.py        # RMSE, Precision, NDCG
│   └── plots.py
├── recommend.py           # 터미널 데모
└── results/
    ├── figures/           # 발표용 그래프 (커밋 O)
    └── metrics/           # 실행 결과 표·로그 (커밋 O)
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
| `prepro/` | 전처리, EDA | `src/preprocessing/`, `notebooks/` | `prepro/split` |
| `model/` | 모델링, 하이퍼파라미터 튜닝, 평가 | `src/modeling/` | `model/tuning-mf` |
| `common/` | 두 파트가 같이 쓰는 파일 | README, `references.md`, `src/config.py` 등 | `common/readme` |

- 오류 수정도 파트 안에서 해요. 예: `prepro/fix-item-filter`, `model/fix-mf-loss`

### 커밋 메시지

`타입: 짧은 설명` 형식으로 써요. 콜론(`:`) 뒤에 한 칸 띄워요.

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

- 경로와 공통 상수(시드, 관광 유형 코드 등)는 코드에 직접 쓰지 말고 `src/config.py`에서 가져와요. `C:/Users/...` 같은 절대 경로는 다른 사람 컴퓨터에서 동작하지 않아요.
- `data/raw/`의 원본은 수정하지 않아요. 가공한 결과는 `data/processed/`에 저장해요.
- 노트북(`.ipynb`, 코랩 등)은 출력을 지운 뒤 커밋해요.
- 발표에 쓸 그래프는 `results/figures/`, 실행 결과 표·로그는 `results/metrics/`에 저장해요.
- 인터넷(블로그, Kaggle, GitHub 등)의 코드를 참고했으면 코드 옆 주석과 [references.md](references.md)에 **출처를 남겨요.**
- 수업에서 다루지 않은 모듈·클래스·라이브러리·함수는 [references.md](references.md)에 파라미터까지 설명해요.

<br>

## 7. 팀

| 이름 | 역할 |
|---|---|
| 노다연 | |
| 박소윤 | |
| 지승찬 | |
| 황다영 | |
