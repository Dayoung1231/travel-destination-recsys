"""
공통 경로와 상수.

경로나 숫자를 코드에 직접 쓰지 말고 여기서 가져다 써요.
그래야 모든 코드가 같은 파일을 읽고 같은 설정을 써요.

    from src.config import PROCESSED_DIR, SEED, TOURIST_TYPES
"""
from pathlib import Path

# ---------------------------------------------------------------- 경로
ROOT = Path(__file__).resolve().parents[1]  # 프로젝트 최상위 폴더 (자동으로 찾음)

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # 원본 데이터 (읽기만, 수정 금지)
TL_DIR = RAW_DIR / "TL"                 #   AI Hub 여행 로그, Training
VL_DIR = RAW_DIR / "VL"                 #   AI Hub 여행 로그, Validation
TOURAPI_DIR = RAW_DIR / "tourapi"       #   TourAPI 수집 결과
ASOS_DIR = RAW_DIR / "asos"             #   기상청 ASOS 일자료
PROCESSED_DIR = DATA_DIR / "processed"  # 전처리 결과 (모델 입력)

RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"    # 발표용 그래프
METRIC_DIR = RESULTS_DIR / "metrics"    # 실행 결과 표·로그

# ---------------------------------------------------------------- 재현성
SEED = 42  # 무작위 분할, 모델 초기값 등 무작위가 들어가는 곳에 모두 사용

# ---------------------------------------------------------------- 추천 대상 장소
# 추천 대상으로 남길 방문지 유형 코드(VISIT_AREA_TYPE_CD).
# 제외: 4 상업지구, 9 역·터미널, 10 상점, 11 식당/카페,
#       12 기타, 21~24 집·친지집·사무실·숙소
TOURIST_TYPES = {
    "1": "Natural attraction",
    "2": "History / heritage / religion",
    "3": "Cultural facility",
    "5": "Leisure / sports",
    "6": "Theme facility",
    "7": "Walking trail",
    "8": "Local festival",
    "13": "Experience attraction",
}

# 수도권 시·도. 주소의 첫 단어와 앞부분이 일치하는지로 비교 ("경기"와 "경기도" 모두 잡힘)
REGIONS = ("서울", "경기", "인천")

# 분할 비율, 추천 개수, 야외 장소 유형 등은 정해지면 그 값을 처음 쓰는 PR에서 여기에 추가해요.
