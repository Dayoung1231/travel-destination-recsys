"""
장소 설명 텍스트 -> TF-IDF 행렬 (CBF 텍스트 유사도용).

설명은 data/raw/tourapi/overview.csv (collect_overview.py로 수집), 장소와는 items의 TOUR_ID로 잇는다.
토큰: Kiwi로 뽑은 일반·고유 명사 (2글자 이상, 불용어 제외). 숫자와 꺾쇠 기호는 지운다.
가중치: scikit-learn TfidfVectorizer (강의 자료 p.70~71과 같은 도구)
    tf'  = 1 + ln(tf)          (sublinear_tf=True, 강의 p.59의 log TF)
    idf  = ln(N / df) + 1      (smooth_idf=False, 강의 p.64의 log(N/df)에 1을 더한 형태)
    w    = tf' x idf, 장소마다 L2 정규화 (강의 p.82)
    N은 설명이 있는 장소 수. MIN_DF곳 미만이나 MAX_DF 비율 넘게 나온 단어는 뺀다 (강의 p.99).
설명이 없는 장소는 0 벡터이고, has_text로 구분한다.
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TOURAPI_DIR
from src.modeling.evaluate import load_table

MIN_DF = 2      # 이보다 적은 장소에 나온 단어는 제외 (오타·고유한 이름)
MAX_DF = 0.5    # 설명이 있는 장소의 절반 넘게 나온 단어는 제외
# 거의 모든 설명에 나오지만 장소의 성격과 관계없는 말
STOPWORDS = set("관광 위치 이용 운영 시간 방문 장소 지역 시설 주변 대표 다양 가능 제공 경우 사람 정도 하나 "
                "당시 이후 현재 이곳 한국 우리나라 대한민국".split())

_kiwi = None


def nouns(text):
    """설명 한 개 -> 명사 목록. Kiwi는 처음 쓸 때 한 번만 불러온다 (몇 초 걸린다)."""
    global _kiwi
    if _kiwi is None:
        try:
            from kiwipiepy import Kiwi
        except ImportError:
            raise SystemExit("kiwipiepy가 없어요. pip install -r requirements.txt로 설치해 주세요.") from None
        _kiwi = Kiwi()
    text = re.sub(r"\d+", " ", re.sub(r"[<>《》]", " ", text.replace("\n", " ")))
    return [t.form for t in _kiwi.tokenize(text)
            if t.tag in ("NNG", "NNP") and len(t.form) >= 2 and t.form not in STOPWORDS]


@lru_cache(maxsize=None)
def item_tfidf():
    """item_features와 같은 장소 순서의 TF-IDF.
    Returns 장소 ID 배열, TF-IDF 행렬 (장소 x 단어, numpy), 설명 있음 여부 (bool 배열), 단어 목록"""
    ids = load_table("item_features")["ITEM_ID"].to_numpy()
    tour = load_table("items").set_index("ITEM_ID")["TOUR_ID"].reindex(ids)
    overview = pd.read_csv(TOURAPI_DIR / "overview.csv", dtype=str, keep_default_na=False)
    overview = overview.set_index("contentid")["overview"]
    text = tour.astype("Int64").astype("string").map(overview).fillna("").to_numpy()
    has = np.array([len(t) > 0 for t in text])

    vec = TfidfVectorizer(tokenizer=nouns, token_pattern=None, lowercase=False, min_df=MIN_DF, max_df=MAX_DF,
                          sublinear_tf=True, smooth_idf=False, norm="l2")
    vec.fit(text[has])  # IDF는 설명이 있는 장소만으로 계산
    M = np.zeros((len(ids), len(vec.vocabulary_)))
    M[has] = vec.transform(text[has]).toarray()
    return ids, M, has, vec.get_feature_names_out()
