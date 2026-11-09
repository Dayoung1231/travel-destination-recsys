# References

과제 명세서 요구 사항:
- 인터넷(블로그, Kaggle, GitHub 등)의 코드를 쓰면 **반드시 출처를 밝혀요.** 밝히지 않으면 표절로 간주돼요.
- 수업에서 다루지 않은 모듈·클래스·라이브러리·함수·메서드는 **파라미터까지 설명**해요.

코드를 추가할 때마다 바로 적어 주세요. 제출 직전에 몰아서 쓰면 빠뜨리기 쉬워요.

## 1. Code sources

| Where it is used | Source (URL) | What was taken / changed | Added by |
|---|---|---|---|
| `src/modeling/mf.py` `ExplicitMF` | Koren, Bell, Volinsky, "Matrix Factorization Techniques for Recommender Systems", IEEE Computer 2009, https://doi.org/10.1109/MC.2009.263 | Biased MF prediction (mu + b_u + b_i + p_u . q_i) and SGD update rule; implemented from scratch with numpy | |
| `src/modeling/mf.py` `ImplicitMF` | Hu, Koren, Volinsky, "Collaborative Filtering for Implicit Feedback Datasets", ICDM 2008, https://doi.org/10.1109/ICDM.2008.22 | Confidence weighting c = 1 + alpha and ALS closed-form update (Y^T Y + Y^T (C_u - I) Y trick); implemented from scratch. Fold-in toward a survey-based prior is our own addition | |

## 2. Libraries and functions not covered in class

| Function / class | Used in | What it does | Parameters we use |
|---|---|---|---|
| `kiwipiepy.Kiwi` (Korean morphological analyzer, https://github.com/bab2min/kiwipiepy) | `src/modeling/text.py` `nouns()` | Loads the Korean morphological analyzer model. Created once and reused because loading takes a few seconds | Default settings (no parameters) |
| `Kiwi.tokenize(text)` | `src/modeling/text.py` `nouns()` | Splits a sentence into morphemes. Each token has `form` (the word) and `tag` (part of speech) | `text`: one place description. We keep tokens whose `tag` is `NNG` (common noun) or `NNP` (proper noun), with 2 or more characters, not in our stopword list |
| `sklearn.feature_extraction.text.TfidfVectorizer` | `src/modeling/text.py` `item_tfidf()` | Turns place descriptions into TF-IDF vectors (place x word). TF-IDF and this class were covered in class (module 8-1); the parameters below go beyond the lecture examples | `tokenizer=nouns`: use Kiwi nouns instead of the default regex split, `token_pattern=None`: turn off the default regex (needed when `tokenizer` is given), `lowercase=False`: no case change for Korean, `min_df=2`: drop words in fewer than 2 places, `max_df=0.5`: drop words in more than half of the places, `sublinear_tf=True`: TF = 1 + ln(tf), `smooth_idf=False`: IDF = ln(N / df) + 1, `norm="l2"`: scale each place vector to length 1 |

## 3. Datasets

| Data | Source | License / terms |
|---|---|---|
| Domestic Travel Log (Seoul Capital Area) (2023) | AI Hub, https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71776 | AI Hub terms: AI training use only, no sharing with unapproved parties, attribution to NIA required |
| TourAPI Korean Tourism Information Service (place list, place descriptions) | Korea Tourism Organization, https://www.data.go.kr/data/15101578/openapi.do | No usage restrictions (images excluded) |
| ASOS Daily Weather Data | KMA, https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36 | KOGL Type 1 (attribution) |

**AI Hub attribution** (use in reports and presentations):
>
> This project uses the "Domestic Travel Log Data (Seoul Capital Area) (2023)" from AI Hub, a result of a project by the National Information Society Agency (NIA) of Korea.
