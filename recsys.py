"""
recsys.py  -  Logika data & model sistem rekomendasi (tanpa Streamlit)
Dataset : Book-Crossing (BX-Users, BX-Books, BX-Book-Ratings)

Isi modul:
  1. Load & cleaning data
  2. Pembentukan model (matriks sparse, TF-IDF, SVD)
  3. Rule-Based      : popularitas tersegmentasi + association rules
  4. Content-Based   : TF-IDF + cosine similarity (item profile & user profile)
  5. Collaborative   : Item-Based, User-Based (memory-based), SVD (model-based)
  6. Hybrid          : Weighted, Switching
  7. Evaluasi        : RMSE/MAE + Precision/Recall/HitRate @K
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

AGE_BINS = [5, 18, 25, 35, 45, 55, 65, 101]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_UNKNOWN = "Tidak diketahui"
AGE_ORDER = AGE_LABELS + [AGE_UNKNOWN]
LIKE_THR = 7          # rating >= 7 dianggap "suka"
SHRINK_MEAN = 2.0      # shrinkage rata-rata pengguna ke rata-rata global (penting utk pengguna baru)
MAX_YEAR = 2006       # dataset dikumpulkan 2004 -> tahun terbit > 2006 dianggap tidak valid

FILES = {
    "ratings": "BX-Book-Ratings.csv",
    "books": "BX-Books.csv",
    "users": "BX-Users.csv",
}


# =====================================================================
# 1. LOAD & CLEANING
# =====================================================================
@dataclass
class Data:
    users: pd.DataFrame      # user_id, Location, Age, City, Country, AgeGroup
    books: pd.DataFrame      # isbn, title, author, year, publisher, image, decade
    ratings: pd.DataFrame    # semua rating (termasuk implisit = 0)
    explicit: pd.DataFrame   # rating > 0 & ISBN ada di tabel buku, + atribut pengguna
    quality: dict            # ringkasan kualitas data


def _clean_users(u: pd.DataFrame) -> pd.DataFrame:
    u = u.rename(columns={"User-ID": "user_id"}).copy()
    u["Age"] = pd.to_numeric(u["Age"], errors="coerce")
    u.loc[(u["Age"] < 5) | (u["Age"] > 100), "Age"] = np.nan
    parts = u["Location"].fillna("").astype(str).str.split(",")
    u["City"] = parts.str[0].str.strip().str.title()
    country = parts.str[-1].str.strip().str.title()
    country = country.replace({"Usa": "USA", "Us": "USA", "U.S.A.": "USA", "United States": "USA",
                               "Uk": "United Kingdom", "": "Unknown", "N/A": "Unknown"})
    u["Country"] = country
    ag = pd.cut(u["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype("object")
    u["AgeGroup"] = ag.where(ag.notna(), AGE_UNKNOWN)
    return u


def _clean_books(b: pd.DataFrame) -> pd.DataFrame:
    b = b.rename(columns={"ISBN": "isbn", "Book-Title": "title", "Book-Author": "author",
                          "Year-Of-Publication": "year", "Publisher": "publisher",
                          "Image-URL-M": "image"}).copy()
    b["isbn"] = b["isbn"].astype(str).str.strip()
    for c in ["title", "author", "publisher"]:
        b[c] = b[c].fillna("Unknown").astype(str).map(html.unescape).str.strip()
    b["year"] = pd.to_numeric(b["year"], errors="coerce")
    b.loc[(b["year"] < 1800) | (b["year"] > MAX_YEAR), "year"] = np.nan
    b["decade"] = (b["year"] // 10) * 10
    return b.drop_duplicates("isbn")


def load_data(folder: str = "data") -> Data:
    """Baca 3 file CSV (separator ';', encoding latin-1) lalu bersihkan."""
    kw = dict(sep=";", encoding="latin-1", on_bad_lines="skip")
    r = pd.read_csv(f"{folder}/{FILES['ratings']}", **kw)
    b = pd.read_csv(f"{folder}/{FILES['books']}", escapechar="\\", **kw,
                    usecols=["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication",
                             "Publisher", "Image-URL-M"])
    u = pd.read_csv(f"{folder}/{FILES['users']}", **kw)

    r = r.rename(columns={"User-ID": "user_id", "ISBN": "isbn", "Book-Rating": "rating"})
    r["isbn"] = r["isbn"].astype(str).str.strip()
    n_raw = len(r)
    r = r.drop_duplicates(["user_id", "isbn"], keep="last")
    users, books = _clean_users(u), _clean_books(b)

    ex = r[r["rating"] > 0]
    in_books = ex["isbn"].isin(books["isbn"])
    ex = ex[in_books].merge(users[["user_id", "Age", "AgeGroup", "Country"]], on="user_id", how="left")
    ex["AgeGroup"] = ex["AgeGroup"].fillna(AGE_UNKNOWN)
    ex["Country"] = ex["Country"].fillna("Unknown")

    quality = {
        "rating_duplikat_dibuang": int(n_raw - len(r)),
        "umur_kosong_atau_tidak_valid_pct": float(users["Age"].isna().mean() * 100),
        "tahun_terbit_tidak_valid": int(books["year"].isna().sum()),
        "isbn_rating_tidak_ada_di_buku_pct": float((~r["isbn"].isin(books["isbn"])).mean() * 100),
        "rating_implisit_pct": float((r["rating"] == 0).mean() * 100),
        "negara_unknown": int((users["Country"] == "Unknown").sum()),
    }
    return Data(users=users, books=books, ratings=r, explicit=ex, quality=quality)


# =====================================================================
# 2. MODEL
# =====================================================================
@dataclass
class Model:
    items: pd.DataFrame        # metadata + n_ratings, avg_rating, wr (per item)
    user_ids: np.ndarray
    R: Any                     # users x items (rating mentah)
    Rc: Any                    # rating dipusatkan per pengguna
    Xu: Any                    # users x items, L2-normalized (Rc)
    Xi: Any                    # items x users, L2-normalized (Rc.T)
    Bu: Any                    # biner users x items
    Bi: Any                    # biner items x users
    user_mean: np.ndarray
    global_mean: float
    V: Any                     # items x k  (faktor laten)
    Vt: Any                    # k x items
    tfidf: Any                 # items x terms (L2-normalized)
    liked: Any                 # biner (rating >= LIKE_THR), csr
    liked_csc: Any
    item_liked: np.ndarray
    uidx: pd.Index
    iidx: pd.Index


def _soup(items: pd.DataFrame) -> pd.Series:
    def tok(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    au = "au_" + items["author"].map(tok)
    pb = "pb_" + items["publisher"].map(tok)
    dc = "dc_" + items["decade"].fillna(0).astype(int).astype(str)
    return items["title"].str.lower() + " " + au + " " + au + " " + pb + " " + dc


def _filter_min(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    for _ in range(6):
        n0 = len(df)
        df = df[df.groupby("user_id")["isbn"].transform("size") >= min_user]
        df = df[df.groupby("isbn")["user_id"].transform("size") >= min_item]
        if len(df) == n0:
            break
    return df


def model_from_matrix(R: sparse.csr_matrix, meta: pd.DataFrame, user_ids: np.ndarray,
                      n_factors: int = 50, tfidf=None) -> Model:
    R = R.tocsr().astype(np.float32)
    n_users, n_items = R.shape
    cnt_u = np.diff(R.indptr)
    gm = float(R.data.mean())
    user_mean = (np.asarray(R.sum(1)).ravel() + SHRINK_MEAN * gm) / (cnt_u + SHRINK_MEAN)
    user_mean = user_mean.astype(np.float32)

    coo = R.tocoo()
    Rc = sparse.csr_matrix((coo.data - user_mean[coo.row], (coo.row, coo.col)), shape=R.shape)
    B = R.copy()
    B.data = np.ones_like(B.data)

    Xu = normalize(Rc)
    Xi = normalize(Rc.T.tocsr())
    Bu, Bi = B, B.T.tocsr()

    k = max(2, min(n_factors, min(R.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=42).fit(Rc)
    Vt = svd.components_.astype(np.float32)

    items = meta.reset_index(drop=True).copy()
    n_r = np.asarray(B.sum(0)).ravel()
    items["n_ratings"] = n_r.astype(int)
    items["avg_rating"] = np.where(n_r > 0, np.asarray(R.sum(0)).ravel() / np.maximum(n_r, 1), np.nan)
    m_votes = max(5.0, float(np.quantile(n_r[n_r > 0], 0.75)))
    items["wr"] = (n_r / (n_r + m_votes)) * items["avg_rating"].fillna(gm) + (m_votes / (n_r + m_votes)) * gm

    if tfidf is None:
        vec = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
        tfidf = vec.fit_transform(_soup(items)).astype(np.float32).tocsr()

    L = R.copy()
    L.data = (L.data >= LIKE_THR).astype(np.float32)
    L.eliminate_zeros()

    return Model(items=items, user_ids=np.asarray(user_ids), R=R, Rc=Rc, Xu=Xu, Xi=Xi, Bu=Bu, Bi=Bi,
                 user_mean=user_mean, global_mean=gm, V=Vt.T.copy(), Vt=Vt, tfidf=tfidf,
                 liked=L, liked_csc=L.tocsc(), item_liked=np.asarray(L.sum(0)).ravel(),
                 uidx=pd.Index(user_ids), iidx=pd.Index(items["isbn"]))


def build_model(data: Data, min_user: int = 5, min_item: int = 5, n_factors: int = 50) -> Model:
    df = _filter_min(data.explicit[["user_id", "isbn", "rating"]], min_user, min_item)
    users = np.sort(df["user_id"].unique())
    isbns = np.sort(df["isbn"].unique())
    rows = pd.Index(users).get_indexer(df["user_id"])
    cols = pd.Index(isbns).get_indexer(df["isbn"])
    R = sparse.csr_matrix((df["rating"].values.astype(np.float32), (rows, cols)),
                          shape=(len(users), len(isbns)))
    meta = data.books.set_index("isbn").reindex(isbns).reset_index()
    return model_from_matrix(R, meta, users, n_factors=n_factors)


# =====================================================================
# 3. UTILITAS
# =====================================================================
def _mu(m: Model, r) -> float:
    """Rata-rata pengguna yang di-shrink ke rata-rata global."""
    r = np.asarray(r, dtype=np.float32)
    return float((r.sum() + SHRINK_MEAN * m.global_mean) / (len(r) + SHRINK_MEAN))


def user_vec(m: Model, user_id) -> tuple[np.ndarray, np.ndarray, int]:
    u = m.uidx.get_loc(user_id)
    s, e = m.R.indptr[u], m.R.indptr[u + 1]
    return m.R.indices[s:e].copy(), m.R.data[s:e].astype(np.float32).copy(), u


def vec_from_isbn(m: Model, ratings: dict) -> tuple[np.ndarray, np.ndarray]:
    """ratings = {isbn: rating} -> (idx, r) untuk pengguna baru (cold-start)."""
    pairs = [(m.iidx.get_loc(k), float(v)) for k, v in ratings.items() if k in m.iidx]
    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=np.float32)
    idx, r = zip(*pairs)
    return np.array(idx, dtype=int), np.array(r, dtype=np.float32)


def top_n(m: Model, scores, exclude, n: int = 10) -> pd.DataFrame:
    s = np.array(scores, dtype=float)
    s[~np.isfinite(s)] = -np.inf
    exclude = np.asarray(exclude, dtype=int)
    if len(exclude):
        s[exclude] = -np.inf
    k = min(n, len(s))
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.argsort(-s[idx])]
    idx = idx[np.isfinite(s[idx])]
    out = m.items.iloc[idx][["isbn", "title", "author", "year", "publisher",
                             "n_ratings", "avg_rating", "image"]].copy()
    out.insert(0, "score", s[idx])
    out.insert(0, "item_idx", idx)
    return out.reset_index(drop=True)


# =====================================================================
# 4. RULE-BASED
# =====================================================================
def popular_books(explicit: pd.DataFrame, books: pd.DataFrame, country=None, age_group=None,
                  year_range=None, min_votes: int = 5, topn: int = 10) -> pd.DataFrame:
    """Data/Domain rule: IF negara=X AND kelompok usia=Y THEN buku dengan weighted rating tertinggi."""
    d = explicit
    if country and country != "Semua":
        d = d[d["Country"] == country]
    if age_group and age_group != "Semua":
        d = d[d["AgeGroup"] == age_group]
    if d.empty:
        return pd.DataFrame()
    C = d["rating"].mean()
    g = d.groupby("isbn")["rating"].agg(["count", "mean"]).reset_index()
    g = g[g["count"] >= min_votes]
    if g.empty:
        return pd.DataFrame()
    g["wr"] = (g["count"] / (g["count"] + min_votes)) * g["mean"] + (min_votes / (g["count"] + min_votes)) * C
    g = g.merge(books[["isbn", "title", "author", "year", "publisher", "image"]], on="isbn", how="left")
    if year_range is not None:
        lo, hi = year_range
        g = g[(g["year"] >= lo) & (g["year"] <= hi)]
    g = g.sort_values("wr", ascending=False).head(topn)
    return g.rename(columns={"count": "n_ratings", "mean": "avg_rating", "wr": "score"}).reset_index(drop=True)


def association_rules(m: Model, basket_idx, min_count: int = 3, topn: int = 10,
                      sort_by: str = "lift") -> pd.DataFrame:
    """Aturan {basket} -> {buku X} pada matriks 'suka' (rating >= 7).
    support = n(basket U X)/N ; confidence = n(basket U X)/n(basket) ; lift = conf / P(X)."""
    basket_idx = np.asarray(basket_idx, dtype=int)
    if len(basket_idx) == 0:
        return pd.DataFrame()
    N = m.liked.shape[0]
    has_all = np.asarray(m.liked_csc[:, basket_idx].sum(1)).ravel() == len(basket_idx)
    users = np.where(has_all)[0]
    n_b = len(users)
    if n_b == 0:
        return pd.DataFrame()
    co = np.asarray(m.liked[users].sum(0)).ravel()
    co[basket_idx] = 0
    keep = np.where(co >= min_count)[0]
    if len(keep) == 0:
        return pd.DataFrame()
    conf = co[keep] / n_b
    supp = co[keep] / N
    p_x = m.item_liked[keep] / N
    lift = np.where(p_x > 0, conf / p_x, 0.0)
    out = m.items.iloc[keep][["isbn", "title", "author", "year"]].copy()
    out["n_pengguna_basket"] = n_b
    out["n_bersama"] = co[keep].astype(int)
    out["support"], out["confidence"], out["lift"] = supp, conf, lift
    return out.sort_values([sort_by, "confidence"], ascending=False).head(topn).reset_index(drop=True)


# =====================================================================
# 5. CONTENT-BASED
# =====================================================================
def _liked_subset(idx, r):
    if len(idx) == 0:
        return idx, r
    sel = r >= LIKE_THR
    if not sel.any():
        sel = r >= r.mean()
    return idx[sel], r[sel]


def score_cb(m: Model, idx, r) -> np.ndarray:
    """User profile = rata-rata tertimbang vektor TF-IDF buku yang disukai; skor = cosine ke semua buku."""
    li, lr = _liked_subset(np.asarray(idx), np.asarray(r))
    if len(li) == 0:
        return np.zeros(m.tfidf.shape[0])
    prof = sparse.csr_matrix(lr.reshape(1, -1)) @ m.tfidf[li]
    prof = normalize(prof)
    return (m.tfidf @ prof.T).toarray().ravel()


def score_cb_knn(m: Model, idx, r, damp: float = 0.3) -> np.ndarray:
    """Prediksi rating berbasis kemiripan konten (dipakai untuk evaluasi RMSE)."""
    idx, r = np.asarray(idx), np.asarray(r, dtype=np.float32)
    if len(idx) == 0:
        return np.full(m.tfidf.shape[0], np.nan)
    mu = _mu(m, r)
    S = (m.tfidf @ m.tfidf[idx].T).toarray()
    num, den = S @ (r - mu), np.abs(S).sum(1)
    return np.where(den > 1e-9, mu + num / (den + damp), np.nan)


def explain_cb(m: Model, rec_idx, idx, r) -> list[tuple[str, float]]:
    li, _ = _liked_subset(np.asarray(idx), np.asarray(r))
    if len(li) == 0 or len(rec_idx) == 0:
        return [("", 0.0)] * len(rec_idx)
    S = (m.tfidf[np.asarray(rec_idx)] @ m.tfidf[li].T).toarray()
    j = S.argmax(1)
    return [(m.items.iloc[li[a]]["title"], float(S[i, a])) for i, a in enumerate(j)]


# =====================================================================
# 6. COLLABORATIVE FILTERING
# =====================================================================
def score_item_cf(m: Model, idx, r, shrink: float = 10, damp: float = 0.3, max_ref: int = 150):
    """Memory-based, Item-Based CF (adjusted cosine + shrinkage). Return prediksi rating."""
    idx, r = np.asarray(idx), np.asarray(r, dtype=np.float32)
    if len(idx) == 0:
        return np.full(m.Xi.shape[0], np.nan)
    mu = _mu(m, r)
    rc = r - mu
    if len(idx) > max_ref:
        sel = np.argsort(-np.abs(rc))[:max_ref]
        idx, rc = idx[sel], rc[sel]
    S = (m.Xi @ m.Xi[idx].T).toarray()
    C = (m.Bi @ m.Bi[idx].T).toarray()
    S *= C / (C + shrink)
    den = np.abs(S).sum(1)
    pred = mu + (S @ rc) / (den + damp)
    return np.where(den > 1e-9, pred, np.nan)


def score_user_cf(m: Model, idx, r, k: int = 50, shrink: float = 10, damp: float = 0.3, self_idx=None):
    """Memory-based, User-Based CF (tetangga terdekat). Return prediksi rating."""
    idx, r = np.asarray(idx), np.asarray(r, dtype=np.float32)
    n_items = m.Xu.shape[1]
    if len(idx) == 0:
        return np.full(n_items, np.nan)
    mu = _mu(m, r)
    v = sparse.csr_matrix((r - mu, (np.zeros(len(idx), dtype=int), idx)), shape=(1, n_items))
    b = sparse.csr_matrix((np.ones(len(idx), dtype=np.float32), (np.zeros(len(idx), dtype=int), idx)),
                          shape=(1, n_items))
    sims = (m.Xu @ normalize(v).T).toarray().ravel()
    co = (m.Bu @ b.T).toarray().ravel()
    sims = sims * co / (co + shrink)
    if self_idx is not None:
        sims[self_idx] = 0
    kk = min(k, len(sims) - 1)
    nb = np.argpartition(-sims, kk)[:kk]
    nb = nb[sims[nb] > 0]
    if len(nb) == 0:
        return np.full(n_items, np.nan)
    w = sims[nb]
    num = m.Rc[nb].T @ w
    den = m.Bu[nb].T @ np.abs(w)
    pred = mu + num / (den + damp)
    return np.where(den > 1e-9, pred, np.nan)


def score_svd(m: Model, idx, r) -> np.ndarray:
    """Model-based CF (Matrix Factorization / SVD): fold-in vektor pengguna ke ruang faktor laten."""
    idx, r = np.asarray(idx), np.asarray(r, dtype=np.float32)
    n_items = m.Vt.shape[1]
    if len(idx) == 0:
        return np.full(n_items, np.nan)
    mu = _mu(m, r)
    v = sparse.csr_matrix((r - mu, (np.zeros(len(idx), dtype=int), idx)), shape=(1, n_items))
    z = v @ m.V                      # (1 x k) profil laten pengguna
    return mu + np.asarray(z @ m.Vt).ravel()


def cf_scores(m: Model, method: str, idx, r, self_idx=None) -> np.ndarray:
    if method == "Item-Based":
        return score_item_cf(m, idx, r)
    if method == "User-Based":
        return score_user_cf(m, idx, r, self_idx=self_idx)
    return score_svd(m, idx, r)


def similar_items(m: Model, item_idx: int, method: str = "Konten", n: int = 10, shrink: float = 10):
    if method == "Konten":
        s = (m.tfidf @ m.tfidf[item_idx].T).toarray().ravel()
    else:
        s = (m.Xi @ m.Xi[item_idx].T).toarray().ravel()
        c = (m.Bi @ m.Bi[item_idx].T).toarray().ravel()
        s = s * c / (c + shrink)
    return top_n(m, s, [item_idx], n)


# =====================================================================
# 7. HYBRID
# =====================================================================
def _pct_rank(x: np.ndarray, exclude) -> np.ndarray:
    cand = np.ones(len(x), dtype=bool)
    cand[np.asarray(exclude, dtype=int)] = False
    v = np.nan_to_num(x[cand], nan=-1e9, posinf=-1e9, neginf=-1e9)
    out = np.full(len(x), -np.inf)
    out[cand] = rankdata(v) / max(len(v), 1)
    return out


def hybrid_weighted(m: Model, idx, r, alpha: float = 0.5, cf_method: str = "SVD", self_idx=None):
    """skor = alpha * rank(CB) + (1-alpha) * rank(CF)   (normalisasi persentil agar skala sebanding)."""
    cb = score_cb(m, idx, r)
    cf = cf_scores(m, cf_method, idx, r, self_idx)
    return alpha * _pct_rank(cb, idx) + (1 - alpha) * _pct_rank(cf, idx)


def hybrid_switching(m: Model, idx, r, threshold: int = 5, cf_method: str = "SVD", self_idx=None):
    """0 rating -> Rule-Based (popularitas) ; < threshold -> Content-Based ; >= threshold -> CF."""
    n = len(idx)
    if n == 0:
        return "Rule-Based (Popularitas)", m.items["wr"].values.copy()
    if n < threshold:
        return "Content-Based", score_cb(m, idx, r)
    return f"Collaborative ({cf_method})", cf_scores(m, cf_method, idx, r, self_idx)


# =====================================================================
# 8. EVALUASI
# =====================================================================
def evaluate(base: Model, n_users: int = 300, test_frac: float = 0.2, k: int = 10, alpha: float = 0.5,
             min_user_ratings: int = 10, seed: int = 42, progress=None):
    """Hold-out per pengguna: sebagian rating disembunyikan, model dilatih ulang pada sisa rating."""
    rng = np.random.default_rng(seed)
    R = base.R.tocsr()
    n_items = R.shape[1]
    cnt = np.diff(R.indptr)
    cand = np.where(cnt >= min_user_ratings)[0]
    if len(cand) == 0:
        return pd.DataFrame(), {}
    users = rng.choice(cand, size=min(n_users, len(cand)), replace=False)

    test = {}
    for u in users:
        cols = R.indices[R.indptr[u]:R.indptr[u + 1]]
        nt = max(1, int(round(len(cols) * test_frac)))
        tc = rng.choice(cols, size=nt, replace=False)
        test[u] = (tc, np.asarray(R[u, tc].todense()).ravel())

    coo = R.tocoo()
    key = coo.row.astype(np.int64) * n_items + coo.col
    tkeys = np.concatenate([u * n_items + test[u][0].astype(np.int64) for u in users])
    keep = ~np.isin(key, tkeys)
    Rtr = sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=R.shape)
    meta = base.items[["isbn", "title", "author", "year", "publisher", "image", "decade"]]
    mt = model_from_matrix(Rtr, meta, base.user_ids, n_factors=base.Vt.shape[0], tfidf=base.tfidf)

    names = ["Rule-Based (Popularitas)", "Content-Based", "CF Item-Based", "CF User-Based",
             "CF SVD (Model-Based)", f"Hybrid Weighted (a={alpha:.1f})"]
    err = {n: [] for n in names}
    cover = {n: 0 for n in names}
    rk = {n: {"p": [], "r": [], "h": []} for n in names}
    n_pairs, item_mean = 0, np.where(np.isnan(mt.items["avg_rating"].values), mt.global_mean,
                                     mt.items["avg_rating"].values)

    for step, u in enumerate(users):
        idx, r, _ = user_vec(mt, base.user_ids[u])
        tc, tr = test[u]
        mu = _mu(mt, r)
        cb_rank = score_cb(mt, idx, r)
        svd = score_svd(mt, idx, r)
        cbk = score_cb_knn(mt, idx, r)
        itm = score_item_cf(mt, idx, r)
        usr = score_user_cf(mt, idx, r, self_idx=u)
        hyb_pred = np.where(np.isnan(cbk), svd, alpha * np.nan_to_num(cbk, nan=0) + (1 - alpha) * svd)
        rating_pred = {names[0]: item_mean, names[1]: cbk, names[2]: itm, names[3]: usr,
                       names[4]: svd, names[5]: hyb_pred}
        rank_score = {names[0]: mt.items["wr"].values, names[1]: cb_rank, names[2]: itm, names[3]: usr,
                      names[4]: svd,
                      names[5]: alpha * _pct_rank(cb_rank, idx) + (1 - alpha) * _pct_rank(svd, idx)}
        rel = set(tc[tr >= LIKE_THR].tolist())
        n_pairs += len(tc)
        for n in names:
            p = rating_pred[n][tc]
            cover[n] += int(np.isfinite(p).sum())
            p = np.clip(np.where(np.isfinite(p), p, mu), 1, 10)
            err[n].extend((p - tr).tolist())
            if rel:
                s = np.array(rank_score[n], dtype=float)
                s[~np.isfinite(s)] = -np.inf
                s[idx] = -np.inf
                top = np.argpartition(-s, k - 1)[:k]
                hits = len(rel.intersection(top.tolist()))
                rk[n]["p"].append(hits / k)
                rk[n]["r"].append(hits / len(rel))
                rk[n]["h"].append(1.0 if hits > 0 else 0.0)
        if progress is not None:
            progress((step + 1) / len(users))

    rows = []
    for n in names:
        e = np.array(err[n])
        rows.append({"Metode": n, "RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean()),
                     "Coverage (%)": 100 * cover[n] / n_pairs,
                     f"Precision@{k}": float(np.mean(rk[n]["p"])) if rk[n]["p"] else np.nan,
                     f"Recall@{k}": float(np.mean(rk[n]["r"])) if rk[n]["r"] else np.nan,
                     f"HitRate@{k}": float(np.mean(rk[n]["h"])) if rk[n]["h"] else np.nan})
    info = {"n_users": len(users), "n_test_pairs": n_pairs, "n_users_ranking": len(rk[names[0]]["p"]),
            "n_train_ratings": int(Rtr.nnz)}
    return pd.DataFrame(rows), info
