"""
app.py  -  Dashboard Analitik & Sistem Rekomendasi Buku (Streamlit)
Dataset : Book-Crossing (BX-Users.csv, BX-Books.csv, BX-Book-Ratings.csv)
Tema    : Donezo Dashboard (Deep Forest Green, Mint & Off-White Canvas)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import recsys as rs

# 1. Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Sistem Rekomendasi Buku",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Injeksi CSS Sistem Desain "Donezo Dashboard"
st.markdown("""
<style>
/* =========================================================
   SISTEM DESAIN DONEZO: FOREST GREEN, MINT & SOFT CANVAS
   ========================================================= */

/* Latar belakang kanvas abu-abu lembut */
.stApp {
    background-color: #F3F5F7 !important;
    color: #111827 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

.main .block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 3rem !important;
    max-width: 1380px !important;
}

/* Sidebar styling bersih dengan border tipis */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #ECEEF1 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem !important;
}

/* =========================================================
   ELIMINASI TOTAL WARNA MERAH STREAMLIT (TABS, RADIO, SLIDER)
   ========================================================= */

/* Tab Navigation */
button[data-baseweb="tab"] {
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
    padding: 10px 20px !important;
    border: none !important;
    background: transparent !important;
    transition: color 0.15s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: #164E33 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #164E33 !important;
    font-weight: 600 !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #164E33 !important;
    height: 3px !important;
    border-radius: 3px 3px 0 0 !important;
}
div[data-baseweb="tab-border"] {
    background-color: #ECEEF1 !important;
}

/* =========================================================
   TRANSFORMASI SIDEBAR NAVIGATION BAR (GAYA DONEZO MENU)
   ========================================================= */

/* Sembunyikan lingkaran radio bulat bawaan */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
    width: 100% !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* Setiap item menu di sidebar diubah menjadi tile/tombol bergaya Donezo */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
    width: 100% !important;
    padding: 12px 16px !important;
    border-radius: 12px !important;
    margin-bottom: 6px !important;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
    display: flex !important;
    align-items: center !important;
}

/* Teks label item menu */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label p {
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: #4B5563 !important;
    margin: 0 !important;
    line-height: 1.3 !important;
}

/* Efek hover pada item menu */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background-color: #F3F5F7 !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover p {
    color: #164E33 !important;
}

/* Item Menu yang Aktif (Active State bergaya Donezo Menu) */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
    background-color: #E8F5EE !important;
    border-left: 4px solid #164E33 !important;
    border-radius: 4px 12px 12px 4px !important;
    box-shadow: 0 1px 3px rgba(22, 78, 51, 0.08) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] p {
    color: #164E33 !important;
    font-weight: 600 !important;
}

/* Radio button reguler (di luar sidebar) */
div[data-testid="stRadio"]:not(section[data-testid="stSidebar"] div[data-testid="stRadio"]) label[data-checked="true"] > div:first-child {
    background-color: #164E33 !important;
    border-color: #164E33 !important;
}
div[data-testid="stRadio"]:not(section[data-testid="stSidebar"] div[data-testid="stRadio"]) label div:first-child:hover {
    border-color: #226E4D !important;
}

/* Header bar Streamlit agar menyatu */
header[data-testid="stHeader"] {
    background-color: #F3F5F7 !important;
}

/* Slider Track & Thumb */
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
    background-color: #164E33 !important;
    border-color: #164E33 !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"] {
    background-color: #164E33 !important;
}

/* Checkbox */
div[data-testid="stCheckbox"] label span:first-child[aria-checked="true"] {
    background-color: #164E33 !important;
    border-color: #164E33 !important;
}

/* Tombol Kapsul (Pill Buttons) */
.stButton > button {
    border-radius: 9999px !important;
    font-weight: 500 !important;
    padding: 6px 20px !important;
    border: 1px solid #D1D5DB !important;
    background-color: #FFFFFF !important;
    color: #111827 !important;
    transition: all 0.15s ease-in-out !important;
}
.stButton > button:hover {
    border-color: #164E33 !important;
    color: #164E33 !important;
    background-color: #E8F5EE !important;
    box-shadow: 0 1px 3px rgba(22, 78, 51, 0.1) !important;
}

/* =========================================================
   KARTU STATISTIK GAYA DONEZO (16px BORDER-RADIUS)
   ========================================================= */

/* Hero Stat Card (Gelap - Forest Green) */
.hero-stat-card {
    background-color: #164E33;
    border-radius: 16px;
    padding: 20px 22px;
    color: #FFFFFF;
    position: relative;
    box-shadow: 0 4px 12px rgba(22, 78, 51, 0.18);
    margin-bottom: 14px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.hero-stat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.hero-stat-label {
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #A7D7BC;
}
.hero-stat-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background-color: rgba(255, 255, 255, 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    color: #FFFFFF;
}
.hero-stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.15;
    margin-bottom: 8px;
}
.hero-stat-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    background-color: rgba(255, 255, 255, 0.16);
    color: #FFFFFF;
    font-size: 0.72rem;
    font-weight: 500;
}

/* Regular Stat Card (Putih Bersih 16px) */
.reg-stat-card {
    background-color: #FFFFFF;
    border: 1px solid #ECEEF1;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    margin-bottom: 14px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.reg-stat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.reg-stat-label {
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #6B7280;
}
.reg-stat-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background-color: #F3F5F7;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    color: #6B7280;
}
.reg-stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.15;
    margin-bottom: 8px;
}
.reg-stat-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    background-color: #E8F5EE;
    color: #164E33;
    font-size: 0.72rem;
    font-weight: 600;
    border: 1px solid #D1EAD9;
}

/* Sub KPI Card (Banner Horizontal) */
.sub-stat-card {
    background-color: #FFFFFF;
    border: 1px solid #ECEEF1;
    border-radius: 12px;
    padding: 12px 18px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    margin-bottom: 14px;
}
.sub-stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6B7280;
}
.sub-stat-value {
    font-size: 0.96rem;
    font-weight: 600;
    color: #111827;
    margin-top: 3px;
}

/* =========================================================
   KARTU BUKU (BOOK CARDS) - 16px BORDER-RADIUS
   ========================================================= */
.book-card-item {
    background-color: #FFFFFF;
    border: 1px solid #ECEEF1;
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    margin-bottom: 16px;
}
.book-card-item:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 18px -4px rgba(22, 78, 51, 0.12);
    border-color: #226E4D;
}
.pill-score {
    display: inline-block;
    padding: 3px 10px;
    font-size: 0.74rem;
    font-weight: 600;
    border-radius: 9999px;
    background-color: #E8F5EE;
    color: #164E33;
    border: 1px solid #D1EAD9;
}

/* Kontainer umum berbatas */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #ECEEF1 !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"
TPL = "plotly_white"
PRIMARY_GREEN = "#164E33"
MID_GREEN = "#226E4D"
SAGE_GREEN = "#4E9B76"
PAGES = [
    "Analisis Data & Eksplorasi",
    "Sistem Rekomendasi",
    "Evaluasi Model",
    "Metodologi & Konfigurasi"
]

_fig_counter = [0]


def show(fig, height=380):
    """Tampilkan figure Plotly dengan konfigurasi margin & palet Donezo."""
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        font=dict(family="sans-serif", size=12, color="#111827"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#ECEEF1")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#ECEEF1")
    _fig_counter[0] += 1
    st.plotly_chart(fig, use_container_width=True, key=f"fig_{_fig_counter[0]}")


# =====================================================================
# DATA & MODEL (CACHING)
# =====================================================================
def ensure_data():
    """Validasi keberadaan file CSV di folder data/."""
    missing = [f for f in rs.FILES.values() if not os.path.exists(os.path.join(DATA_DIR, f))]
    if not missing:
        return
    st.title("Dashboard Sistem Rekomendasi Buku")
    st.warning("File dataset belum ditemukan di folder `data/`. Silakan upload ketiga file CSV Book-Crossing di bawah ini.")
    ups = {f: st.file_uploader(f, type="csv", key=f"up_{f}") for f in missing}
    if all(ups.values()):
        os.makedirs(DATA_DIR, exist_ok=True)
        for f, up in ups.items():
            with open(os.path.join(DATA_DIR, f), "wb") as fh:
                fh.write(up.getbuffer())
        st.rerun()
    st.stop()


@st.cache_resource(show_spinner="Memuat dan memproses data...")
def get_data():
    return rs.load_data(DATA_DIR)


@st.cache_resource(show_spinner="Membangun matriks dan model rekomendasi...")
def get_model(min_user: int, min_item: int, n_factors: int):
    return rs.build_model(get_data(), min_user, min_item, n_factors)


@st.cache_resource(show_spinner=False)
def item_stats():
    d = get_data()
    g = d.explicit.groupby("isbn")["rating"].agg(n_ratings="count", avg_rating="mean").reset_index()
    return g.merge(d.books[["isbn", "title", "author", "year", "decade", "publisher"]], on="isbn", how="left")


def short(s, n=45):
    return s.astype(str).str.slice(0, n)


# =====================================================================
# KOMPONEN KARTU METRIK & KARTU BUKU (DONEZO STYLE)
# =====================================================================
def render_hero_kpi(label: str, value: str, badge: str):
    """Render kartu metrik utama berwarna Forest Green gelap (seperti 'Total Projects' pada Donezo)."""
    st.markdown(
        f'<div class="hero-stat-card">'
        f'<div class="hero-stat-header">'
        f'<div class="hero-stat-label">{label}</div>'
        f'<div class="hero-stat-icon">↗</div>'
        f'</div>'
        f'<div class="hero-stat-value">{value}</div>'
        f'<div><span class="hero-stat-badge">{badge}</span></div>'
        f'</div>',
        unsafe_allow_html=True
    )


def render_reg_kpi(label: str, value: str, badge: str):
    """Render kartu metrik reguler berwarna putih dengan sudut tumpul 16px."""
    st.markdown(
        f'<div class="reg-stat-card">'
        f'<div class="reg-stat-header">'
        f'<div class="reg-stat-label">{label}</div>'
        f'<div class="reg-stat-icon">↗</div>'
        f'</div>'
        f'<div class="reg-stat-value">{value}</div>'
        f'<div><span class="reg-stat-badge">{badge}</span></div>'
        f'</div>',
        unsafe_allow_html=True
    )


def render_sub_kpi(label: str, text: str):
    """Render kartu info sekunder berbentuk kapsul rapi."""
    st.markdown(
        f'<div class="sub-stat-card">'
        f'<div class="sub-stat-label">{label}</div>'
        f'<div class="sub-stat-value">{text}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def render_book_cards(df: pd.DataFrame, covers: bool = True, score_label: str = "Skor",
                      score_fmt: str = "%.2f", extra: list[str] | None = None, n_cols: int = 4):
    """Render daftar buku dalam format grid kartu Donezo 16px."""
    if df is None or len(df) == 0:
        st.info("Tidak ada buku yang sesuai dengan kriteria filter.")
        return

    records = df.to_dict(orient="records")
    for i in range(0, len(records), n_cols):
        batch = records[i:i + n_cols]
        cols = st.columns(n_cols)
        for j, row in enumerate(batch):
            with cols[j]:
                img_url = str(row.get("image", "")).strip()
                safe_url = img_url.replace("http://", "https://") if (img_url and img_url.startswith("http")) else ""
                img_html = (
                    f'<div style="text-align:center; height:160px; background:#F8FAFC; border-radius:10px; margin-bottom:10px; display:flex; align-items:center; justify-content:center; overflow:hidden; border:1px solid #ECEEF1;">'
                    f'<img src="{safe_url}" style="max-height:100%; max-width:100%; object-fit:contain;" '
                    f'onerror="this.onerror=null; this.src=\'https://placehold.co/110x150/f8fafc/6b7280?text=Katalog\';">'
                    f'</div>'
                ) if covers else ""

                title = str(row.get("title", "Judul Tidak Diketahui"))
                author = str(row.get("author", "Penulis Tidak Diketahui"))
                year = row.get("year", None)
                year_str = f"{int(year)}" if pd.notna(year) and year else "-"

                score_val = row.get("score", None)
                score_html = ""
                if score_val is not None and pd.notna(score_val):
                    score_formatted = score_fmt % score_val
                    score_html = f'<span class="pill-score">{score_label}: {score_formatted}</span>'

                extra_html = ""
                if extra:
                    for ex_col in extra:
                        if ex_col in row and pd.notna(row[ex_col]):
                            extra_html += f'<div style="font-size:0.72rem; color:#4B5563; background:#F9FAFB; border:1px solid #E5E7EB; border-radius:6px; padding:4px 8px; margin-top:8px; line-height:1.25;">{row[ex_col]}</div>'

                card_html = (
                    f'<div class="book-card-item">'
                    f'{img_html}'
                    f'<div style="font-weight:600; font-size:0.88rem; line-height:1.3; color:#111827; height:2.5rem; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; margin-bottom:4px;" title="{title}">{title}</div>'
                    f'<div style="font-size:0.78rem; color:#6B7280; margin-bottom:8px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{author} ({year_str})</div>'
                    f'<div>{score_html}</div>'
                    f'{extra_html}'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)


def show_recs(df: pd.DataFrame, covers: bool = True, extra: list[str] | None = None,
              score_label: str = "Skor", score_fmt: str = "%.3f", key: str | None = None,
              default_view: str = "Kartu Visual"):
    """Tampilkan hasil rekomendasi dengan toggle antara Kartu Visual atau Tabel Data."""
    if df is None or len(df) == 0:
        st.info("Tidak ada rekomendasi yang memenuhi kriteria filter.")
        return

    c_left, c_right = st.columns([3, 1])
    view_key = f"view_{key or 'default'}_{_fig_counter[0]}"
    view_mode = c_right.radio(
        "Format Tampilan",
        ["Kartu Visual", "Tabel Data"],
        index=0 if default_view == "Kartu Visual" else 1,
        horizontal=True,
        key=view_key,
        label_visibility="collapsed"
    )

    if view_mode == "Kartu Visual":
        render_book_cards(df, covers=covers, score_label=score_label, score_fmt=score_fmt, extra=extra)
    else:
        out = df.copy()
        out.insert(0, "No", range(1, len(out) + 1))
        cols = ["No", "title", "author", "year", "publisher", "n_ratings", "avg_rating", "score"] + (extra or [])
        cols = [c for c in cols if c in out.columns]
        cfg = {
            "title": st.column_config.TextColumn("Judul Buku", width="large"),
            "author": st.column_config.TextColumn("Penulis"),
            "year": st.column_config.NumberColumn("Tahun Terbit", format="%d"),
            "publisher": st.column_config.TextColumn("Penerbit"),
            "n_ratings": st.column_config.NumberColumn("Jumlah Rating", format="%d"),
            "avg_rating": st.column_config.NumberColumn("Rata-rata Rating", format="%.2f"),
            "score": st.column_config.NumberColumn(score_label, format=score_fmt)
        }
        if covers and "image" in out.columns:
            cols.insert(1, "image")
            cfg["image"] = st.column_config.ImageColumn("Sampul")
        st.dataframe(out[cols], hide_index=True, column_config=cfg, use_container_width=True)


# =====================================================================
# MODUL 1 : ANALISIS DATA & EKSPLORASI
# =====================================================================
def page_analytics(d, m):
    # Header bergaya Donezo
    st.title("Analisis Data & Eksplorasi")
    st.caption("Eksplorasi profil pengguna, karakteristik katalog buku, dan pola distribusi rating interaksi.")

    ex = d.explicit
    n_u, n_i = ex["user_id"].nunique(), ex["isbn"].nunique()

    # Baris Kartu Metrik: 1 Hero Card (Hijau Gelap) + 3 Regular Cards (Putih 16px)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_hero_kpi("TOTAL INTERAKSI", f"{len(d.ratings):,}", f"{len(ex):,} Eksplisit (1–10)")
    with kpi2:
        render_reg_kpi("PENGGUNA TERDAFTAR", f"{len(d.users):,}", f"{n_u:,} Pengguna Aktif")
    with kpi3:
        render_reg_kpi("KATALOG BUKU", f"{len(d.books):,}", f"{n_i:,} Judul Aktif Rating")
    with kpi4:
        render_reg_kpi("RATA-RATA SKOR", f"{ex['rating'].mean():.2f}", f"Median/User: {ex.groupby('user_id').size().median():.0f}")

    # Banner Info Sekunder (3 Kolom)
    sub_kpi1, sub_kpi2, sub_kpi3 = st.columns(3)
    with sub_kpi1:
        render_sub_kpi("INTERAKSI IMPLISIT (SKOR 0)", f"{d.quality['rating_implisit_pct']:.1f}% dari total interaksi")
    with sub_kpi2:
        render_sub_kpi("KEPADATAN MATRIKS USER × BUKU", f"{100 * len(ex) / (n_u * n_i):.4f}% (Extreme Sparsity)")
    with sub_kpi3:
        render_sub_kpi("MATRIKS MODEL REKOMENDASI", f"{m.R.shape[0]:,} pengguna × {m.R.shape[1]:,} buku")

    # Tab Navigasi Analisis (Aksen Forest Green)
    t_user, t_book, t_rating = st.tabs([
        "Demografi Pengguna",
        "Katalog Buku & Penulis",
        "Karakteristik Rating & Kualitas Data"
    ])

    # TAB 1: DEMOGRAFI PENGGUNA
    with t_user:
        u = d.users
        c1, c2 = st.columns(2)
        with c1:
            fig_age = px.histogram(
                u.dropna(subset=["Age"]), x="Age", nbins=48, template=TPL,
                color_discrete_sequence=[PRIMARY_GREEN],
                title=f"Distribusi Usia Pengguna (Data tersedia pada {100 - d.quality['umur_kosong_atau_tidak_valid_pct']:.0f}% profil)",
                labels={"Age": "Usia", "count": "Jumlah Pengguna"}
            )
            show(fig_age)
        with c2:
            ag = u["AgeGroup"].value_counts().reindex(rs.AGE_ORDER).reset_index()
            ag.columns = ["Kelompok Usia", "Pengguna"]
            fig_ag = px.bar(
                ag, x="Kelompok Usia", y="Pengguna", template=TPL,
                color_discrete_sequence=[PRIMARY_GREEN],
                title="Jumlah Pengguna per Kelompok Usia"
            )
            show(fig_ag)

        c1, c2 = st.columns(2)
        with c1:
            cc = u["Country"].value_counts().head(15).reset_index()
            cc.columns = ["Negara", "Pengguna"]
            fig_cc = px.bar(
                cc.sort_values("Pengguna"), x="Pengguna", y="Negara", orientation="h",
                template=TPL, color_discrete_sequence=[PRIMARY_GREEN],
                title="15 Negara dengan Pengguna Terbanyak"
            )
            show(fig_cc, height=420)
        with c2:
            min_n = st.slider("Minimal jumlah rating per negara", 100, 5000, 500, step=100, key="sl_c_min")
            cr = ex.groupby("Country")["rating"].agg(rata2="mean", jumlah="count").reset_index()
            cr = cr[(cr["jumlah"] >= min_n) & (cr["Country"] != "Unknown")].nlargest(15, "rata2")
            fig_cr = px.bar(
                cr.sort_values("rata2"), x="rata2", y="Country", orientation="h",
                template=TPL, color_discrete_sequence=[PRIMARY_GREEN],
                hover_data=["jumlah"], title="Rata-rata Rating Tertinggi per Negara",
                labels={"rata2": "Rata-rata Rating", "Country": "Negara"}, range_x=[5, 9]
            )
            show(fig_cr, height=420)

        top8 = ex[ex["Country"] != "Unknown"]["Country"].value_counts().head(8).index
        pv = ex[ex["Country"].isin(top8)].pivot_table(index="AgeGroup", columns="Country", values="rating", aggfunc="mean")
        pv = pv.reindex([a for a in rs.AGE_ORDER if a in pv.index])
        fig_heat = px.imshow(
            pv, text_auto=".2f", aspect="auto",
            color_continuous_scale=["#F4F9F6", "#D1EAD9", "#78BD99", "#3D8C67", "#164E33", "#0B2E1E"],
            template=TPL, title="Heatmap Rata-rata Rating: Kelompok Usia × 8 Negara Terbesar"
        )
        show(fig_heat, height=400)
        st.caption("Pola variasi rating antar segmen geografis dan demografis menjadi basis aturan pada model rekomendasi berbasis aturan (Rule-Based).")

    # TAB 2: KATALOG BUKU & PENULIS
    with t_book:
        b = d.books
        st_ = item_stats()

        c1, c2 = st.columns(2)
        with c1:
            yc = b.dropna(subset=["year"])
            yc = yc[yc["year"] >= 1950].groupby("year").size().reset_index(name="jumlah")
            fig_yc = px.line(
                yc, x="year", y="jumlah", template=TPL,
                color_discrete_sequence=[PRIMARY_GREEN],
                title="Jumlah Judul Buku per Tahun Terbit (≥1950)",
                labels={"year": "Tahun Terbit", "jumlah": "Jumlah Judul"}
            )
            show(fig_yc)
        with c2:
            tmp = st_.dropna(subset=["decade"])
            tmp = tmp[tmp["decade"] >= 1900].assign(w=lambda x: x["avg_rating"] * x["n_ratings"])
            dec = tmp.groupby("decade").agg(w=("w", "sum"), n=("n_ratings", "sum")).reset_index()
            dec["rata2"] = dec["w"] / dec["n"]
            fig_dec = px.bar(
                dec, x="decade", y="rata2", template=TPL,
                color_discrete_sequence=[PRIMARY_GREEN], hover_data=["n"],
                title="Rata-rata Rating Menurut Dekade Publikasi",
                labels={"decade": "Dekade Terbit", "rata2": "Rata-rata Rating"}, range_y=[5, 9]
            )
            show(fig_dec)

        # Bagian Penulis
        c1, c2 = st.columns(2)
        with c1:
            au = b[b["author"] != "Unknown"]["author"].value_counts().head(15).reset_index()
            au.columns = ["Penulis", "Judul"]
            fig_au = px.bar(
                au.sort_values("Judul"), x="Judul", y="Penulis", orientation="h",
                template=TPL, color_discrete_sequence=[PRIMARY_GREEN],
                title="15 Penulis dengan Jumlah Judul Terbanyak"
            )
            show(fig_au, height=420)
        with c2:
            min_v_author = st.slider("Minimal jumlah rating untuk peringkat penulis", 5, 200, 30, step=5, key="sl_au_min")
            ex_au = ex.merge(b[["isbn", "author"]], on="isbn", how="inner")
            au_stat = ex_au[ex_au["author"] != "Unknown"].groupby("author")["rating"].agg(jumlah="count", rata2="mean").reset_index()
            au_stat = au_stat[au_stat["jumlah"] >= min_v_author]
            au_top = au_stat.nlargest(15, "rata2").sort_values("rata2")
            fig_au_top = px.bar(
                au_top, x="rata2", y="author", orientation="h",
                template=TPL, color_discrete_sequence=[PRIMARY_GREEN], hover_data=["jumlah"],
                title=f"15 Penulis dengan Rata-rata Rating Tertinggi (≥{min_v_author} rating)",
                labels={"rata2": "Rata-rata Rating", "author": "Penulis"}, range_x=[5, 10]
            )
            show(fig_au_top, height=420)

        c1, c2 = st.columns(2)
        with c1:
            top_pop = st_.nlargest(15, "n_ratings").assign(judul=lambda x: short(x["title"]))
            fig_pop = px.bar(
                top_pop.sort_values("n_ratings"), x="n_ratings", y="judul", orientation="h",
                template=TPL, color_discrete_sequence=[PRIMARY_GREEN], hover_data=["author", "avg_rating"],
                title="15 Buku Paling Banyak Diberi Rating",
                labels={"n_ratings": "Jumlah Rating", "judul": "Judul Buku"}
            )
            show(fig_pop, height=440)
        with c2:
            min_v_book = st.slider("Minimal jumlah rating untuk buku terbaik", 5, 200, 50, step=5, key="sl_bk_min")
            best_bk = st_[st_["n_ratings"] >= min_v_book].nlargest(15, "avg_rating").assign(judul=lambda x: short(x["title"]))
            fig_best = px.bar(
                best_bk.sort_values("avg_rating"), x="avg_rating", y="judul", orientation="h",
                template=TPL, color_discrete_sequence=[PRIMARY_GREEN], hover_data=["author", "n_ratings"],
                title=f"15 Buku dengan Rata-rata Rating Tertinggi (≥{min_v_book} rating)",
                labels={"avg_rating": "Rata-rata Rating", "judul": "Judul Buku"}, range_x=[5, 10]
            )
            show(fig_best, height=420)

        sc = st_[st_["n_ratings"] >= 5].sample(min(len(st_[st_["n_ratings"] >= 5]), 4000), random_state=1)
        fig_sc = px.scatter(
            sc, x="n_ratings", y="avg_rating", log_x=True, opacity=0.35, hover_name="title",
            template=TPL, color_discrete_sequence=[PRIMARY_GREEN],
            title="Popularitas vs Kualitas: Jumlah Rating (Skala Log) vs Rata-rata Skor",
            labels={"n_ratings": "Jumlah Rating (Log)", "avg_rating": "Rata-rata Rating"}
        )
        show(fig_sc, height=400)
        st.caption("Katalog dengan jumlah interaksi sedikit rentan terhadap variansi ekstrem. Oleh sebab itu, model rule-based menerapkan pembobotan Bayesian (Weighted Rating).")

    # TAB 3: KARAKTERISTIK RATING & KUALITAS DATA
    with t_rating:
        c1, c2 = st.columns(2)
        with c1:
            vc = d.ratings["rating"].value_counts().sort_index().reset_index()
            vc.columns = ["rating", "jumlah"]
            vc["Kategori"] = np.where(vc["rating"] == 0, "Implisit (0)", "Eksplisit (1–10)")
            fig_vc = px.bar(
                vc, x="rating", y="jumlah", color="Kategori", template=TPL,
                color_discrete_map={"Implisit (0)": "#9CA3AF", "Eksplisit (1–10)": PRIMARY_GREEN},
                title="Distribusi Seluruh Nilai Rating (Termasuk 0 = Implisit)",
                labels={"rating": "Nilai Rating", "jumlah": "Jumlah Interaksi"}
            )
            show(fig_vc)
            st.info("Interaksi bernilai 0 (implisit) mencakup mayoritas data (62%). Pada model prediksi rating, hanya rating eksplisit (1–10) yang dianalisis.")
        with c2:
            vc_ex = ex["rating"].value_counts().sort_index().reset_index()
            vc_ex.columns = ["rating", "jumlah"]
            fig_ex = px.bar(
                vc_ex, x="rating", y="jumlah", template=TPL,
                color_discrete_sequence=[PRIMARY_GREEN],
                title="Distribusi Rating Eksplisit (1–10)",
                labels={"rating": "Skor Rating", "jumlah": "Jumlah"}
            )
            show(fig_ex)
            st.caption(f"Rata-rata rating eksplisit adalah {ex['rating'].mean():.2f} dengan median {ex['rating'].median():.0f}. Distribusi ini menunjukkan kecenderungan pengguna memberi nilai tinggi (Positivity Bias).")

        c1, c2 = st.columns(2)
        with c1:
            uc = ex.groupby("user_id").size()
            fig_uc = px.histogram(
                x=uc.clip(upper=100).values, nbins=50, log_y=True, template=TPL,
                color_discrete_sequence=[PRIMARY_GREEN],
                title="Distribusi Frekuensi Rating per Pengguna (Sumbu-Y Log)",
                labels={"x": "Jumlah Rating per Pengguna", "count": "Jumlah Pengguna"}
            )
            show(fig_uc)
            st.caption(f"Sekitar {100 * (uc == 1).mean():.0f}% pengguna hanya memberikan 1 rating. Hal ini menjadi tantangan cold-start pada sistem rekomendasi.")
        with c2:
            cnt = np.sort(st_["n_ratings"].values)[::-1]
            cum = np.cumsum(cnt) / cnt.sum()
            step = max(1, len(cnt) // 600)
            lt = pd.DataFrame({
                "Persentase Buku (Terpopuler → Jarang)": 100 * np.arange(1, len(cnt) + 1)[::step] / len(cnt),
                "Persentase Kumulatif Rating": 100 * cum[::step]
            })
            fig_lt = px.line(
                lt, x=lt.columns[0], y=lt.columns[1], template=TPL,
                color_discrete_sequence=[PRIMARY_GREEN],
                title="Kurva Long-Tail Popularitas Buku"
            )
            show(fig_lt)
            s10 = 100 * cum[int(0.1 * len(cnt)) - 1]
            st.caption(f"10% buku teratas mengakumulasi {s10:.0f}% dari seluruh interaksi rating (Popularity Bias).")

        st.subheader("Audit Pembersihan Data")
        q = d.quality
        qdf = pd.DataFrame([
            ["Umur tidak valid (<5 atau >100 tahun)", f"{q['umur_kosong_atau_tidak_valid_pct']:.1f}% data pengguna", "Dikonversi menjadi NaN; dikelompokkan ke 'Tidak diketahui'"],
            ["Tahun terbit di luar rentang valid (<1800 atau >2006)", f"{q['tahun_terbit_tidak_valid']:,} judul buku", "Dikonversi menjadi NaN (dataset dikoleksi tahun 2004)"],
            ["ISBN rating tidak terdaftar di metadata buku", f"{q['isbn_rating_tidak_ada_di_buku_pct']:.1f}% interaksi rating", "Dieliminasi dari model komputasi"],
            ["Rating implisit (nilai 0)", f"{q['rating_implisit_pct']:.1f}% seluruh interaksi", "Dikecualikan dari Collaborative Filtering"],
            ["Entri lokasi negara tidak teridentifikasi", f"{q['negara_unknown']:,} profil pengguna", "Diberi label 'Unknown'"],
            ["Duplikasi entri (User-ID, ISBN)", f"{q['rating_duplikat_dibuang']:,} baris duplikat", "Dieliminasi dengan mempertahankan catatan interaksi terakhir"]
        ], columns=["Karakteristik Data", "Besaran Masalah", "Tindakan Pra-pemrosesan"])
        st.dataframe(qdf, hide_index=True, use_container_width=True)


# =====================================================================
# MODUL 2 : SISTEM REKOMENDASI BUKU
# =====================================================================
def page_recommender(d, m):
    st.title("Sistem Rekomendasi Buku")
    st.caption("Eksplorasi dan perbandingan berbagai paradigma sistem rekomendasi: Rule-Based, Content-Based, Collaborative Filtering, dan Hybrid.")

    ex = d.explicit
    items = m.items

    # 1. Panel Profil Pengguna (16px Card)
    with st.container(border=True):
        st.subheader("Profil Pengguna Aktif")
        mode = st.radio(
            "Mode Pengguna",
            ["Pengguna Terdaftar (Data Model)", "Simulasi Pengguna Baru (Cold-Start)"],
            horizontal=True
        )

        if mode.startswith("Pengguna Terdaftar"):
            def _rand_user():
                st.session_state["uid_sel"] = int(np.random.choice(m.user_ids))

            c_uid, c_btn = st.columns([4, 1])
            uid = c_uid.selectbox(
                "Pilih User-ID",
                list(m.user_ids),
                key="uid_sel",
                help="Pengguna yang telah melewati k-core filter dan memiliki riwayat rating di model."
            )
            c_btn.write("")
            c_btn.button("Acak Sampel", on_click=_rand_user, use_container_width=True)

            idx, r, self_idx = rs.user_vec(m, uid)
            urow = d.users[d.users["user_id"] == uid].iloc[0]
            country, age_group = urow["Country"], urow["AgeGroup"]
            age_display = "-" if pd.isna(urow["Age"]) else int(urow["Age"])
            st.caption(f"Profil: Negara **{country}** | Usia **{age_display}** ({age_group}) | Riwayat Rating di Model: **{len(idx)} buku**")
        else:
            self_idx = None
            if "new_r" not in st.session_state:
                st.session_state["new_r"] = {}

            c_c, c_a = st.columns(2)
            countries = ["Semua"] + list(ex["Country"].value_counts().head(30).index)
            country = c_c.selectbox("Negara Pengguna", countries)
            age_group = c_a.selectbox("Kelompok Usia", ["Semua"] + rs.AGE_ORDER)

            q_search = st.text_input("Cari judul buku atau penulis untuk ditambahkan ke profil Anda:")
            if q_search:
                mask = (items["title"].str.contains(q_search, case=False, regex=False)
                        | items["author"].str.contains(q_search, case=False, regex=False))
                res = items[mask].sort_values("n_ratings", ascending=False).head(30)
                if len(res):
                    label_map = dict(zip(res["isbn"], res["title"].str.slice(0, 65) + " — " + res["author"]))
                    c_pick, c_val, c_add = st.columns([4, 2, 1])
                    picked_isbn = c_pick.selectbox("Pilih Buku", list(res["isbn"]), format_func=lambda i: label_map[i])
                    val_rating = c_val.slider("Rating Anda", 1, 10, 8)
                    c_add.write("")
                    if c_add.button("Tambah Rating", use_container_width=True):
                        st.session_state["new_r"][picked_isbn] = val_rating
                else:
                    st.warning("Buku tidak ditemukan pada katalog model.")

            idx, r = rs.vec_from_isbn(m, st.session_state["new_r"])
            if len(idx):
                cur_df = items.iloc[idx][["title", "author"]].copy()
                cur_df["Rating"] = r
                st.dataframe(cur_df, hide_index=True, use_container_width=True)
                if st.button("Kosongkan Riwayat"):
                    st.session_state["new_r"] = {}
                    st.rerun()
            else:
                st.info("Belum ada buku dalam riwayat pengguna baru. Sistem akan menerapkan strategi cold-start (Rule-Based).")

    # 2. Panel Pengaturan Tampilan & Filter
    with st.container(border=True):
        c_topn, c_minpop, c_cover = st.columns(3)
        topn = c_topn.slider("Jumlah Rekomendasi (Top-N)", 4, 32, 8, step=4)
        min_pop = c_minpop.slider("Minimal Rating pada Buku Rekomendasi", 1, 50, 10)
        covers = c_cover.checkbox("Tampilkan Sampul Buku", value=True)

    seen = np.asarray(idx, dtype=int)
    excl = np.union1d(seen, np.where(items["n_ratings"].values < min_pop)[0]).astype(int)

    if len(idx):
        with st.expander(f"Daftar Riwayat Rating Pengguna Aktif ({len(idx)} buku)"):
            h_df = items.iloc[idx][["title", "author", "year"]].copy()
            h_df["Rating"] = r
            st.dataframe(h_df.sort_values("Rating", ascending=False), hide_index=True, use_container_width=True)

    # 3. Tab Paradigma Sistem Rekomendasi
    t_rule, t_cb, t_cf, t_hyb, t_sim = st.tabs([
        "Rule-Based",
        "Content-Based",
        "Collaborative Filtering",
        "Hybrid",
        "Pencarian Buku Serupa"
    ])

    # TAB 1: RULE-BASED
    with t_rule:
        t_sub_demo, t_sub_assoc = st.tabs(["Aturan Demografis (Domain Rules)", "Aturan Asosiasi (Association Rules)"])

        with t_sub_demo:
            cs = ["Semua"] + list(ex["Country"].value_counts().head(30).index)
            ags = ["Semua"] + rs.AGE_ORDER
            c1, c2, c3, c4 = st.columns(4)
            sel_c = c1.selectbox("Filter Negara", cs, index=cs.index(country) if country in cs else 0, key="rb_c")
            sel_a = c2.selectbox("Filter Usia", ags, index=ags.index(age_group) if age_group in ags else 0, key="rb_a")
            yr = c3.slider("Tahun Terbit", 1900, 2006, (1950, 2006), key="rb_yr")
            mv = c4.slider("Minimal Vote (m)", 1, 50, 5, key="rb_mv")

            st.info(f"**Logika Aturan:** IF Negara = *{sel_c}* AND Usia = *{sel_a}* AND Tahun Terbit {yr[0]}–{yr[1]} THEN Rekomendasikan buku dengan Bayesian Weighted Rating tertinggi.")
            seen_isbn = set(items["isbn"].iloc[seen]) if len(seen) else set()
            big_n = topn + len(seen_isbn) + 5
            pop = rs.popular_books(ex, d.books, sel_c, sel_a, yr, mv, big_n)
            if len(pop):
                pop = pop[~pop["isbn"].isin(seen_isbn)]
            if len(pop) < topn:
                st.warning("Segmen data terlalu sempit. Menerapkan fallback otomatis ke aturan global.")
                pop = rs.popular_books(ex, d.books, "Semua", "Semua", yr, mv, big_n)
                if len(pop):
                    pop = pop[~pop["isbn"].isin(seen_isbn)]

            show_recs(pop.head(topn) if len(pop) else pop, covers=covers, score_label="Weighted Rating", score_fmt="%.2f", key="rb_demo")
            st.caption("Formula: Weighted Rating = v/(v+m)·R + m/(v+m)·C (v = vote buku, m = ambang batas, R = rata-rata buku, C = rata-rata segmen).")

        with t_sub_assoc:
            liked = [int(i) for i in idx[r >= rs.LIKE_THR]] if len(idx) else []
            if not liked:
                st.info("Butuh minimal 1 buku yang diberi rating ≥ 7 sebagai keranjang referensi (basket).")
            else:
                liked = sorted(liked, key=lambda i: -items["n_ratings"].iloc[i])
                c1, c2, c3 = st.columns([3, 1, 1])
                basket = c1.multiselect(
                    "Pilih Buku yang Disukai (Basket):",
                    liked,
                    default=liked[:1],
                    format_func=lambda i: items["title"].iloc[i][:75]
                )
                min_c = c2.slider("Minimal Pembaca Bersama", 2, 20, 3, key="ar_min_c")
                sort_by = c3.selectbox("Urutkan Berdasarkan", ["lift", "confidence", "support"], key="ar_sort")

                if basket:
                    rules = rs.association_rules(m, basket, min_c, topn + len(seen) + 5, sort_by)
                    if len(rules):
                        rules = rules[~rules["isbn"].isin(set(items["isbn"].iloc[seen]))].head(topn)
                    if len(rules):
                        b_txt = " DAN ".join(f"'{items['title'].iloc[i][:35]}'" for i in basket)
                        top_r = rules.iloc[0]
                        st.success(f"**Aturan Dominan:** IF menyukai {b_txt} THEN menyukai '{top_r['title'][:50]}' (Support: {top_r['support']:.3%}, Confidence: {top_r['confidence']:.0%}, Lift: {top_r['lift']:.2f})")

                        # Visualisasi Nilai Lift (Fitur Unggulan Adopsi dari Shiny)
                        top_plot = rules.nlargest(10, "lift").sort_values("lift")
                        top_plot["Judul Singkat"] = top_plot["title"].str.slice(0, 50)
                        fig_lift = px.bar(
                            top_plot, x="lift", y="Judul Singkat", orientation="h",
                            template=TPL, color_discrete_sequence=[PRIMARY_GREEN],
                            labels={"lift": "Nilai Lift", "Judul Singkat": "Buku Terkait (Consequent)"},
                            title="Buku yang Paling Sering Ikut Disukai Berdasarkan Nilai Lift",
                            hover_data=["title", "confidence", "support", "author"]
                        )
                        show(fig_lift, height=360)

                        st.dataframe(
                            rules[["title", "author", "year", "n_pengguna_basket", "n_bersama", "support", "confidence", "lift"]],
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "title": st.column_config.TextColumn("Judul Buku", width="large"),
                                "author": st.column_config.TextColumn("Penulis"),
                                "year": st.column_config.NumberColumn("Tahun", format="%d"),
                                "n_pengguna_basket": st.column_config.NumberColumn("Pembaca Basket", format="%d"),
                                "n_bersama": st.column_config.NumberColumn("Pembaca Bersama", format="%d"),
                                "support": st.column_config.NumberColumn("Support", format="%.4f"),
                                "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                                "lift": st.column_config.NumberColumn("Lift", format="%.2f")
                            }
                        )
                    else:
                        st.info("Tidak ada aturan asosiasi yang memenuhi ambang batas minimal dukungan.")

    # TAB 2: CONTENT-BASED
    with t_cb:
        st.caption("Rekomendasi berbasis profil teks metadata buku (judul, penulis, penerbit, dekade) menggunakan TF-IDF dan Cosine Similarity.")
        if len(idx) == 0:
            st.info("Profil pengguna belum memiliki riwayat rating. Tambahkan minimal 1 buku pada mode simulasi.")
        else:
            rec_cb = rs.top_n(m, rs.score_cb(m, idx, r), excl, topn)
            why = rs.explain_cb(m, rec_cb["item_idx"].values, idx, r)
            rec_cb["Alasan"] = [f"Mirip: '{t[:35]}' (cos: {c:.2f})" for t, c in why]
            show_recs(rec_cb, covers=covers, extra=["Alasan"], score_label="Cosine", score_fmt="%.3f", key="cb_recs")

    # TAB 3: COLLABORATIVE FILTERING
    with t_cf:
        cf_method = st.radio(
            "Metode Collaborative Filtering",
            ["Item-Based", "User-Based", "SVD"],
            horizontal=True,
            format_func=lambda x: {
                "Item-Based": "Item-Based CF (Memory-Based)",
                "User-Based": "User-Based CF (Memory-Based)",
                "SVD": "SVD / Matrix Factorization (Model-Based)"
            }[x]
        )
        if len(idx) == 0:
            st.info("Collaborative Filtering membutuhkan riwayat interaksi rating pengguna (User Cold-Start).")
        else:
            cf_sc = rs.cf_scores(m, cf_method, idx, r, self_idx)
            show_recs(rs.top_n(m, cf_sc, excl, topn), covers=covers, score_label="Prediksi Rating", score_fmt="%.2f", key="cf_recs")
            method_desc = {
                "Item-Based": "Menghitung kemiripan antar-buku menggunakan Adjusted Cosine dengan Bayesian Shrinkage dari pola rating seluruh pengguna.",
                "User-Based": "Mencari 50 tetangga pengguna terdekat (k-NN pada rating terpusat) lalu menghitung rata-rata tertimbang deviasi mereka.",
                "SVD": f"Mendekomposisi matriks ke ruang faktor laten (k = {m.Vt.shape[0]}). Vektor pengguna diproyeksikan dan direkonstruksi."
            }[cf_method]
            st.caption(method_desc)

    # TAB 4: HYBRID
    with t_hyb:
        hyb_mode = st.radio(
            "Strategi Hybrid",
            ["Weighted Hybrid", "Switching Hybrid", "Parallel Comparison"],
            horizontal=True
        )
        if hyb_mode == "Weighted Hybrid":
            c_a, c_cf = st.columns(2)
            alpha = c_a.slider("Bobot Content-Based (α)", 0.0, 1.0, 0.5, step=0.1)
            cf_choice = c_cf.selectbox("Komponen Collaborative Filtering", ["SVD", "Item-Based", "User-Based"])
            if len(idx) == 0:
                st.info("Membutuhkan riwayat rating. Silakan pilih tab Switching Hybrid untuk simulasi cold-start.")
            else:
                hyb_sc = rs.hybrid_weighted(m, idx, r, alpha, cf_choice, self_idx)
                st.caption(f"Skor Akhir = {alpha:.1f} × Rank(CB) + {1 - alpha:.1f} × Rank(CF {cf_choice}) dalam skala persentil terstandarisasi.")
                show_recs(rs.top_n(m, hyb_sc, excl, topn), covers=covers, score_label="Skor Hybrid", score_fmt="%.3f", key="hyb_w")
        elif hyb_mode == "Switching Hybrid":
            th_sw = st.slider("Ambang Batas Jumlah Rating untuk Beralih ke CF", 2, 20, 5)
            sw_name, sw_sc = rs.hybrid_switching(m, idx, r, th_sw, "SVD", self_idx)
            st.success(f"Strategi Aktif: **{sw_name}** (Jumlah rating pengguna saat ini: {len(idx)}, Ambang batas transisi: {th_sw})")
            show_recs(rs.top_n(m, sw_sc, excl, topn), covers=covers, score_label="Skor", score_fmt="%.3f", key="hyb_sw")
            st.caption("Logika: 0 rating → Rule-Based (Popularitas); 1–(ambang-1) rating → Content-Based; ≥ ambang → Collaborative Filtering.")
        else:
            col1, col2, col3 = st.columns(3)
            sub_n = min(topn, 4)
            with col1:
                st.markdown("**Content-Based**")
                if len(idx):
                    show_recs(rs.top_n(m, rs.score_cb(m, idx, r), excl, sub_n), covers=covers, score_label="Cosine", score_fmt="%.2f", key="par_cb")
                else:
                    st.info("Butuh riwayat.")
            with col2:
                st.markdown("**Collaborative (SVD)**")
                if len(idx):
                    show_recs(rs.top_n(m, rs.score_svd(m, idx, r), excl, sub_n), covers=covers, score_label="Prediksi", score_fmt="%.2f", key="par_cf")
                else:
                    st.info("Butuh riwayat.")
            with col3:
                st.markdown("**Rule-Based Terpopuler**")
                show_recs(rs.top_n(m, items["wr"].values, excl, sub_n), covers=covers, score_label="W. Rating", score_fmt="%.2f", key="par_rb")

    # TAB 5: BUKU SERUPA
    with t_sim:
        st.caption("Pencarian buku tunggal dan komparasi kemiripan konten (metadata) vs kemiripan interaksi pembaca (Item-CF).")
        q_sim = st.text_input("Cari judul buku atau penulis:", key="sim_query")
        if q_sim:
            mask_sim = (items["title"].str.contains(q_sim, case=False, regex=False)
                        | items["author"].str.contains(q_sim, case=False, regex=False))
            res_sim = items[mask_sim].sort_values("n_ratings", ascending=False).head(30)
            if len(res_sim):
                pick_sim = st.selectbox(
                    "Pilih Buku Referensi",
                    list(res_sim.index),
                    format_func=lambda i: f"{items['title'].iloc[i][:65]} — {items['author'].iloc[i]}"
                )
                c_sim1, c_sim2 = st.columns(2)
                with c_sim1:
                    st.markdown("**Mirip Berdasarkan Konten (TF-IDF)**")
                    show_recs(rs.similar_items(m, int(pick_sim), "Konten", topn), covers=covers, score_label="Cosine", score_fmt="%.3f", key="sim_cb")
                with c_sim2:
                    st.markdown("**Mirip Berdasarkan Pola Pembaca (Item-CF)**")
                    show_recs(rs.similar_items(m, int(pick_sim), "CF", topn), covers=covers, score_label="Kemiripan", score_fmt="%.3f", key="sim_cf")
            else:
                st.warning("Buku tidak ditemukan pada katalog model.")


# =====================================================================
# MODUL 3 : EVALUASI MODEL
# =====================================================================
def page_evaluation(d, m):
    st.title("Evaluasi Model & Tolok Ukur")
    st.caption("Validasi performa model menggunakan skema hold-out per pengguna (80% training, 20% testing).")

    with st.container(border=True):
        st.subheader("Konfigurasi Pengujian Offline")
        c1, c2, c3, c4 = st.columns(4)
        n_users = c1.slider("Jumlah Pengguna Uji", 50, 1000, 300, step=50)
        frac = c2.slider("Proporsi Data Uji", 0.1, 0.5, 0.2, step=0.05)
        k_val = c3.selectbox("Ambang Batas Top-K", [5, 10, 20], index=1)
        alpha = c4.slider("Bobot CB pada Hybrid (α)", 0.0, 1.0, 0.5, step=0.1)

        if st.button("Jalankan Evaluasi Benchmark", use_container_width=True):
            progress_bar = st.progress(0.0, text="Mengevaluasi seluruh algoritma...")
            res_df, info = rs.evaluate(
                m, n_users, frac, k_val, alpha,
                progress=lambda p: progress_bar.progress(min(p, 1.0))
            )
            st.session_state["eval"] = (res_df, info, k_val)

    if "eval" not in st.session_state:
        st.info("Klik tombol 'Jalankan Evaluasi Benchmark' untuk memulai perhitungan performa komparatif.")
        return

    res_df, info, k_val = st.session_state["eval"]
    if len(res_df) == 0:
        st.warning("Tidak ada pengguna yang memenuhi kualifikasi minimal 10 rating. Pertimbangkan melonggarkan filter model.")
        return

    st.caption(f"Hasil Evaluasi: {info['n_users']} pengguna uji, {info['n_test_pairs']} pasang interaksi diuji; {info['n_users_ranking']} pengguna memiliki minimal 1 buku relevan (rating ≥ {rs.LIKE_THR}) untuk metrik perankingan.")

    # Format Tabel Hasil
    fmt = {c: st.column_config.NumberColumn(c, format="%.4f") for c in res_df.columns if c != "Metode"}
    fmt["Coverage (%)"] = st.column_config.NumberColumn("Coverage (%)", format="%.1f")
    st.dataframe(res_df, hide_index=True, column_config=fmt, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        e_df = res_df.melt(id_vars="Metode", value_vars=["RMSE", "MAE"], var_name="Metrik", value_name="Nilai")
        fig_err = px.bar(
            e_df, x="Metode", y="Nilai", color="Metrik", barmode="group",
            template=TPL, color_discrete_map={"RMSE": PRIMARY_GREEN, "MAE": "#9CA3AF"},
            title="Galat Prediksi Rating (Nilai Lebih Rendah = Lebih Akurat)"
        )
        show(fig_err)
    with c2:
        rk_df = res_df.melt(
            id_vars="Metode",
            value_vars=[f"Precision@{k_val}", f"Recall@{k_val}", f"HitRate@{k_val}"],
            var_name="Metrik", value_name="Nilai"
        )
        fig_rk = px.bar(
            rk_df, x="Metode", y="Nilai", color="Metrik", barmode="group",
            template=TPL, color_discrete_map={
                f"Precision@{k_val}": "#A7D7BC",
                f"Recall@{k_val}": SAGE_GREEN,
                f"HitRate@{k_val}": PRIMARY_GREEN
            },
            title=f"Kualitas Perankingan Top-{k_val} (Nilai Lebih Tinggi = Lebih Baik)"
        )
        show(fig_rk)

    best_rmse = res_df.loc[res_df["RMSE"].idxmin(), "Metode"]
    best_hr = res_df.loc[res_df[f"HitRate@{k_val}"].idxmax(), "Metode"]
    with st.container(border=True):
        st.markdown(f"**Ringkasan Performa:** Model dengan RMSE terendah adalah **{best_rmse}**, sedangkan model dengan HitRate@{k_val} tertinggi adalah **{best_hr}**.")
        st.caption("Catatan: Tingginya sparsity data menyebabkan nilai presisi absolut cenderung kecil. Evaluasi offline difokuskan pada perbandingan kinerja relatif antar model.")


# =====================================================================
# MODUL 4 : METODOLOGI & KONFIGURASI SISTEM
# =====================================================================
def page_methodology(d, m):
    st.title("Metodologi & Konfigurasi Sistem")
    st.caption("Dokumentasi teoritis, formulasi matematika, dan pengaturan parameter global model.")

    t_map, t_math, t_conf = st.tabs([
        "Pemetaan Materi Perkuliahan",
        "Formulasi Matematika",
        "Parameter Global Model"
    ])

    with t_map:
        st.markdown("""
| Materi Kuliah Sistem Rekomendasi | Implementasi pada Proyek Dashboard | Keterangan & Karakteristik |
|---|---|---|
| **Rule-Based: Data-Driven** | Association Rules `{Buku}` → `{Buku}` | Berdasarkan buku yang disukai (rating ≥ 7) dengan metrik Support, Confidence, dan Lift. |
| **Rule-Based: Domain/Business** | Aturan Segmen Demografis (Negara, Usia, Tahun) | Menggunakan IMDb Weighted Rating (Bayesian average) dengan fallback otomatis. |
| **Content-Based Filtering** | TF-IDF pada Metadata Soup + Cosine Similarity | Menggabungkan judul, penulis (bobot ganda), penerbit, dan dekade terbit. |
| **Collaborative: Memory-Based** | Item-Based & User-Based CF | Normalisasi mean-centering dan Bayesian shrinkage untuk meredam pengguna baru. |
| **Collaborative: Model-Based** | TruncatedSVD (Matrix Factorization) | Dekomposisi faktor laten dengan metode fold-in untuk representasi profil pengguna baru. |
| **Hybrid Recommender** | Weighted, Switching, dan Parallel Mixed | Menggabungkan kekuatan berbagai pendekatan untuk mengatasi cold-start dan sparsity. |
| **Evaluasi Kinerja** | Skema Hold-out Offline per Pengguna | Mengukur galat prediksi (RMSE, MAE) dan kualitas perankingan (Precision, Recall, HitRate). |
""")

    with t_math:
        st.markdown(r"""
##### 1. Normalisasi Mean-Centering & Bayesian Shrinkage
Untuk menghilangkan bias subjektif penilaian pengguna:
$$\mu_u = \frac{\sum r_{u,i} + \lambda_{\mu} \cdot \mu_g}{N_u + \lambda_{\mu}}$$
$$R_c(u, i) = R(u, i) - \mu_u$$
dengan $\mu_g$ rata-rata populasi global dan $\lambda_{\mu} = 2.0$ sebagai shrinkage dampener.

##### 2. Prediksi Rating Collaborative Filtering
$$\hat{r}(u, i) = \mu_u + \frac{\sum_{j} S_{i,j} \cdot (r_{u,j} - \mu_u)}{\sum_{j} |S_{i,j}| + \lambda}$$
dengan $S_{i,j}$ kemiripan adjusted cosine yang telah dikalikan faktor penyusutan bobot bersama:
$$S'_{i,j} = S_{i,j} \cdot \frac{C_{i,j}}{C_{i,j} + \text{shrink}}$$

##### 3. Bayesian Weighted Rating (IMDb Formula)
$$WR = \frac{v}{v + m} R + \frac{m}{v + m} C$$
dengan $v$ jumlah rating buku, $m$ ambang batas minimal rating, $R$ rata-rata buku, dan $C$ rata-rata segmen.
""")

    with t_conf:
        st.subheader("Konfigurasi Parameter Model Aktif")
        st.caption("Perubahan pada parameter di bawah ini akan memicu pembentukan ulang matriks model rekomendasi.")
        with st.container(border=True):
            c_u, c_i, c_k = st.columns(3)
            p_u = c_u.slider("Minimal Rating per Pengguna (k-core)", 1, 20, 5)
            p_i = c_i.slider("Minimal Rating per Buku (k-core)", 1, 30, 5)
            p_k = c_k.slider("Faktor Laten SVD (k)", 10, 100, 50, step=10)

        st.caption(f"Dimensi Matriks Aktif: **{m.R.shape[0]:,}** pengguna × **{m.R.shape[1]:,}** buku ({m.R.nnz:,} entri rating eksplisit).")


# =====================================================================
# MAIN ENTRY POINT & SIDEBAR
# =====================================================================
ensure_data()
d = get_data()

with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:1.4rem; padding-bottom:1rem; border-bottom:1px solid #ECEEF1;">
        <div style="width:38px; height:38px; border-radius:10px; background:#164E33; color:#FFFFFF; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1.15rem; box-shadow:0 3px 8px rgba(22,78,51,0.22);">
            B
        </div>
        <div>
            <div style="font-weight:700; font-size:1.05rem; color:#111827; line-height:1.15;">BookRec</div>
            <div style="font-size:0.75rem; color:#6B7280;">Dashboard Analitika</div>
        </div>
    </div>
    <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:#9CA3AF; margin-bottom:10px;">
        MENU UTAMA
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigasi Modul", PAGES, key="nav_page", label_visibility="collapsed")

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    with st.expander("Parameter Model"):
        sb_u = st.slider("Min. rating pengguna", 1, 20, 5, key="sb_u")
        sb_i = st.slider("Min. rating buku", 1, 30, 5, key="sb_i")
        sb_k = st.slider("Faktor laten SVD", 10, 100, 50, step=10, key="sb_k")

    st.markdown("""
    <div style="background:#164E33; border-radius:14px; padding:16px; color:#FFFFFF; margin-top:20px; box-shadow:0 4px 12px rgba(22,78,51,0.15);">
        <div style="font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.6px; color:#A7D7BC;">DATASET AKTIF</div>
        <div style="font-weight:700; font-size:0.95rem; color:#FFFFFF; margin-top:4px;">Book-Crossing</div>
        <div style="font-size:0.75rem; color:#E8F5EE; margin-top:6px; line-height:1.4;">
            1.15M Rating interaksi<br>278K Pengguna terdaftar<br>271K Judul katalog buku
        </div>
    </div>
    """, unsafe_allow_html=True)

m = get_model(sb_u, sb_i, sb_k)

if page == PAGES[0]:
    page_analytics(d, m)
elif page == PAGES[1]:
    page_recommender(d, m)
elif page == PAGES[2]:
    page_evaluation(d, m)
else:
    page_methodology(d, m)
