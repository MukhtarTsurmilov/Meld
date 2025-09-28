import os
import re
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import process, fuzz
import pymorphy3
from scipy import sparse
import gdown

# 🔥 НАСТРОЙКА ЗАГРУЗКИ АРТЕФАКТОВ ИЗ GOOGLE DRIVE
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 🔥 ЗАМЕНИ ССЫЛКИ НА ТВОИ ФАЙЛЫ ИЗ GOOGLE DRIVE
files_to_download = [
    ("data_arrays.npz", "https://drive.google.com/uc?id=1J2aZ4Din2s3W2JVlROH7A3PhPkhzZTOA"),
    ("vectorizer.joblib", "https://drive.google.com/uc?id=1t9UjgGZfCYubSZbyip_UNzEVSRgEzqm8"),
    ("tfidf_matrix.joblib", "https://drive.google.com/uc?id=1trO2RiHvggQkNzmLWDWdCaku9DvXnjvv"),
    ("vectorizer_lemma.joblib", "https://drive.google.com/uc?id=1p2iYFFxxYnCZKVL7irBr2KP-PggoG8Fa"),
    ("tfidf_matrix_lemma.joblib", "https://drive.google.com/uc?id=1ZynEXjWfp-cl00f2oIOjuE98UOnMynU7"),
]


st.info("🚀 Приложение запущено, начинаю загрузку артефактов...")

for filename, url in files_to_download:
    st.info(f"📥 Пытаюсь скачать {filename}...")
    filepath = os.path.join(ARTIFACT_DIR, filename)
    if not os.path.exists(filepath):
        st.info(f"📥 Скачиваю {filename}...")
        gdown.download(url, filepath, quiet=False)
        st.success(f"✅ {filename} скачан")
    else:
        st.success(f"✅ {filename} уже есть")

# Вспомогательные функции
def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("ё", "е")
    # Сохраняем запятые между числами
    while re.search(r'\d,\d', text):
        text = re.sub(r'(\d),(\d)', r'\1_TEMP_COMMA_\2', text)
    # Удаляем нежелательные символы
    text = re.sub(r"[^\w\s\-\.\/+]", " ", text)
    # Восстанавливаем запятые
    text = text.replace('_TEMP_COMMA_', ',')
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def get_morph():
    return pymorphy3.MorphAnalyzer()

def lemmatize_text(text, morph):
    """Лемматизация текста с кэшированием"""
    if not isinstance(text, str):
        return ""
    word_lemma_cache = {}
    def lemmatize_word(word):
        if len(word) <= 2:
            return word
        if word in word_lemma_cache:
            return word_lemma_cache[word]
        try:
            lemma = morph.parse(word)[0].normal_form
        except:
            lemma = word
        word_lemma_cache[word] = lemma
        return lemma
    return " ".join(lemmatize_word(w) for w in text.split())

def get_all_word_forms(word: str):
    morph = get_morph()
    forms = set()
    try:
        parses = morph.parse(word)
        for p in parses:
            for wf in p.lexeme:
                forms.add(wf.word.lower())
    except:
        pass
    forms.add(word.lower())
    return forms

def highlight_morphological_matches(text: str, query: str) -> str:
    if not query.strip():
        return html.escape(text)
    # Нормализуем запрос и разбиваем на слова
    q_norm = normalize(query)
    query_words = q_norm.split()
    if not query_words:
        return html.escape(text)
    highlighted = html.escape(text)
    # Для каждого слова — ищем вхождение
    for word in query_words:
        if len(word) < 2:
            continue
        escaped_word = re.escape(word)
        matches = list(re.finditer(escaped_word, highlighted, re.IGNORECASE))
        for match in reversed(matches):
            start, end = match.span()
            original_fragment = highlighted[start:end]
            highlighted = highlighted[:start] + f"<strong>{original_fragment}</strong>" + highlighted[end:]
    return highlighted

def artifacts_exist(dir_path):
    # Обновляем проверку для NPZ архива
    required_files = ["vectorizer.joblib", "tfidf_matrix.joblib", "data_arrays.npz"]
    return all(os.path.exists(os.path.join(dir_path, f)) for f in required_files)

@st.cache_resource
def load_index(artifacts_dir=ARTIFACT_DIR):
    # Загружаем joblib файлы
    vectorizer = joblib.load(os.path.join(artifacts_dir, "vectorizer.joblib"))
    X = joblib.load(os.path.join(artifacts_dir, "tfidf_matrix.joblib"))
    # 🔥 ЗАГРУЗКА ИЗ NPZ АРХИВА вместо отдельных .npy
    npz_path = os.path.join(artifacts_dir, "data_arrays.npz")
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as data:
            names = data['names']
            texts = data['texts']
            ids = data.get('ids', None)
            texts_lemmatized = data.get('texts_lemmatized', None)
    else:
        # Fallback для обратной совместимости
        st.warning("NPZ архив не найден, пытаюсь загрузить отдельные NPY файлы...")
        names = np.load(os.path.join(artifacts_dir, "names.npy"), allow_pickle=True)
        texts = np.load(os.path.join(artifacts_dir, "texts.npy"), allow_pickle=True)
        ids_path = os.path.join(artifacts_dir, "ids.npy")
        ids = np.load(ids_path, allow_pickle=True) if os.path.exists(ids_path) else None
        lemmatized_path = os.path.join(artifacts_dir, "texts_lemmatized.npy")
        texts_lemmatized = np.load(lemmatized_path, allow_pickle=True) if os.path.exists(lemmatized_path) else None
    # Загружаем лемматизированные артефакты — если есть
    vectorizer_lemma = None
    X_lemma = None
    vectorizer_lemma_path = os.path.join(artifacts_dir, "vectorizer_lemma.joblib")
    if os.path.exists(vectorizer_lemma_path):
        vectorizer_lemma = joblib.load(vectorizer_lemma_path)
    tfidf_lemma_path = os.path.join(artifacts_dir, "tfidf_matrix_lemma.joblib")
    if os.path.exists(tfidf_lemma_path):
        X_lemma = joblib.load(tfidf_lemma_path)

    # 🔥 ПРОВЕРКА: Если X_lemma есть, но vectorizer_lemma нет — обнуляем X_lemma
    if X_lemma is not None and vectorizer_lemma is None:
        st.warning("⚠️ Загружена матрица лемм, но нет векторайзера — игнорирую леммы")
        X_lemma = None

    return vectorizer, X, names, texts, ids, vectorizer_lemma, X_lemma, texts_lemmatized

def get_candidates(query, top_k, vectorizer, X, names, texts, 
                   vectorizer_lemma=None, X_lemma=None, texts_lemmatized=None):
    """Улучшенный поиск кандидатов с раздельным поиском и дедупликацией"""
    q_norm = normalize(query)
    morph = get_morph()
    #РАЗДЕЛЬНЫЙ ПОИСК С ДЕДУПЛИКАЦИЕЙ
    if vectorizer_lemma is not None and X_lemma is not None and texts_lemmatized is not None:
        # 1. Поиск по обычным текстам (50% кандидатов)
        q_vec_standard = vectorizer.transform([q_norm])
        sims_standard = linear_kernel(X, q_vec_standard).ravel()
        # 2. Поиск по лемматизированным текстам (50% кандидатов)
        q_lemma = lemmatize_text(q_norm, morph)
        q_vec_lemma = vectorizer_lemma.transform([q_lemma])
        sims_lemma = linear_kernel(X_lemma, q_vec_lemma).ravel()
        # 3. Берем раздельно топ-K/2 от каждого метода
        k_half = max(top_k // 2, 1)  # Минимум 1 от каждого метода
        k_rest = top_k - k_half
        # Топ кандидатов из обычного поиска
        if len(sims_standard) >= k_half:
            idxs_standard = np.argpartition(-sims_standard, k_half)[:k_half]
        else:
            idxs_standard = np.argsort(-sims_standard)[:len(sims_standard)]
        # Топ кандидатов из поиска по леммам
        if len(sims_lemma) >= k_rest:
            idxs_lemma = np.argpartition(-sims_lemma, k_rest)[:k_rest]
        else:
            idxs_lemma = np.argsort(-sims_lemma)[:len(sims_lemma)]
        # 4. Объединяем и дедуплицируем
        all_indices = set(idxs_standard.tolist() + idxs_lemma.tolist())
        # 5. Если после дедупликации меньше чем нужно - добираем лучшими из обоих методов
        if len(all_indices) < top_k:
            # Создаем объединенный список скоров для всех индексов
            combined_scores = []
            for idx in range(len(sims_standard)):
                if idx < len(sims_lemma):  # Защита от выхода за границы
                    combined_score = 0.5 * sims_standard[idx] + 0.5 * sims_lemma[idx]
                    combined_scores.append((idx, combined_score))
            # Сортируем по комбинированному скору и добираем недостающие
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            for idx, score in combined_scores:
                if len(all_indices) >= top_k:
                    break
                if idx not in all_indices:
                    all_indices.add(idx)
        # Преобразуем обратно в массив
        idxs = np.array(list(all_indices))
        # Сортируем по комбинированному скору для итогового порядка
        final_scores = []
        for idx in idxs:
            if idx < len(sims_lemma):
                score = 0.5 * sims_standard[idx] + 0.5 * sims_lemma[idx]
            else:
                score = sims_standard[idx]  # Fallback
            final_scores.append(score)
        final_scores = np.array(final_scores)
        sorted_order = np.argsort(-final_scores)
        idxs = idxs[sorted_order]
        sims = final_scores[sorted_order]
        st.caption(f"🔍 Раздельный поиск: {len(idxs_standard)} обычных + {len(idxs_lemma)} лемм = {len(idxs)} уникальных")
    else:
        # Fallback: только обычный поиск (оригинальная логика)
        q_vec = vectorizer.transform([q_norm])
        sims = linear_kernel(X, q_vec).ravel()
        n = len(sims)
        if top_k >= n:
            idxs = np.argsort(-sims)
        else:
            idx_part = np.argpartition(-sims, top_k)[:top_k]
            idxs = idx_part[np.argsort(-sims[idx_part])]
        st.caption("🔍 Используется обычный поиск (леммы недоступны)")
    return names[idxs], texts[idxs], sims, idxs

def smart_rerank(query, names, texts, sims, idxs, top_n=20, ids_full=None,
                 vectorizer_lemma=None, X_lemma=None, texts_lemmatized=None):
    if not query.strip():
        return pd.DataFrame()
    q_norm = normalize(query)
    query_words = q_norm.split()
    query_len = len(q_norm)
    # Лемматизируем запрос — если есть инструменты
    q_lemma = q_norm
    if vectorizer_lemma is not None and texts_lemmatized is not None:
        try:
            morph = get_morph()
            q_lemma = lemmatize_text(q_norm, morph)
        except:
            pass
    results = []
    for i, text in enumerate(texts):
        # 1. Fuzzy score (по форме) — основной
        fuzzy_score = fuzz.partial_ratio(q_norm, text) / 100.0
        # 2. TF-IDF по леммам (по смыслу) — вспомогательный
        lemma_score = 0.0
        if vectorizer_lemma is not None and X_lemma is not None and texts_lemmatized is not None:
            try:
                q_vec = vectorizer_lemma.transform([q_lemma])
                global_idx = idxs[i] if idxs is not None else i
                if global_idx < X_lemma.shape[0]:
                    doc_vec = X_lemma[global_idx]
                    sim = linear_kernel(doc_vec, q_vec).ravel()[0]
                    lemma_score = float(sim)
            except:
                lemma_score = 0.0
        # 3. Комбинируем: 70% форма + 30% смысл
        if vectorizer_lemma is not None and X_lemma is not None:
            combined_score = 0.7 * fuzzy_score + 0.3 * lemma_score
        else:
            combined_score = fuzzy_score
        # 4. Бонусы
        bonus = 0.0
        # Бонус за начало строки
        if text.startswith(q_norm):
            bonus += 0.20
        # Бонус за отдельное слово
        if f" {q_norm} " in f" {text} " or text.startswith(f"{q_norm} ") or text.endswith(f" {q_norm}"):
            bonus += 0.10
        # Бонус за числа
        number_bonus = 0.0
        try:
            query_numbers = re.findall(r'\d+', q_norm)
            if query_numbers:
                text_numbers = re.findall(r'\d+', text)
                for q_num in query_numbers:
                    if q_num in text_numbers:
                        number_bonus += 0.15
                        break
        except:
            pass
        # БОНУС: если все слова из запроса есть в тексте — +20%
        all_words_present = True
        for word in query_words:
            if word not in text:
                all_words_present = False
                break
        if all_words_present and len(query_words) > 0:
            bonus += 0.20
        # 5. Штрафы
        if query_len <= 2:
            combined_score *= 0.5
        elif query_len <= 4:
            combined_score *= 0.8
        # 6. Итоговый скор
        final_score = min(combined_score * (1.0 + bonus + number_bonus), 1.0)
        results.append((i, final_score))
    # Сортируем
    results.sort(key=lambda x: x[1], reverse=True)
    top_local_indices = [i for i, _ in results[:top_n]]
    global_indices = idxs[top_local_indices] if idxs is not None else np.array(top_local_indices)
    # Формируем DataFrame
    if ids_full is not None:
        id_values = np.asarray(ids_full)[global_indices]
        id_col = "ЕНС: Код записи (ID)"
    else:
        id_values = global_indices
        id_col = "Индекс"
    return pd.DataFrame({
        "Наименование": np.asarray(names)[top_local_indices],
        id_col: id_values
    })

def render_fixed_table(df, query, name_col="Наименование", idx_col="Индекс", idx_width_px=200):
    def esc(x):
        return html.escape(str(x)) if x is not None else ""
    css = f"""
    <style>
      .tbl-wrap {{ width: 100%; }}
      table.table-fixed {{
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        background: #ffffff;
      }}
      table.table-fixed col.col-name {{ width: calc(100% - {idx_width_px}px); }}
      table.table-fixed col.col-idx  {{ width: {idx_width_px}px; }}
      table.table-fixed th, td {{
        border: 1px solid #e6e6e6;
        padding: 8px 10px;
        font-size: 14px;
        line-height: 1.3;
        vertical-align: top;
      }}
      thead th {{
        background: #ffd100;
        color: #000;
        font-weight: 600;
      }}
      td.name-cell {{
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      td.idx-cell {{
        text-align: right;
        white-space: nowrap;
      }}
      mark {{
        background: #ffeb3b !important;
        padding: 0 3px !important;
        border-radius: 3px;
        font-weight: 500;
      }}
    </style>
    """
    parts = [css, '<div class="tbl-wrap"><table class="table-fixed">']
    parts.append('<colgroup><col class="col-name"><col class="col-idx"></colgroup>')
    parts.append(f'<thead><tr><th>{esc(name_col)}</th><th>{esc(idx_col)}</th></tr></thead><tbody>')
    for _, row in df.iterrows():
        highlighted_name = highlight_morphological_matches(row[name_col], query)
        parts.append(f'<tr><td class="name-cell">{highlighted_name}</td><td class="idx-cell">{esc(row[idx_col])}</td></tr>')
    parts.append('</tbody></table></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)

# Стили и UI
st.set_page_config(page_title="Поиск по каталогу", layout="wide")
st.markdown("""
<style>
:root{ --ros-yellow: #ffd100; --ros-gray: #f8f9fa; --ros-black: #000000; }
html, body, [data-testid="stAppViewContainer"]{ background: var(--ros-gray); color: var(--ros-black); }
[data-testid="stHeader"]{ background: var(--ros-yellow); height: 5rem; }
[data-testid="stSidebar"] { background: #4a4a4a !important; color: #e9e9e9 !important; }
[data-testid="stSidebar"] * { color: #e9e9e9 !important; }
.stButton > button{ background: var(--ros-yellow); color: var(--ros-black); border: 1px solid; border-radius: 6px; }
.stButton > button:hover{ filter: brightness(0.95); }
.stSlider div[data-baseweb="slider"] > div > div > div[style*="left: 0px"] { background: var(--ros-yellow) !important; }
div[data-testid="stAppViewContainer"] .stTextInput input{ background: #fff; color: #000; border: 1px solid #ccc; border-radius: 6px; }
div[data-testid="stAppViewContainer"] .stTextInput input:focus{ outline: none; border: 1px solid #FFD100; box-shadow: 0 0 0 2px rgba(255,209,0,0.25); }
</style>
""", unsafe_allow_html=True)

st.title("Поиск по каталогу")

with st.sidebar:
    st.header("Параметры")
    top_n = st.number_input("Количество записей", min_value=1, max_value=200, value=20, step=1)

query = st.text_input("Запрос", value="", placeholder="например: болт")
btn = st.button("Искать", type="primary")
top_k = 2000

try:
    vectorizer, X, names_full, texts_full, ids_full, vectorizer_lemma, X_lemma, texts_lemmatized = load_index()
    lemma_status = "доступны" if vectorizer_lemma is not None else "не доступны"
    st.caption(f"✅ Индекс загружен: {len(names_full):,} строк. Леммы: {lemma_status}")
except Exception as e:
    st.error(f"❌ Ошибка загрузки: {e}")
    st.stop()

if btn:
    if not query.strip():
        st.warning("Введите запрос.")
    else:
        with st.spinner("🔍 Поиск кандидатов (раздельный поиск с дедупликацией)..."):
            names, texts, sims, idxs = get_candidates(
                query, top_k, vectorizer, X, names_full, texts_full,
                vectorizer_lemma, X_lemma, texts_lemmatized
            )
        with st.spinner("🎯 Переранжирование (70% fuzzy + 30% леммы)..."):
            df = smart_rerank(
                query, names, texts, sims, idxs,
                top_n=top_n,
                ids_full=ids_full,
                vectorizer_lemma=vectorizer_lemma,
                X_lemma=X_lemma,
                texts_lemmatized=texts_lemmatized
            )
        st.subheader("Результаты")
        if df.empty:
            st.info("Ничего не найдено.")
        else:
            id_col = "ЕНС: Код записи (ID)" if "ЕНС: Код записи (ID)" in df.columns else "Индекс"
            render_fixed_table(df, query, name_col="Наименование", idx_col=id_col, idx_width_px=180)
