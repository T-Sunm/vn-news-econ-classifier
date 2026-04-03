# Vietnamese Economic News Classifier

Phân loại nhị phân bài báo tiếng Việt: **kinh tế / không kinh tế** bằng các phương pháp NLP truyền thống.

> **Bài tập:** Buổi 3 — Text Classification với Bag-of-Words, TF-IDF, N-gram và Word Segmentation (Underthesea).

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc project](#cấu-trúc-project)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cài đặt](#cài-đặt)
- [Dữ liệu](#dữ-liệu)
- [Chạy theo thứ tự](#chạy-theo-thứ-tự)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Chi tiết các hypothesis](#chi-tiết-các-hypothesis)

---

## Tổng quan

Bài toán phân loại nhị phân: bài báo có thuộc chủ đề **kinh tế/kinh doanh** (label = 1) hay không (label = 0).

**Pipeline tổng thể:**

```
news_dataset.json
      │
      ▼
data_labeling.ipynb   ←  Gán nhãn dựa trên topic
      │  output/news_labeled_clean.csv
      ▼
20_03_2026.ipynb      ←  Baseline: BoW / TF-IDF + unigram
      │  results_df (in-memory)
      ▼
27_3_2026.ipynb       ←  H4: Bigram/Trigram  |  H7: Word Segmentation
      │
      ▼
03_04_2026_.ipynb     ←  Word Embedding (Word2Vec) + LR/SVM
```

---

## Cấu trúc project

```
Classification_news/
├── news_dataset.json          # Raw dataset (184,539 bài báo)
├── output/
│   ├── news_labeled_all.csv   # Toàn bộ bài đã gán nhãn (140,820 dòng)
│   └── news_labeled_clean.csv # Bản clean để huấn luyện (130,085 dòng)
└── notebooks/
    ├── data_labeling.ipynb    # Bước 1: Gán nhãn
    ├── 20_03_2026.ipynb       # Bước 2: Baseline experiments
    ├── 27_3_2026.ipynb        # Bước 3: Advanced experiments
    └── 03_04_2026_.ipynb      # Bước 4: Word embedding experiments (Word2Vec)
```

---

## Yêu cầu môi trường

| Thành phần | Phiên bản |
|---|---|
| Python | 3.10+ |
| pandas | ≥ 2.0 |
| scikit-learn | ≥ 1.3 |
| numpy | ≥ 1.24 |
| underthesea | ≥ 6.8 (cho H7) |
| gensim | ≥ 4.3 (cho Word2Vec) |

---

## Cài đặt

```bash
# Clone repo
git clone <repo-url>
cd Classification_news

# Tạo và kích hoạt môi trường (tùy chọn)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Cài thư viện cơ bản
pip install pandas numpy scikit-learn

# Cài underthesea (bắt buộc cho notebook 27_3_2026) và gensim (cho Word2Vec)
pip install underthesea gensim
```

> **Lưu ý Windows:** `underthesea` yêu cầu Visual C++ Build Tools.  
> Cài tại: https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

## Dữ liệu

### Raw dataset — `news_dataset.json`

| Trường | Mô tả |
|---|---|
| `id` | ID bài báo |
| `title` | Tiêu đề |
| `content` | Nội dung đầy đủ |
| `topic` | Chủ đề gốc từ trang báo |
| `source` | Tên miền báo (vnexpress, dantri, ...) |
| `url` | Đường dẫn bài |
| `crawled_at` | Thời điểm thu thập |

Tổng: **184,539 bài** từ nhiều nguồn báo Việt Nam (2022).

### Labeled dataset — `output/news_labeled_clean.csv`

| Thống kê | Giá trị |
|---|---|
| Tổng mẫu | 130,085 |
| Label = 0 (không kinh tế) | 118,306 (90.95%) |
| Label = 1 (kinh tế) | 11,779 (9.05%) |

**Quy tắc gán nhãn (label = 1):** topic thuộc một trong các nhóm:
`kinh doanh`, `kinh tế`, `tài chính - kinh doanh`, `tài chính`, `doanh nghiệp`, `thị trường`, `chứng khoán`.

---

## Chạy theo thứ tự

### Bước 1 — Gán nhãn dữ liệu

```bash
jupyter notebook notebooks/data_labeling.ipynb
```

Chạy toàn bộ cell từ trên xuống. Kết quả lưu tại:
- `output/news_labeled_all.csv`
- `output/news_labeled_clean.csv`

### Bước 2 — Baseline experiments

```bash
jupyter notebook notebooks/20_03_2026.ipynb
```

Chạy toàn bộ cell. Notebook này đọc `output/news_labeled_clean.csv` và chạy các experiment baseline, tạo ra `results_df` dùng cho bước so sánh.

### Bước 3 — Advanced experiments

```bash
jupyter notebook notebooks/27_3_2026.ipynb
```

> ⚠️ **Yêu cầu:** Phải cài `underthesea` trước (xem mục [Cài đặt](#cài-đặt)).

> ⚠️ **Yêu cầu:** Cell so sánh H4.3 và H7 cần `results_df` được định nghĩa — nhớ chạy lại các cell baseline trong notebook này trước khi chạy các cell so sánh.

### Bước 4 — Word Embedding experiments

```bash
jupyter notebook notebooks/03_04_2026_.ipynb
```

Chạy toàn bộ cell. Notebook này áp dụng Word2Vec để tạo vector biểu diễn văn bản và huấn luyện với LogisticRegression và SVM. Quá trình tiền xử lý sử dụng lại/tạo mới cache tách từ `segmented_cache.csv` để tiết kiệm thời gian.

---

## Kết quả thực nghiệm

### Baseline (20_03_2026.ipynb) — Unigram

| Vectorizer | Model | Accuracy | F1 (class 1) |
|---|---|---|---|
| TF-IDF | LinearSVC | ~0.97 | ~0.84 |
| TF-IDF | LogisticRegression | ~0.97 | ~0.84 |
| BoW | LinearSVC | ~0.97 | ~0.83 |
| TF-IDF | MultinomialNB | ~0.96 | ~0.79 |

> Kết quả chính xác xem tại output cell cuối của `20_03_2026.ipynb`.

### H4 — N-gram (27_3_2026.ipynb)

Thử nghiệm Bigram `(1,2)` và Trigram `(1,3)` với BoW và TF-IDF.

**Kỳ vọng:** Bigram/Trigram cải thiện recall cho class 1 nhờ bắt được cụm từ như "chứng khoán tăng", "lãi suất ngân hàng", "tốc độ tăng trưởng".

### H7 — Word Segmentation (27_3_2026.ipynb)

Tách từ tiếng Việt bằng `underthesea.word_tokenize` trước khi vectorize.

**Kỳ vọng:** "kinh_doanh", "tài_chính" trở thành token đơn, tăng độ chính xác cho các cụm từ ghép tiếng Việt.

> Kết quả chi tiết: xem cell tổng hợp cuối `27_3_2026.ipynb`.

### Word Embedding (03_04_2026_.ipynb)

Sử dụng Word2Vec để train embedding và lấy trung bình (average vectors) kết hợp với các mô hình classification cơ bản.

| Vectorizer | Model | Accuracy | F1 (class 1) |
|---|---|---|---|
| Word2Vec (avg) | LogisticRegression | ~0.9415 | ~0.6457 |
| Word2Vec (avg) | SVM | ~0.9421 | ~0.6365 |

**Nhận xét:** So với TF-IDF baseline, Word2Vec ở dạng lấy trung bình (average) chưa mang lại F1-score vượt trội (~0.64 so với ~0.84 của baseline). Có thể cần tinh chỉnh dimension, window size, hoặc dùng kiến trúc sâu hơn (LSTM, BERT) để tối ưu khả năng biểu diễn của word embedding.

---

## Chi tiết các hypothesis

| Hypothesis | Mô tả | Notebook |
|---|---|---|
| **Baseline** | BoW + TF-IDF, unigram `(1,1)`, 3 models | `20_03_2026.ipynb` |
| **H4** | Mở rộng N-gram: bigram `(1,2)`, trigram `(1,3)` | `27_3_2026.ipynb` |
| **H4.3** | So sánh H4 với baseline | `27_3_2026.ipynb` |
| **H7** | Tiền xử lý: word segmentation (underthesea) | `27_3_2026.ipynb` |
| **Word2Vec** | Biểu diễn văn bản bằng Word embedding trung bình (gensim Word2Vec) | `03_04_2026_.ipynb` |

---

## Các vấn đề đã biết

1. **`underthesea` không cài được:** Thử `pip install underthesea --no-build-isolation` hoặc cài Visual C++ Build Tools (Windows).
2. **`results_df` chưa được định nghĩa:** Chạy lại cell baseline trong `27_3_2026.ipynb` trước khi chạy cell so sánh H4.3 và H7.
3. **Dataset không cân bằng (imbalanced):** Class 1 chiếm ~9%. Metric quan trọng là **F1 của class 1**, không phải Accuracy.
