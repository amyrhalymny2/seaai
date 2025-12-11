Below is a clean, professional **README.md** you can drop directly into your project.

---

# Company-Dedupe-API

A FastAPI microservice for **Arabic/English company-name deduplication and clustering** using **SentenceTransformer embeddings** + **Annoy (Approximate Nearest Neighbors)**.

This service normalizes company names, generates semantic embeddings, builds an ANN index, and groups similar names into clusters based on cosine similarity.

---

## 🚀 Features

* 🔤 **Robust name normalization** (Arabic/English, removes common terms like “شركة”, “Limited”, “Co” etc.)
* 🔎 **Semantic similarity** using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* ⚡ **Fast clustering** using **Annoy**
* 📁 **Two modes**:

  * `/cluster-file` → upload CSV, get CSV with `cluster_id`
  * `/cluster-json` → send JSON, get JSON clusters
* 🧪 `/health` readiness probe

---

## 📦 Requirements

Python 3.8+

```bash
pip install fastapi uvicorn pandas numpy sentence-transformers scikit-learn annoy
```

---

## ▶️ Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

```
http://localhost:8000/docs
```

Swagger UI is fully enabled.

---

## 📁 1) `/cluster-file` — CSV Upload

### **Endpoint**

`POST /cluster-file`

### **Parameters**

| Name      | Type       | Default  | Description                   |
| --------- | ---------- | -------- | ----------------------------- |
| file      | CSV Upload | —        | File with company names       |
| col       | string     | `"Name"` | Column to cluster             |
| threshold | float      | 0.85     | Cosine similarity threshold   |
| batch     | int        | 64       | Batch size for embedding      |
| k         | int        | 10       | Annoy top-K nearest neighbors |
| trees     | int        | 10       | Annoy tree count              |

### **Returns**

A CSV file identical to the uploaded one with a new column:

| cluster_id              |
| ----------------------- |
| -1 = no match           |
| 1,2,3… = cluster groups |

### **Example (curl)**

```bash
curl -X POST "http://localhost:8000/cluster-file" \
  -F "file=@companies.csv" \
  -F "col=Name" \
  -o clustered.csv
```

---

## 📄 2) `/cluster-json` — JSON API

### **Endpoint**

`POST /cluster-json`

### **Payload Example**

```json
{
  "names": [
    "شركة الزاهد للتراكتورات",
    "شركه الزاهد للتراكتورات",
    "Other Company"
  ],
  "threshold": 0.85,
  "batch": 64,
  "k": 10,
  "trees": 10
}
```

### **Response Example**

```json
{
  "clusters": [
    [
      "شركة الزاهد للتراكتورات",
      "شركه الزاهد للتراكتورات"
    ]
  ]
}
```

---

## 🧠 How It Works

### 1. **Normalization**

* Lowercasing
* Removing Arabic diacritics
* Removing punctuation
* Removing common business suffixes (e.g., شركة, Limited, Ltd, Co)

### 2. **Embeddings**

Vectors generated via:

```
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 3. **Annoy Index**

Fast ANN search using angular distance.

### 4. **Clustering**

Pairs with cosine similarity ≥ threshold are unioned via Union-Find.

---

## 🛠 Environment Variables (optional)

If loading large models:

```
export TRANSFORMERS_CACHE=/your/cache/dir
export HF_HOME=/your/home/dir
```

---

## 💡 Tips for Best Results

* Threshold **0.82–0.87** works best for Arabic company names.
* If clustering too aggressively → raise threshold.
* If under-clustering → increase `k` and `trees`.

---

If you'd like, I can generate:
✅ Dockerfile
✅ example CSV
✅ Postman collection
✅ Deployment instructions for AWS / Azure / GCP

Just tell me!
