# NourishNet AI 🍽️

**An intelligent AI/ML backend suite for food delivery platforms**, covering end-to-end machine learning pipelines from data generation to model deployment.

---

## 🚀 Project Overview

NourishNet AI is a comprehensive machine learning project built to simulate real-world AI capabilities required in a food delivery platform. It covers classical ML, deep learning, computer vision, NLP, generative AI, RAG, recommendation systems, and MLOps deployment — all unified under a single FastAPI service.

---

## 🧠 Modules

| Module | Technology | Key Metric |
|---|---|---|
| Delivery Time Prediction | Scikit-learn + TensorFlow MLP | R² = 0.916, MAE = 2.92 min |
| Customer Churn Prediction | Random Forest Classifier | F1 = 0.667, AUC = 0.853 |
| Food Image Classification (CV) | PyTorch MobileNetV2 (Transfer Learning) | Accuracy = 88.74% |
| Sentiment Analysis (NLP) | DistilBERT (HuggingFace Transformers) | Accuracy = 100% (synthetic) |
| RAG Customer Support | LangChain + FAISS + OPT-125m | Retrieval-based QA |
| Generative AI | OPT-125m Text Generation | Menu description generation |
| Recommendation System | Collaborative + Content-Based Filtering | Cosine similarity |
| REST API (MLOps) | FastAPI + Uvicorn | 6 live endpoints |

---
```
📁 Project Structure
nourishnet-ai/
├── src/
│   ├── data/
│   │   └── generate_dataset.py        # Synthetic dataset generator
│   ├── models/
│   │   ├── train_delivery_time.py     # Regression models
│   │   └── train_churn.py             # Classification models
│   └── api/
│       └── main.py                    # FastAPI service
├── data/
│   └── raw/
│       ├── orders.csv                 # 8000 order records
│       └── customers.csv              # 1500 customer records
├── models/
│   └── saved/                         # Trained model artifacts
├── notebooks/
│   ├── nourishnet_cv_module.ipynb
│   ├── nourishnet_nlp_module.ipynb
│   ├── nourishnet_rag_module.ipynb
│   └── nourishnet_recommendation_module.ipynb
├── docs/                              # Screenshots and outputs
└── docker/
```
---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/nourishnet-ai.git
cd nourishnet-ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python src/data/generate_dataset.py

# Train models
python src/models/train_delivery_time.py
python src/models/train_churn.py

# Run API
uvicorn src.api.main:app --reload
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict/delivery-time` | Predict estimated delivery time |
| POST | `/predict/churn` | Predict customer churn risk |
| POST | `/recommend/collaborative` | User-based food recommendations |
| POST | `/recommend/content-based` | Item-based food recommendations |
| GET | `/health` | API health status |

Interactive API docs available at: `http://127.0.0.1:8000/docs`

---

## 📊 Results

### Delivery Time Prediction
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 3.06 min | 3.95 | 0.901 |
| Random Forest | 3.18 min | 4.08 | 0.895 |
| Gradient Boosting | 2.90 min | 3.69 | 0.914 |
| Deep Learning MLP | 2.92 min | 3.65 | **0.916** |

### Customer Churn Classification
| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Logistic Regression | 0.754 | 0.482 | 0.827 |
| Random Forest | **0.901** | **0.667** | **0.853** |
| Gradient Boosting | 0.902 | 0.653 | 0.876 |

### Food Image Classifier (CV)
- Architecture: MobileNetV2 (Transfer Learning, frozen base)
- Classes: 7 food categories
- Best Test Accuracy: **88.74%** (10 epochs, GPU)

### Sentiment Analysis (NLP)
- Model: DistilBERT fine-tuned
- Classes: Positive, Neutral, Negative
- Best Accuracy: **100%** on synthetic dataset

---

## 🛠️ Tech Stack

- **Languages:** Python 3.12
- **ML/DL:** Scikit-learn, TensorFlow, PyTorch
- **NLP:** HuggingFace Transformers (DistilBERT, OPT-125m)
- **CV:** TorchVision, MobileNetV2
- **RAG:** LangChain, FAISS, Sentence Transformers
- **API:** FastAPI, Uvicorn, Pydantic
- **Data:** Pandas, NumPy, Faker
- **Visualization:** Matplotlib

---

## 📸 Screenshots

All module output screenshots are available in the `docs/` folder.

---

## 👤 Author

**Thillak K**
B.E. Computer Science & Engineering (IoT)
Sri Krishna College of Technology, Coimbatore

LinkedIn: [linkedin.com/in/thillak-k](https://linkedin.com/in/thillak-k)
GitHub: [github.com/thillak19](https://github.com/thillak19)
