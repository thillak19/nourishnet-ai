# NourishNet AI - System Architecture

## Overview

# 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                CLIENT / USER                                │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                │ HTTP Requests
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FastAPI REST API (Port 8000)                         │
│                                                                              │
│  /predict/delivery-time      /predict/churn                                 │
│  /predict/sentiment          /predict/food-image                            │
│  /generate/food-desc         /recommend/collaborative                       │
│  /recommend/content-based    /health                                        │
└───────┬─────────────┬─────────────┬─────────────┬────────────────────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────────┐
│ ML Models   │ │ Deep Learning│ │ NLP Models │ │ Computer Vision│
│─────────────│ │─────────────│ │────────────│ │────────────────│
│ • GBM       │ │ • MLP       │ │ • DistilBERT│ │ • MobileNetV2 │
│ • RFC       │ │   (Keras)   │ │ • OPT-125M │ │   (PyTorch)   │
│ • LR        │ │             │ │            │ │               │
└──────┬──────┘ └──────┬──────┘ └──────┬─────┘ └──────┬────────┘
       │               │               │              │
       └───────────────┴───────────────┴──────────────┘
                               │
                               ▼
                     ┌──────────────────────┐
                     │ Recommendation System│
                     │──────────────────────│
                     │ • Collaborative      │
                     │ • Content-Based      │
                     └──────────┬───────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Model Storage Layer                               │
│──────────────────────────────────────────────────────────────────────────────│
│  models/saved/                                                        │
│  ├── *.pkl                                                            │
│  ├── *.keras                                                          │
│  ├── *.pth                                                            │
│  └── FAISS Vector Store                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```
## Module Summary

| Module | Model | Framework |
|---|---|---|
| Delivery Time Prediction | GradientBoosting + MLP | Scikit-learn + TensorFlow |
| Churn Classification | RandomForest | Scikit-learn |
| Sentiment Analysis | DistilBERT | HuggingFace Transformers |
| Food Image Classification | MobileNetV2 | PyTorch |
| Generative AI | OPT-125m | HuggingFace Transformers |
| RAG Customer Support | LangChain + FAISS | LangChain |
| Recommendation System | Collaborative + Content-Based | Scikit-learn |
| REST API | FastAPI | FastAPI + Uvicorn |
| Containerization | Docker | Docker + docker-compose |
| CI Pipeline | GitHub Actions | YAML |