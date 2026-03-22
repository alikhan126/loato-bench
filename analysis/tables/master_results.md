## Master Results Table

| Embedding            | Classifier   |   CV_F1 |   LOATO_F1 |   Delta_F1 |
|:---------------------|:-------------|--------:|-----------:|-----------:|
| MiniLM (384d)        | LogReg       |  0.977  |     0.9169 |     0.0601 |
| MiniLM (384d)        | SVM          |  0.9281 |     0.8102 |     0.1179 |
| MiniLM (384d)        | XGBoost      |  0.9829 |     0.931  |     0.0519 |
| MiniLM (384d)        | MLP          |  0.992  |     0.9626 |     0.0294 |
| BGE-Large (1024d)    | LogReg       |  0.9894 |     0.9565 |     0.0329 |
| BGE-Large (1024d)    | SVM          |  0.8301 |     0.7588 |     0.0712 |
| BGE-Large (1024d)    | XGBoost      |  0.9856 |     0.9275 |     0.0581 |
| BGE-Large (1024d)    | MLP          |  0.9941 |     0.976  |     0.0181 |
| Instructor (768d)    | LogReg       |  0.9945 |     0.967  |     0.0275 |
| Instructor (768d)    | SVM          |  0.8956 |     0.7461 |     0.1496 |
| Instructor (768d)    | XGBoost      |  0.9936 |     0.9577 |     0.0359 |
| Instructor (768d)    | MLP          |  0.9966 |     0.977  |     0.0196 |
| OpenAI-Small (1536d) | LogReg       |  0.9952 |     0.9659 |     0.0293 |
| OpenAI-Small (1536d) | SVM          |  0.902  |     0.8184 |     0.0836 |
| OpenAI-Small (1536d) | XGBoost      |  0.9925 |     0.9571 |     0.0354 |
| OpenAI-Small (1536d) | MLP          |  0.9974 |     0.9758 |     0.0216 |
| E5-Mistral (4096d)   | LogReg       |  0.9937 |     0.9641 |     0.0297 |
| E5-Mistral (4096d)   | SVM          |  0.8514 |     0.7749 |     0.0765 |
| E5-Mistral (4096d)   | XGBoost      |  0.9871 |     0.9398 |     0.0473 |
| E5-Mistral (4096d)   | MLP          |  0.9958 |     0.9772 |     0.0187 |
