## Statistical Significance Tests (CV vs LOATO)

| Embedding            | Classifier   | Test          |   Statistic |   p-value | Significant   |
|:---------------------|:-------------|:--------------|------------:|----------:|:--------------|
| MiniLM (384d)        | LogReg       | Paired t-test |      3.0377 |    0.0385 | Yes           |
| MiniLM (384d)        | SVM          | Paired t-test |      2.7658 |    0.0505 | No            |
| MiniLM (384d)        | XGBoost      | Paired t-test |      2.7888 |    0.0494 | Yes           |
| MiniLM (384d)        | MLP          | Paired t-test |      2.1157 |    0.1018 | No            |
| BGE-Large (1024d)    | LogReg       | Paired t-test |      3.1046 |    0.0361 | Yes           |
| BGE-Large (1024d)    | SVM          | Paired t-test |      1.6937 |    0.1656 | No            |
| BGE-Large (1024d)    | XGBoost      | Paired t-test |      2.3362 |    0.0797 | No            |
| BGE-Large (1024d)    | MLP          | Paired t-test |      2.5326 |    0.0645 | No            |
| Instructor (768d)    | LogReg       | Paired t-test |      2.4299 |    0.072  | No            |
| Instructor (768d)    | SVM          | Paired t-test |      2.2498 |    0.0877 | No            |
| Instructor (768d)    | XGBoost      | Paired t-test |      2.3215 |    0.081  | No            |
| Instructor (768d)    | MLP          | Paired t-test |      2.3874 |    0.0754 | No            |
| OpenAI-Small (1536d) | LogReg       | Paired t-test |      2.1797 |    0.0948 | No            |
| OpenAI-Small (1536d) | SVM          | Paired t-test |      2.3262 |    0.0806 | No            |
| OpenAI-Small (1536d) | XGBoost      | Paired t-test |      2.2653 |    0.0862 | No            |
| OpenAI-Small (1536d) | MLP          | Paired t-test |      1.7663 |    0.1521 | No            |
| E5-Mistral (4096d)   | LogReg       | Paired t-test |      2.564  |    0.0624 | No            |
| E5-Mistral (4096d)   | SVM          | Paired t-test |      2.2878 |    0.0841 | No            |
| E5-Mistral (4096d)   | XGBoost      | Paired t-test |      2.5633 |    0.0624 | No            |
| E5-Mistral (4096d)   | MLP          | Paired t-test |      2.4225 |    0.0726 | No            |
