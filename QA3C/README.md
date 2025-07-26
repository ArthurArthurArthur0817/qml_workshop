# QA3C for Financial Trading with Quantum Circuits
---

## 🚀 Highlights
- 學長的程式碼: `discrete_A3C_vqc.py`
- QLSTM + classical A3C 版本: `A3C_trading.py`
- QLSTM + QA3C 版本: `QA3C_trading.py` # 2025-7-26 updated
---

# 執行方式
```
cd ./QA3C
python ./QA3C_trading.py
```

# 保存模型
訓練完後 model 會被保存到 `./models` 下

# Evaluation
```
# evaluate QLSTM + QA3C
python test_qa3c_model_results.py

# evaluate QLSTM + classical QA3C
python test_model_results.py
```