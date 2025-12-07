# QA3C for Financial Trading with Quantum Circuits
---

## 🚀 Highlights
- QLSTM + classical A3C version: `A3C_trading.py`
- QLSTM + QA3C version: `QA3C_trading.py` 
---

# How to Run
```
cd ./QA3C
python ./QA3C_trading.py
```

# Model Saving
After training, the model will be saved under the ./models directory.

# Evaluation
```
# evaluate QLSTM + QA3C
python test_qa3c_model_results.py

# evaluate QLSTM + classical QA3C
python test_model_results.py
```
