# qml_workshop
專案方 2 階段 :
1. 由 QLSTM 提供 [下跌, 上漲] 的訊號 :
  - [x] 建立 QLSTM 環境，並成功執行學長 code
  - [x] 先建立 GPU 版的 LSTM 實驗較快，驗證 label 能否訓練起來 
  - [ ] 改成 QLSTM 純 CPU 執行 (pending...) 
2. 拿取 QLSTM 提供的訊號作為 state，使用 A3C 進行自動交易 :
  - [x] 建立 QA3C 環境，並成功執行學長 code
  - [ ] 使用 LSTM 給予訊號，先建立第一版最簡單的 A3C
  - [ ] 建立完整版的 A3C

# enviroments
新的環境可執行 QLSTM 和 QA3C，已確定能在 win11, mac, colab 上安裝執行
```python
conda create -n qml python=3.12.11
conda activate qml
pip install -r requirements.txt
```
