import sys
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 將專案根目錄加入 Python 路徑，以便引用 QLSTM 模組
# 當前腳本位於 QA3C/A3C_trading.py
# 我們需要回到上一層 qml_workshop/ 才能找到 QLSTM/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from QLSTM.LSTM_trading import (
    create_features_and_labels,
    create_sequences,
    normalize_sequences,
    StandardLSTM,
    plot_confusion_matrix,
)

def predict_with_lstm():
    """
    使用預先訓練好的 LSTM 模型進行預測。
    """
    print("開始執行 LSTM 模型預測...")

    # --- 步驟 1: 設定參數 ---
    # 這些參數必須與訓練時使用的參數相符
    params = {
        'feature_columns': ['open', 'high', 'low', 'close', 'ma5', 'ma10'],
        'sequence_length': 8,
        'hidden_size': 16,
        'output_size': 2,
        'num_layers': 1,
        'epochs': 20,  # 用於載入正確的模型檔案
        'batch_size': 16,
    }
    print("模型參數設定完成。")

    # --- 步驟 2: 設定檔案路徑 ---
    # 相對於目前腳本 (QA3C/A3C_trading.py) 的路徑
    data_path = os.path.join(project_root, 'QLSTM', 'USD_TWD_Historical Data.csv')
    model_path = os.path.join(project_root, 'QLSTM', 'models', f"lstm_model_epochs_{params['epochs']}.pth")
    print(f"資料路徑: {data_path}")
    print(f"模型路徑: {model_path}")

    # --- 步驟 3: 載入並處理資料 ---
    # 這裡我們只載入資料的後半部分 (測試集) 來進行預測，以模擬真實情況
    print("正在載入並處理資料...")
    full_data_df = create_features_and_labels(file_path=data_path, num_rows=10000)
    
    # 與訓練腳本相同，使用 80% 資料作為分割點
    split_point = int(len(full_data_df) * 0.8)
    test_df = full_data_df[split_point:]
    test_df.reset_index(inplace=True, drop=True)

    # 建立序列並進行標準化
    X_test, y_test = create_sequences(test_df[params['feature_columns']], test_df['label'], params['sequence_length'])
    X_test_normalized = normalize_sequences(X_test)
    
    test_dataset = TensorDataset(X_test_normalized, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=params['batch_size'], shuffle=False)
    print(f"資料處理完成。待預測的資料筆數: {len(X_test)}")

    # --- 步驟 4: 載入模型 ---
    print("正在載入預訓練模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = StandardLSTM(
        input_size=len(params['feature_columns']),
        hidden_size=params['hidden_size'],
        output_size=params['output_size'],
        num_layers=params['num_layers']
    ).to(device)
    
    # 載入模型權重
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"模型成功從 {model_path} 載入。")

    # --- 步驟 5: 執行預測 ---
    print("模型評估中...")
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # --- 步驟 6: 顯示評估結果 ---
    if not all_labels:
        print("沒有可供預測的數據。")
        return

    correct = (np.array(all_preds) == np.array(all_labels)).sum()
    total = len(all_labels)
    accuracy = 100 * correct / total
    print(f'預測準確率: {accuracy:.2f} % ({correct}/{total})')

    class_names = ['Fall (0)', 'Rise (1)']
    plot_confusion_matrix(all_labels, all_preds, class_names)
    print("預測完成。")


if __name__ == "__main__":
    predict_with_lstm()
