import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
import os
import pickle
import matplotlib.pyplot as plt
import pennylane as qml
import time
import argparse

# --- 專案路徑設定與模組引用 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from QLSTM.QLSTM_trading_final import (
    create_sequences, normalize_sequences,
    CustomQLSTMCell, CustomLSTM
)

from QA3C.utils import v_wrap, set_init

from QA3C.QA3C_trading import (
    TorchVQC, Net,
    prepare_trading_data, TradingEnv,
    plot_trade_history, run_evaluation, calculate_and_print_metrics,
    qlstm_params, A3C_PARAMS
)

os.environ["OMP_NUM_THREADS"] = "1"

def main(weight_path, data_path=None, train_split=0.8):
    """
    Args:
        weight_path: 模型權重路徑
        data_path: 數據文件路徑
        train_split: 訓練集比例
    """
    print(f"=== QA3C Model ===")
    print(f"Weight path: {weight_path}")
    
    # --- 步驟 1: 載入預訓練 LSTM 模型 ---
    print("\n載入預訓練的 QLSTM 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化 QLSTM 模型
    qlstm_cell = CustomQLSTMCell(
        input_size=qlstm_params['input_size'],
        hidden_size=qlstm_params['hidden_size'],
        output_size=qlstm_params['output_size'],
        vqc_depth=qlstm_params['qnn_depth']
    ).float().to(device)
    
    lstm_model = CustomLSTM(
        input_size=qlstm_params['input_size'],
        hidden_size=qlstm_params['hidden_size'],
        lstm_cell_QT=qlstm_cell
    ).float().to(device)
    
    qlstm_model_path = os.path.join(project_root, 'QLSTM', 'models', 'qlstm_model_epochs_50.pth')
    lstm_model.load_state_dict(torch.load(qlstm_model_path, map_location=device))
    lstm_model.eval()
    print("QLSTM 模型載入完成。")

    # --- 步驟 2: 準備資料 ---
    print("\n準備交易資料...")
    if data_path is None:
        data_path = os.path.join(project_root, 'QLSTM', 'USD_TWD_Historical Data.csv')
    
    full_data_df = prepare_trading_data(file_path=data_path, num_rows=10000)
    
    # 檢查是否存在價格為 0 的異常資料
    zero_price_rows = full_data_df[full_data_df['close'] == 0]
    if not zero_price_rows.empty:
        print("錯誤：在資料中發現 'close' 價格為 0 的異常行：")
        print(zero_price_rows)
        raise ValueError("資料驗證失敗：'close' 價格不應為 0。請檢查原始資料檔案。")
    
    if full_data_df.empty:
        raise ValueError("錯誤：經過資料清理後，沒有剩餘的有效資料可供評估。")
    
    # 分割資料
    split_point = int(len(full_data_df) * train_split)
    train_df = full_data_df[:split_point]
    test_df = full_data_df[split_point:]
    
    print(f"資料準備完成：訓練集 {len(train_df)} 筆，測試集 {len(test_df)} 筆")
    print('*** training info ***')
    print(train_df.head(2), train_df.tail(2))
    print('*** testing info ***')
    print(test_df.head(2), test_df.tail(2))

    # --- 步驟 3: 初始化並載入 A3C 模型 ---
    print("\n初始化 A3C 模型...")
    dummy_env = TradingEnv(train_df, lstm_model, device)
    N_S = dummy_env.observation_space_shape[0]
    N_A = dummy_env.action_space_n
    del dummy_env
    
    gnet = Net(N_S, N_A)
    
    # 載入模型權重
    if not os.path.isabs(weight_path):
        checkpoint_path = os.path.join(project_root, weight_path)
    else:
        checkpoint_path = weight_path
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型檔案不存在：{checkpoint_path}")
    
    print(f"載入模型權重：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        gnet.load_state_dict(checkpoint['model_state_dict'])
        episode = checkpoint.get('episode', 'unknown')
        print(f"成功載入模型 (episode {episode})")
    else:
        # 處理舊格式
        gnet.load_state_dict(checkpoint)
        print("成功載入模型 (舊格式)")
    
    gnet.eval()

    # --- 步驟 4: 根據模式執行評估或預測 ---
    all_results = {}
    print("\n--- 開始執行評估 ---")
    
    # # 評估訓練集
    # train_results = run_evaluation(
    #     df=train_df, 
    #     gnet=gnet, 
    #     lstm_model=lstm_model, 
    #     device=device,
    #     title="Training Data",
    #     filename="test_results_train.png"
    # )
    # all_results['training'] = train_results
    
    # 評估測試集
    test_results = run_evaluation(
        df=test_df,
        gnet=gnet,
        lstm_model=lstm_model,
        device=device,
        title="Testing Data",
        filename="test_results_test.png"
    )
    all_results['testing'] = test_results

    # --- 步驟 5: 儲存結果 ---
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    pickle_filepath = os.path.join(results_dir, "test_model_results.pkl")
    
    with open(pickle_filepath, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n所有評估結果已儲存至: {pickle_filepath}")
    
    # 印出總結
    print("\n=== 評估總結 ===")
    if 'training' in all_results:
        print(f"訓練集總回報率: {all_results['training']['performance_metrics']['total_return_pct']:.2f}%")
    if 'testing' in all_results:
        print(f"測試集總回報率: {all_results['testing']['performance_metrics']['total_return_pct']:.2f}%")
    print("\n評估完成！")


if __name__ == "__main__":
    WEIGHT_PATH = 'QA3C/models/qa3c_model_ep.pth'  # <- 請在這裡修改模型路徑
    DATA_PATH = None   # 有預設路徑
    TRAIN_SPLIT = 0.8  # 訓練集比例
    # =====================================
    
    print(f"使用手動設定的參數：")
    print(f"  模型路徑: {WEIGHT_PATH}")
    print(f"  資料路徑: {DATA_PATH if DATA_PATH else '使用預設路徑'}")
    print(f"  訓練集比例: {TRAIN_SPLIT}")
    
    main(WEIGHT_PATH, DATA_PATH, TRAIN_SPLIT)


    # 歷史測試結果記錄：
    # 11.87% *
