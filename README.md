# House Prices - Advanced Regression Techniques

第一阶段 baseline：对 `SalePrice` 使用 `log1p`，数值特征中位数填充并标准化，类别特征众数填充并 One-Hot 编码，最后训练 Ridge 回归。

第二个实验 `exp002_semantic_ridge.py` 在不修改原始 CSV 的前提下，删除 `Id`，把“没有车库/地下室/游泳池”等设施缺失编码为 `None` 或 `0`，并将 `MSSubClass`、`MoSold`、`YrSold` 作为类别变量处理。

第三个实验 `exp003_feature_engineering_ridge.py` 在此基础上加入总面积、总卫浴、总阳台面积、房龄、翻修龄、车库龄和质量交互特征，并对明显右偏的非负连续变量进行 `log1p`。

## 运行

```powershell
python -m pip install -r requirements.txt
python src/baseline_ridge.py
python src/exp002_semantic_ridge.py
python src/exp003_feature_engineering_ridge.py
```

脚本使用 5 折 `KFold(shuffle=True, random_state=42)` 输出本地 log-RMSE，并生成：

```text
outputs/submissions/baseline_ridge.csv
```

该 CSV 只作为本地检查和后续提交候选，不会自动上传到 Kaggle。

## 数据

Kaggle 原始数据放在 `data/raw`，不纳入 Git。比赛指标是预测房价对数的 RMSE。
