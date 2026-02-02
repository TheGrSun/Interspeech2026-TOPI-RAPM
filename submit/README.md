# 提交文件生成指南

## 📋 提交文件要求总结

### 核心要求
1. **输入**: 测试集EN特征文件 (`EN_*.npy`, 1024维)
2. **输出**: 预测ES特征文件 (`ES_*.npy`, 101维)
3. **格式**: 
   - 形状: `(101,)` - 一维数组
   - 类型: `np.float64` - 必须是float64
   - 命名: 将`EN_`替换为`ES_`，其他保持不变
4. **打包**: 所有`.npy`文件打包成zip文件

### 快速生成提交

```bash
# 使用最佳模型 (ensemble_v1, 0.8863)
cd /home/luoxiaoyang/interspeech2026/submit
python generate_rapm_submission.py
```

这将：
- 使用 `checkpoints/ensemble_v1/best_model.pth` (SOTA: 0.8863)
- 从 `test-features/` 读取测试数据
- 输出到 `predictions_rapm/`
- 生成 `submission_rapm.zip`

### 自定义参数

```bash
python generate_rapm_submission.py \
    --model_path ../checkpoints/ensemble_v1/best_model.pth \
    --test_dir ../test-features \
    --output_dir predictions_rapm \
    --zip_path submission_rapm.zip
```

## 📁 文件说明

- `generate_rapm_submission.py` - R-APM v2 SOTA模型提交脚本
- `generate_submission.py` - 简单检索基线提交脚本
- `generate_mlp_submission.py` - 官方MLP基线提交脚本
- `SUBMISSION_REQUIREMENTS.md` - 详细提交要求文档

## 🎯 当前最佳模型

**推荐使用**: `checkpoints/ensemble_v1/best_model.pth` (性能: 0.8863)

详细性能对比见 `SUBMISSION_REQUIREMENTS.md`
