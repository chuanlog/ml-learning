# hand-write-number-identify

用**纯 NumPy** 从零实现神经网络的 MNIST 手写数字识别。不依赖 PyTorch/TensorFlow，
目的是把**前向传播、反向传播（链式法则）、梯度下降**每一步都摊开在代码里，便于理解。

网络结构：`784 -> 128(ReLU) -> 64(ReLU) -> 10(softmax)`，两层隐藏层，共 3 组权重。

## 文件说明

| 文件 | 职责 |
|------|------|
| `identify_train.py` | **训练**：读 MNIST → 梯度下降 + 反向传播 → 输出权重 `mnist_mlp.npz`。不含任何 GUI。 |
| `identify_inference.py` | **推理 + GUI**：读权重 → 鼠标手写识别。内部又拆成「纯推理类 `MnistInference`」和「界面类 `DigitRecognizerGUI`」，互不耦合。 |
| `data_exploration.py` | 数据探索：看看 MNIST 长什么样（用 torchvision 加载，需 torch）。 |
| `data/MNIST/raw/` | MNIST 原始 IDX 数据文件。 |
| `mnist_mlp.npz` | 训练输出的权重（zip 格式，内含 W1/b1/W2/b2/W3/b3 六个数组）。 |

## 用法

> 需要 numpy、Pillow、tkinter（本机 `E:\Python\python.exe` 已具备）。

```bat
:: 1) 训练（约 24 秒，15 轮，测试集准确率 ~98%），生成 mnist_mlp.npz
E:\Python\python.exe identify_train.py

:: 2) 验证反向传播推导是否正确（数值梯度 vs 解析梯度）
E:\Python\python.exe identify_train.py --gradcheck

:: 3) 打开手写识别窗口
E:\Python\python.exe identify_inference.py
```

## 想学反向传播，重点看这几处（identify_train.py）

- `forward()` —— 前向传播，数据从左流到右
- `compute_loss_and_grads()` —— **反向传播核心**。三层梯度【故意展开写】，让你看清每层都在重复同样的两步：
  1. `da_prev = dz · Wᵀ`（把梯度乘本层权重转置，往回传）
  2. `dz_prev = da_prev · act'(z_prev)`（再乘激活函数导数）
- `train()` —— 梯度下降：`W = W - lr * dW`
- `grad_check()` —— 用数值求导证明上面的推导没问题
