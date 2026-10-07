# -*- coding: utf-8 -*-
r"""
================================================================================
 identify_train.py  ——  纯 NumPy 神经网络【训练脚本】
================================================================================

【职责（只管一件事）】
  读取 MNIST -> 用梯度下降 + 反向传播训练 -> 把训练好的权重写到 mnist_mlp.npz。
  本文件【不包含任何 GUI / 推理代码】，训练和推理彻底分开。

【网络结构：两层隐藏层（3 组权重）】
      输入            隐藏层1           隐藏层2           输出
   784 像素   -->   128 神经元   -->    64 神经元   -->   10 类
   (28x28)          ReLU             ReLU             softmax

  为什么要用两层？因为"链式法则在反向传播里长什么样"，要至少两层才看得清：
  梯度从输出层出发，一路往回穿 W3、W2、W1，每穿一层就乘一次该层的"局部导数"。

【链式法则（本文件的核心）】
  loss 对 W1 的偏导，必须一层层"链"回去：
    ∂L/∂W1 = ∂L/∂p · ∂p/∂z3 · ∂z3/∂a2 · ∂a2/∂z2 · ∂z2/∂a1 · ∂a1/∂z1 · ∂z1/∂W1
  看着吓人，但拆开后你会发现每层只重复做两件事：
    (a) 把"从后面传回来的梯度"乘上本层权重转置，继续往前送： da_prev = dz · Wᵀ
    (b) 再乘上本层激活函数的导数：                        dz_prev = da_prev · act'(z_prev)
  把这两件事对每层做一遍，就等于手写了一个"迷你版 PyTorch autograd"。
  代码里我把三层【故意展开写】，不写成循环，就是为了让你一眼看清这个重复的模式。

================================================================================
【用法】
    E:\Python\python.exe identify_train.py              # 正常训练并保存权重
    E:\Python\python.exe identify_train.py --gradcheck  # 只做数值梯度校验（验证反向传播推导对不对）
================================================================================
"""

import os
import sys
import struct
import time

import numpy as np

# ==============================================================================
# 0. 超参数
# ==============================================================================
SEED      = 42      # 固定随机种子，保证结果可复现
HIDDEN1   = 128     # 隐藏层 1 神经元个数
HIDDEN2   = 64      # 隐藏层 2 神经元个数
EPOCHS    = 15      # 训练轮数
BATCH_SIZE= 64      # 一个小批量多少条数据更新一次
LR        = 0.3     # 学习率
IMG_SIZE  = 28
N_CLASSES = 10

HERE        = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(HERE, "data", "MNIST", "raw")   # 原始 idx 文件
PARAMS_PATH = os.path.join(HERE, "mnist_mlp.npz")          # 权重输出文件


# ==============================================================================
# 1. 读取 MNIST 原始数据（IDX 二进制格式，自己解析，不用 torchvision）
# ==============================================================================
# idx3-ubyte（图片）：magic(4B) + 图片数(4B) + 行(4B) + 列(4B) + 像素字节流
# idx1-ubyte（标签）：magic(4B) + 数量(4B) + 标签字节流
# ">IIII" 的 ">" 表示大端序，I 表示 4 字节无符号整数。
def load_idx_images(path):
    """读取 .idx 图片文件，返回形状 (N, 784) 的 uint8 数组，每个元素 0~255"""
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return buf.reshape(n, rows * cols)


def load_idx_labels(path):
    """读取 .idx 标签文件，返回形状 (N,) 的 uint8 数组，每个元素是 0~9 的标签"""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return buf


def load_mnist(raw_dir=RAW_DIR):
    """
    读训练集、测试集，并把像素 0~255 归一化成 0~1 的小数。
    返回 4 个值：(训练图, 训练标签, 测试图, 测试标签)
      训练图/测试图：float32，形状 (N, 784)
      训练标签/测试标签：uint8，形状 (N,)
    """
    print("[数据] 读取 MNIST ...")
    tr_x = load_idx_images(os.path.join(raw_dir, "train-images-idx3-ubyte"))
    tr_y = load_idx_labels(os.path.join(raw_dir, "train-labels-idx1-ubyte"))
    te_x = load_idx_images(os.path.join(raw_dir, "t10k-images-idx3-ubyte"))
    te_y = load_idx_labels(os.path.join(raw_dir, "t10k-labels-idx1-ubyte"))
    tr_x = tr_x.astype(np.float32) / 255.0
    te_x = te_x.astype(np.float32) / 255.0
    print(f"       训练 {tr_x.shape[0]} 张，测试 {te_x.shape[0]} 张")
    return tr_x, tr_y, te_x, te_y


# ==============================================================================
# 2. 初始化参数（两层隐藏层 -> 3 组 (W, b)）
# ==============================================================================
def init_params():
    """
    返回权重字典，包含 6 个数组：
      W1 (784,HIDDEN1)  b1 (HIDDEN1,)  W2 (HIDDEN1,HIDDEN2)  b2 (HIDDEN2,)  W3 (HIDDEN2,10)  b3 (10,)
    He 初始化：W ~ N(0, sqrt(2/输入维度))，配合 ReLU 让每层输出方差稳定；偏置一律置 0。
    """
    rng = np.random.default_rng(SEED)
    params = {
        # 输入(784) -> 隐藏1(HIDDEN1)
        "W1": rng.normal(0, np.sqrt(2.0 / 784), (784, HIDDEN1)).astype(np.float32),
        "b1": np.zeros(HIDDEN1, dtype=np.float32),
        # 隐藏1(HIDDEN1) -> 隐藏2(HIDDEN2)
        "W2": rng.normal(0, np.sqrt(2.0 / HIDDEN1), (HIDDEN1, HIDDEN2)).astype(np.float32),
        "b2": np.zeros(HIDDEN2, dtype=np.float32),
        # 隐藏2(HIDDEN2) -> 输出(10)
        "W3": rng.normal(0, np.sqrt(2.0 / HIDDEN2), (HIDDEN2, N_CLASSES)).astype(np.float32),
        "b3": np.zeros(N_CLASSES, dtype=np.float32),
    }
    return params


# ==============================================================================
# 3. 激活函数
# ==============================================================================
def relu(z):
    """ReLU：f(z)=max(0,z)。注意它的导数：z>0 处为 1，否则为 0。"""
    return np.maximum(0.0, z)


def softmax(z):
    """
    输入 z 形状 (N, 10)（10 个类别的"得分"），返回 (N, 10) 概率（每行和为 1）。
    减最大值是为了防止 exp 溢出到无穷大。
    """
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ==============================================================================
# 4. 前向传播：数据从输入流到输出
# ==============================================================================
def forward(X, params):
    """
    前向传播：返回 (概率 p, 中间结果 cache)。
      z1 = X · W1 + b1        a1 = ReLU(z1)      第 1 个隐藏层
      z2 = a1 · W2 + b2       a2 = ReLU(z2)      第 2 个隐藏层
      z3 = a2 · W3 + b3       p  = softmax(z3)   输出层
    cache 里存了 X, z1, a1, z2, a2, z3, p，反向传播要取出来用。
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    W3, b3 = params["W3"], params["b3"]

    z1 = X  @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    p  = softmax(z3)

    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "p": p}
    return p, cache


# ==============================================================================
# 5. 反向传播（链式法则展开，逐层往回传）
# ==============================================================================
def compute_loss_and_grads(X, y, params, cache):
    """
    输入一批图 X、真实标签 y、权重 params、forward 返回的 cache。
    返回 (loss, grads)：
      loss  —— 一个 float（交叉熵损失，越小越好）
      grads —— 梯度字典，键为 "dW1","db1","dW2","db2","dW3","db3"，形状与参数一致。

    ---------- 链式法则，逐层往回 ----------
    记 dzL 表示 ∂L/∂zL（"从后面传回来的梯度"）。

    ① 输出层：softmax + 交叉熵 合并求导，结果极其简洁：
           dz3 = (p - onehot) / N
       dW3 = a2ᵀ · dz3        db3 = Σ dz3

    ② 往第 2 个隐藏层传（链式法则：先乘 W3ᵀ，再过 ReLU 导数）：
           da2 = dz3 · W3ᵀ                 ← (a) 乘本层权重转置
           dz2 = da2 · (z2 > 0)            ← (b) 乘激活导数
           dW2 = a1ᵀ · dz2        db2 = Σ dz2

    ③ 往第 1 个隐藏层传（同样的两步，再重复一次）：
           da1 = dz2 · W2ᵀ
           dz1 = da1 · (z1 > 0)
           dW1 = Xᵀ · dz1         db1 = Σ dz1

    你看：(a)(b) 这两步在每一层都出现一次，这就是"链式法则"在反向传播里的真实样子。
    """
    N = X.shape[0]
    W2, W3 = params["W2"], params["W3"]
    z1, a1, z2, a2, p = cache["z1"], cache["a1"], cache["z2"], cache["a2"], cache["p"]

    # --- 损失 ---
    correct_logp = -np.log(p[np.arange(N), y] + 1e-12)
    loss = float(correct_logp.mean())

    # --- one-hot：标签 3 -> [0,0,0,1,0,0,0,0,0,0] ---
    y_onehot = np.zeros((N, N_CLASSES), dtype=np.float32)
    y_onehot[np.arange(N), y] = 1.0

    # ==================== ① 输出层 ====================
    dz3 = (p - y_onehot) / N            # (N,10)  ← softmax+交叉熵 的合并结果
    dW3 = a2.T @ dz3                    # (HIDDEN2,10)
    db3 = dz3.sum(axis=0)               # (10,)

    # ==================== ② 隐藏层 2 ====================
    da2 = dz3 @ W3.T                    # (N,HIDDEN2)  乘 W3 转置往回传
    dz2 = da2 * (z2 > 0)                # (N,HIDDEN2)  乘 ReLU 导数
    dW2 = a1.T @ dz2                    # (HIDDEN1,HIDDEN2)
    db2 = dz2.sum(axis=0)               # (HIDDEN2,)

    # ==================== ③ 隐藏层 1 ====================
    da1 = dz2 @ W2.T                    # (N,HIDDEN1)  乘 W2 转置往回传
    dz1 = da1 * (z1 > 0)                # (N,HIDDEN1)  乘 ReLU 导数
    dW1 = X.T @ dz1                     # (784,HIDDEN1)
    db1 = dz1.sum(axis=0)               # (HIDDEN1,)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return loss, grads


# ==============================================================================
# 6. 训练循环
# ==============================================================================
def train(tr_x, tr_y, te_x, te_y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, params=None):
    """
    用训练集训练，返回训练好的权重字典。
    参数：tr_x/tr_y 训练图/标签；te_x/te_y 测试图/标签；
          epochs 轮数；batch_size 每批条数；lr 学习率；
          params 初始权重，给 None 则内部随机初始化。
    1 个 epoch = 把全部训练数据按 batch_size 切成很多批，每批：
        前向 -> 算 loss -> 反向算梯度 -> 梯度下降更新参数
    注意：是【每批更新一次】，不是每条更新、也不是全量更新。
    """
    if params is None:
        params = init_params()
    n_train = tr_x.shape[0]
    n_batch = (n_train + batch_size - 1) // batch_size   # 向上取整，最后一批不足也用它

    for ep in range(1, epochs + 1):
        t0 = time.time()
        perm = np.random.permutation(n_train)            # 每轮先打乱
        tr_x_s, tr_y_s = tr_x[perm], tr_y[perm]

        epoch_loss, cnt = 0.0, 0
        # range(0, n, batch_size) 让最后一批自动"短一截"，不丢样本
        for s in range(0, n_train, batch_size):
            e = min(s + batch_size, n_train)
            xb, yb = tr_x_s[s:e], tr_y_s[s:e]

            p, cache = forward(xb, params)                               # ① 前向
            loss, grads = compute_loss_and_grads(xb, yb, params, cache)  # ② 反向

            # ③ 梯度下降：沿梯度反方向走一小步  W = W - lr * dW
            params["W1"] -= lr * grads["dW1"]
            params["b1"] -= lr * grads["db1"]
            params["W2"] -= lr * grads["dW2"]
            params["b2"] -= lr * grads["db2"]
            params["W3"] -= lr * grads["dW3"]
            params["b3"] -= lr * grads["db3"]

            epoch_loss += loss
            cnt += 1

        acc = evaluate(te_x, te_y, params)
        print(f"[训练] 第 {ep:2d}/{epochs} 轮 | 平均损失 {epoch_loss/cnt:.4f} "
              f"| 测试准确率 {acc*100:5.2f}% | 用时 {time.time()-t0:.1f}s")
    return params


def evaluate(x, y, params):
    """只做前向，返回准确率（0.0~1.0），不反向、不更新参数。"""
    p, _ = forward(x, params)
    return float((p.argmax(axis=1) == y).mean())


# ==============================================================================
# 7. 数值梯度校验（可选，用来证明反向传播推导是对的）
# ==============================================================================
def loss_only(X, y, params):
    """只返回损失（float），供数值求导用。"""
    p, _ = forward(X, params)
    N = X.shape[0]
    return float(-np.log(p[np.arange(N), y] + 1e-12).mean())


def grad_check(params, eps=1e-5, per_param=3, seed=0):
    """
    数值梯度校验：随机抽查参数位置，用 ∂L/∂θ ≈ (L(θ+ε)-L(θ-ε))/(2ε)
    算数值梯度，再和反向传播算出的解析梯度对比，一致则说明推导正确。
    打印最大相对误差。

    注意：数值求导对精度极其敏感，必须用 float64。
    float32 只有约 7 位有效数字，(L(θ+ε)-L(θ-ε)) 是个很小的差值，
    会被舍入误差淹没，导致算出来的"数值梯度"本身就不准（会得到 0.5 这种假误差）。
    """
    print("\n[校验] 数值梯度 vs 反向传播梯度（float64）")
    rng = np.random.default_rng(seed)
    p64 = {k: v.astype(np.float64) for k, v in params.items()}   # 关键：升到 float64
    X = rng.normal(0, 1, size=(8, 784)).astype(np.float64)
    y = rng.integers(0, N_CLASSES, size=8)
    _, cache = forward(X, p64)
    _, grads = compute_loss_and_grads(X, y, p64, cache)

    max_rel = 0.0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        arr = p64[key]
        for _ in range(per_param):
            pos = tuple(int(i) for i in rng.integers(0, arr.shape, size=arr.ndim))
            orig = float(arr[pos])
            arr[pos] = orig + eps; lp = loss_only(X, y, p64)
            arr[pos] = orig - eps; lm = loss_only(X, y, p64)
            arr[pos] = orig
            num = (lp - lm) / (2 * eps)
            ana = float(grads["d" + key][pos])
            rel = abs(num - ana) / (max(abs(num), abs(ana)) + 1e-8)
            max_rel = max(max_rel, rel)
    print(f"        最大相对误差 = {max_rel:.2e}   (< 1e-4 即说明反向传播正确)\n")


# ==============================================================================
# 8. 保存权重
# ==============================================================================
def save_params(params, path=PARAMS_PATH):
    """把 6 个数组写成一个 .npz（本质是 zip），供推理脚本读取。"""
    np.savez(path, **params)
    print(f"[保存] 权重已写入 {path}")


# ==============================================================================
# 9. 入口
# ==============================================================================
def main():
    """从命令行 sys.argv 读参数；--gradcheck 则只做梯度校验，否则正常训练。"""
    # --gradcheck：随机初始化后只做梯度校验，不训练
    if "--gradcheck" in sys.argv:
        grad_check(init_params())
        return

    tr_x, tr_y, te_x, te_y = load_mnist()
    params = train(tr_x, tr_y, te_x, te_y)
    print(f"\n[结果] 最终测试集准确率：{evaluate(te_x, te_y, params)*100:.2f}%")
    save_params(params)
    print("[提示] 接下来运行 identify_inference.py 打开手写识别窗口")


if __name__ == "__main__":
    main()
