# -*- coding: utf-8 -*-
r"""
================================================================================
 identify_inference.py  ——  【推理】手写数字识别 + GUI
================================================================================

【职责（只管一件事）】
  读取 identify_train.py 训练好的权重 mnist_mlp.npz，对手写输入做预测。
  本文件【不做任何训练】，也不碰 MNIST 原始数据。

【本文件内部也是解耦的，分成两块】
  ┌───────────────────────────────────────────────────────────────┐
  │ 第一部分：推理逻辑（MnistInference 类）                        │
  │   - 只依赖 numpy（+ PIL 用于图像预处理）                       │
  │   - 完全不知道 tkinter / 窗口 / 鼠标 的存在                    │
  │   - 可以脱离 GUI 单独使用（见下方"无界面用法"）                │
  ├───────────────────────────────────────────────────────────────┤
  │ 第二部分：GUI（DigitRecognizerGUI 类）                         │
  │   - 只管界面：画布、按钮、鼠标事件、结果展示                   │
  │   - 自己不写任何数学，全部通过 self.infer 调用第一部分         │
  └───────────────────────────────────────────────────────────────┘

【无界面用法（证明推理逻辑与 GUI 解耦）】
    from PIL import Image
    from identify_inference import MnistInference
    infer = MnistInference()                 # 加载权重
    img = Image.open("some_digit.png").convert("L")
    digit, probs, x784 = infer.predict_image(img)
    print(digit, probs)                      # 没有窗口，一样能识别

【用法】
    E:\Python\python.exe identify_inference.py
================================================================================
"""

import os

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

HERE        = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "mnist_mlp.npz")   # 与 identify_train.py 约定同一路径

IMG_SIZE = 28   # 网络输入尺寸


# ##############################################################################
# #                          第一部分：推理逻辑                                 #
# #                    （不依赖任何 GUI 库，可单独使用）                         #
# ##############################################################################

def load_params(path=PARAMS_PATH):
    """读取权重文件，返回包含 6 个数组的字典：W1,b1,W2,b2,W3,b3。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到权重文件 {path}\n请先运行 identify_train.py 训练出权重。")
    data = np.load(path)
    return {k: data[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}


def relu(z):
    """ReLU：f(z)=max(0,z)。"""
    return np.maximum(0.0, z)


def softmax(z):
    """输入 z 形状 (N,10)（10 个类别的"得分"），返回 (N,10) 概率（每行和为 1）。"""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(X, params):
    """
    前向传播：和训练时【必须完全一致】。
      z1 = X · W1 + b1        a1 = ReLU(z1)      第 1 个隐藏层
      z2 = a1 · W2 + b2       a2 = ReLU(z2)      第 2 个隐藏层
      z3 = a2 · W3 + b3       p  = softmax(z3)   输出层
    返回 (N,10) 概率数组。
    注意：隐藏层数（这里 2 层）写在代码里，由权重形状决定；改了层数这里也要跟着改。
    """
    z1 = X  @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    return softmax(z3)


def preprocess(pil_img):
    """
    把一张 PIL 灰度图（画布，白字黑底，值 0~255）转成形状 (784,) 的 float32 数组（0~1）。
    若画布全黑则返回 None。
    必须模仿 MNIST 的标准化（否则分布对不上，识别率会很低）：
      1) 找非黑像素的外接框，裁出来；
      2) 等比缩放到最长边 20 像素；
      3) 按"重心"对齐，贴到 28x28 正中间；
      4) 归一化到 0~1 并拉平成 784。
    """
    arr = np.array(pil_img, dtype=np.uint8)

    ys, xs = np.where(arr > 20)                  # 阈值 20 滤噪点
    if len(xs) == 0:
        return None
    digit = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    d = np.array(Image.fromarray(digit).resize((new_w, new_h), Image.Resampling.LANCZOS),
                 dtype=np.float32)

    total = d.sum()
    cy = (d.sum(axis=1) * np.arange(new_h)).sum() / total   # 竖直重心
    cx = (d.sum(axis=0) * np.arange(new_w)).sum() / total   # 水平重心
    top, left = int(round(IMG_SIZE / 2 - cy)), int(round(IMG_SIZE / 2 - cx))

    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    y0, y1 = max(0, top), min(IMG_SIZE, top + new_h)
    x0, x1 = max(0, left), min(IMG_SIZE, left + new_w)
    if y0 < y1 and x0 < x1:
        canvas[y0:y1, x0:x1] = d[y0 - top:y1 - top, x0 - left:x1 - left]

    return (canvas / 255.0).reshape(-1)


class MnistInference:
    """
    推理器：只负责"图像 -> 概率"。与界面完全无关。
    以后想接别的界面（网页 / 命令行），直接复用这个类即可。
    """
    def __init__(self, params_path=PARAMS_PATH):
        self.params_path = params_path
        self.params = load_params(params_path)

    def reload(self):
        """重新从磁盘加载权重（比如在别处重新训练之后）。"""
        self.params = load_params(self.params_path)

    def predict_probs(self, x784):
        """输入 784 维数组（或 (1,784)），返回 10 个概率组成的数组（形状 (10,)）。"""
        x = np.asarray(x784, dtype=np.float32).reshape(1, -1)
        return forward(x, self.params)[0]

    def predict_image(self, pil_img):
        """
        输入一张 PIL 灰度图（画布），输出 (数字, 概率数组, 预处理后的784输入)。
        没画东西时返回 (None, None, None)。
        """
        x784 = preprocess(pil_img)
        if x784 is None:
            return None, None, None
        probs = self.predict_probs(x784)
        return int(probs.argmax()), probs, x784


# ##############################################################################
# #                          第二部分：GUI                                      #
# #             （只管界面；所有计算都交给 MnistInference）                     #
# ##############################################################################

SCALE     = 10                # 28 -> 280，放大显示，1 个 MNIST 像素 = 10 个画布像素
CANVAS_PX = IMG_SIZE * SCALE  # 280
BRUSH     = 20                # 画笔粗细（画布像素）

class DigitRecognizerGUI:
    def __init__(self, infer):
        """
        输入一个 MnistInference 实例：GUI 通过它拿结果，自己不碰数学。
        """
        self.infer = infer
        self.last = None

        # 图片缓冲区：画布上每一笔也同步画进这张 PIL 图，识别时以它为准
        self.buffer = Image.new("L", (CANVAS_PX, CANVAS_PX), 0)
        self.buf_draw = ImageDraw.Draw(self.buffer)

        # ---------- 搭窗口 ----------
        self.root = tk.Tk()
        self.root.title("MNIST 手写数字识别（推理）")
        self.root.configure(bg="#f5f5f5")

        left = tk.Frame(self.root, bg="#f5f5f5")
        left.grid(row=0, column=0, padx=14, pady=14)
        right = tk.Frame(self.root, bg="#f5f5f5")
        right.grid(row=0, column=1, padx=14, pady=14, sticky="n")

        # 黑色画布 + 白色笔画，与 MNIST 的白字黑底一致
        self.canvas = tk.Canvas(left, width=CANVAS_PX, height=CANVAS_PX, bg="black",
                                highlightthickness=1, highlightbackground="#bbb",
                                cursor="crosshair")
        self.canvas.grid(row=0, column=0, columnspan=3)
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        tk.Button(left, text="识别", width=10, command=self.recognize).grid(row=1, column=0, pady=10)
        tk.Button(left, text="清空", width=10, command=self.clear).grid(row=1, column=1, pady=10)
        tk.Button(left, text="重载权重", width=10, command=self.reload_weights).grid(row=1, column=2, pady=10)
        tk.Label(left, text="在黑框里写一个 0~9，松手自动识别", bg="#f5f5f5", fg="#666") \
            .grid(row=2, column=0, columnspan=3)

        # 权重文件提示
        self.status = tk.Label(left, text=f"权重：{os.path.basename(infer.params_path)}",
                               bg="#f5f5f5", fg="#999", font=("Segoe UI", 9))
        self.status.grid(row=3, column=0, columnspan=3, pady=(6, 0))

        # ---------- 右侧结果 + 概率条 ----------
        self.result = tk.Label(right, text="—", font=("Segoe UI", 48, "bold"),
                               bg="#f5f5f5", fg="#d32f2f")
        self.result.grid(row=0, column=0, pady=(0, 4))
        tk.Label(right, text="识别结果", bg="#f5f5f5", fg="#888").grid(row=1, column=0, pady=(0, 10))
        tk.Label(right, text="各类别概率", bg="#f5f5f5", fg="#666").grid(row=2, column=0, sticky="w")
        self.bars = tk.Canvas(right, width=260, height=10 * 24, bg="#f5f5f5", highlightthickness=0)
        self.bars.grid(row=3, column=0, pady=6)
        self.draw_bars(np.zeros(10))

    # ---------- 鼠标事件（只负责画，识别交给 infer）----------
    def on_press(self, event):
        """鼠标按下：记下起点，并画一个点。"""
        self.last = (event.x, event.y)
        self._draw_segment(event.x, event.y, event.x, event.y)

    def on_drag(self, event):
        """鼠标拖动：从上一个点到当前点画一段线。"""
        if self.last is not None:
            self._draw_segment(self.last[0], self.last[1], event.x, event.y)
        self.last = (event.x, event.y)

    def on_release(self, event):
        """鼠标松开：清空起点，并自动识别一次。"""
        self.last = None
        self.recognize()

    def _draw_segment(self, x0, y0, x1, y1):
        """把一个线段同时画到显示用画布和 PIL 缓冲区。"""
        self.canvas.create_line(x0, y0, x1, y1, fill="white", width=BRUSH,
                                capstyle=tk.ROUND, smooth=True)
        self.buf_draw.line([x0, y0, x1, y1], fill=255, width=BRUSH, joint="curve")

    # ---------- 识别：一行调用推理逻辑，其余都是界面工作 ----------
    def recognize(self):
        """识别当前画布内容并更新界面。"""
        digit, probs, _ = self.infer.predict_image(self.buffer)   # ← 全部计算在这里完成
        if digit is None or probs is None:
            self.result.config(text="—")
            self.draw_bars(np.zeros(10))
            return
        self.result.config(text=str(digit))
        self.draw_bars(probs)
        print(f"[识别] 结果={digit}  置信度={probs[digit]*100:.1f}%")

    def draw_bars(self, probs):
        """在右侧画出 0~9 的概率条。"""
        c = self.bars
        c.delete("all")
        top = int(probs.argmax())
        for k in range(10):
            y = k * 24 + 6
            c.create_text(12, y + 5, text=str(k), fill="#333", font=("Segoe UI", 10))
            c.create_rectangle(26, y, 226, y + 10, outline="#ddd", fill="#f0f0f0")
            w = int(probs[k] * 200)
            color = "#d32f2f" if (k == top and probs[k] > 0) else "#3b78c3"
            c.create_rectangle(26, y, 26 + w, y + 10, outline="", fill=color)
            c.create_text(232, y + 5, text=f"{probs[k]*100:4.1f}%", fill="#666",
                          font=("Segoe UI", 9), anchor="w")

    def clear(self):
        """清空画布和结果。"""
        self.canvas.delete("all")
        self.buf_draw.rectangle([0, 0, CANVAS_PX, CANVAS_PX], fill=0)
        self.result.config(text="—")
        self.draw_bars(np.zeros(10))

    def reload_weights(self):
        """重新从磁盘加载权重并刷新提示文字。"""
        try:
            self.infer.reload()
            self.status.config(text=f"权重：{os.path.basename(self.infer.params_path)}（已重载）",
                               fg="#3B6D11")
            print("[界面] 已重新加载权重")
        except Exception as ex:
            self.status.config(text=f"重载失败：{ex}", fg="#A32D2D")

    def run(self):
        """进入 tkinter 事件循环（阻塞在这里直到关窗）。"""
        self.root.mainloop()


def main():
    """加载权重并打开窗口。"""
    infer = MnistInference()          # 先在界面之外把权重加载好
    print("[启动] 权重已加载，打开窗口")
    DigitRecognizerGUI(infer).run()


if __name__ == "__main__":
    main()
