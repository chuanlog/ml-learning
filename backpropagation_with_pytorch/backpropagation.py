import torch

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]

w = torch.Tensor([0.0, 0.0])  # w,b
w.requires_grad = True


# y= w*x+b
def forward(x):
    x_tensor = torch.Tensor([x, 1])
    return (x_tensor * w).sum()  # 求和得到标量 w*x+b


# 前向+损失
def loss(x, y):
    y_pred = forward(x)
    return ((y_pred - y) ** 2).sum()  # 归约成标量，才能直接 backward

print("predicted values before train:", forward(4).item())
learning_rate = 0.0001
for epoch in range(100000):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        print('loss=', l.item())
        l.backward()
        grad = w.grad
        assert grad is not None
        print('\tgrad:', x, y, grad.tolist())
        with torch.no_grad():
            w -= learning_rate * grad
        grad.zero_()  # 梯度是累加的，每步更新后必须清零

print("predicted values after train:", forward(4).item())

