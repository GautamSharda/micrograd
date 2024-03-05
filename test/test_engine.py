import torch
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP
import matplotlib.pyplot as plt

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def f(x):
    return 2*x

def test_linear():

    n = Neuron(1, False)
    x = [1]

    costs = []

    for i in range(0, 10):
        p = n(x)
        l = (f(x)[0] - p)**2
        costs.append(l)
        print(f"weights {i}: ", n.w, f"prediction {i}: ", p, f"cost {i}: ", l)
        p.backward()
        for w in n.w:
            print("w", w, "w.grad", w.grad)
            w += -0.1*w.grad
            print(w)

    # Generating x coordinates from 0 to the length of the list - 1
    x = list(range(len(costs)))
    y = list(map(lambda cost: cost.data, costs))

    # Plotting each value on the x,y plane
    plt.plot(x, y, marker='o')  # 'o' creates a circle marker for each point
    plt.title('Plot of Values')
    plt.xlabel('Index (x)')
    plt.ylabel('Value (y)')

    # Displaying the plot
    plt.show()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

test_sanity_check()
test_linear()