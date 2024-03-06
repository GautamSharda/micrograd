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
    return 3*x

def normalize(x):
    mean = sum(x)/len(x)
    std = (sum([(xi - mean)**2 for xi in x])/len(x))**0.5
    return [(xi - mean)/std for xi in x]

def train(x, lr, e):

    x_norm = normalize(x)

    n = Neuron(len(x), False)
    costs = []
    for _ in range(e):
        cost = 0
        for i in range(len(x_norm)):
            xi = x_norm[i]
            prediction = n([xi])
            cost += (f(xi) - prediction)**2
        cost = cost/len(x_norm)
        costs.append(cost.data)

        cost.backward()
        for i in range(len(n.w)):
            n.w[i] += -lr * n.w[i].grad
    
    print(costs)
    print(n([x[0]]), f(x[0]))
    show_loss_curve(costs)
    return n

def test(x, n):
    avg_loss = 0
    for xi in x:
        print(xi, n([xi]).data, f(xi))
        avg_loss += (f(xi) - n([xi]))**2
    avg_loss = avg_loss/len(x)

def show_loss_curve(costs):
    x_coords = list(range(len(costs)))
    y_coords = [cost for cost in costs]

    plt.plot(x_coords, y_coords, marker='o')
    plt.title('Plot of Values')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

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

training_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n = train(x=training_set, lr=0.5, e=3)
test_set = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
test(test_set, n)
# test_set = [[11], [12], [13], [14], [15], [16], [17], [18], [19], [20]]

# trained_neuron = train_neuron(training_set)
# predictions = test_neuron(trained_neuron, test_set)
# print(predictions)
# train_neuron_single()