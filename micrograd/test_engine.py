import torch
from engine import Value
from nn import Neuron, Layer, MLP
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

def f(xi):
    #return int(bool(xi[0]) != bool(xi[1])) # xor
    return xi[0] and xi[1] # and
    #return 3*xi[0] # 3x

def normalize(xi):
    if len(xi) < 2:
        return xi
    mean = sum(xi)/len(xi)
    std = (sum([(xii - mean)**2 for xii in xi])/len(xi))**0.5
    if std == 0:
        return xi
    return [(xii - mean)/std for xii in xi]

def train(x, y, lr, e):
    n = Neuron(len(x))
    costs = []
    # training loop for e epochs
    for _ in range(e):
        cost = 0
        # forward pass to compute "avg loss" (cost) over all inputs in batch
        for i in range(len(x)):
            xi = x[i]# normalize(x[i])
            prediction = n(xi)
            #print(y[i], prediction)
            cost += (y[i] - prediction)**2
        cost = cost/len(x)
        costs.append(cost.data)
        # backprop to compute gradients of cost w.r.t weights
        cost.backward_iterative()
        for i in range(len(n.w)):
            # update weights as per gradient descent
            n.w[i] += -lr * n.w[i].grad
            # n.b += -lr * n.b.grad
    
    #print(costs)
    #print(n(x[0]), y[0])
    show_loss_curve(costs)
    return n

def test(x, y, n):
    avg_loss = 0
    for i in range(len(x)):
        print(x[i], n(x[i]).threshold_sig().data, y[i])
        avg_loss += (y[i] - n(x[i]).threshold_sig())**2
    avg_loss = avg_loss/len(x)
    print("Avg test loss: ", avg_loss.data)

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

training_set_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
#training_set_x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
training_set_y = [f(xi) for xi in training_set_x]
n = train(x=training_set_x, y=training_set_y, lr=0.1, e=9500)
test_set_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
#test_set_x = [[11], [12], [13], [14], [15], [16], [17], [18], [19], [20]]
test_set_y = [f(xi) for xi in test_set_x]
test(test_set_x, test_set_y, n)