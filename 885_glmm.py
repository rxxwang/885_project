import numpy as np
import pandas as pd

file_path = 'heart.csv'
df = pd.read_csv(file_path)

print(df.columns)

for col in df.columns:
    globals()[col] = df[col].to_numpy()

n = age.size
x = transplant
z = np.round(age, decimals=0)
y = fustat

z0 = np.sort(np.unique(z))
N = np.zeros((n, len(z0)), dtype=int)

for i in range(n):
    for j, val in enumerate(z0):
        if z[i] == val:
            N[i, j] = 1

m = len(z0)

h = z0[1:] - z0[:-1]

R = np.zeros((m - 2, m - 2))
for i in range(m - 2):
    R[i, i] = (1 / 3) * (h[i] + h[i + 1])
    if i < m - 3:
        R[i, i + 1] = (1 / 6) * h[i + 1]
        R[i + 1, i] = (1 / 6) * h[i + 1]

Q = np.zeros((m, m - 2))
for i in range(m - 2):
    Q[i, i] = 1 / h[i]
    Q[i + 1, i] = -1 / h[i] - 1 / h[i + 1]
    Q[i + 2, i] = 1 / h[i + 1]

from scipy.linalg import cholesky, solve

R_inv = np.linalg.inv(R)
R_chol = cholesky(R_inv, lower=True)
L = Q @ R_chol.T

LTL = L.T @ L
LTL_inv = np.linalg.inv(LTL)
B = L @ LTL_inv

xz = x * z
NB = N @ B
diag_x = np.diag(x)
xNB = diag_x @ N @ B

import torch
import torch.nn as nn

x = torch.tensor(x.astype(float))
z = torch.tensor(z.astype(float))
xz = torch.tensor(xz.astype(float))
NB = torch.tensor(NB.astype(float))
NB = NB.to(torch.float32)
xNB = torch.tensor(xNB.astype(float))
xNB = xNB.to(torch.float32)
y_true = torch.tensor(y.astype(float))

class GLMM(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.beta0 = nn.Parameter(torch.zeros(1))
        self.beta1 = nn.Parameter(torch.zeros(1))
        self.beta2 = nn.Parameter(torch.zeros(1))
        self.beta3 = nn.Parameter(torch.zeros(1))
        self.a0 = nn.Parameter(torch.zeros(dim,1))
        nn.init.xavier_uniform_(self.a0)
        self.a1 = nn.Parameter(torch.zeros(dim,1))
        nn.init.xavier_uniform_(self.a1)
        self.sigmoid = nn.Sigmoid()

    # Define a forward method that outlines the forward pass
    def forward(self, intercept, x, z, xz, NB, xNB):
        out = self.beta0 * intercept
        out = out + self.beta1 * x
        out = out + self.beta2 * z
        out = out + self.beta3 * xz
        out = out + torch.matmul(NB, self.a0).squeeze()
        out = out + torch.matmul(xNB, self.a1).squeeze()
        out = self.sigmoid(out)
        return out

model = GLMM(dim = m-2)
epochs = 1000

intercept = torch.ones(n)
lossfn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lamda = 1
losses = np.zeros(epochs)

for i in range(epochs):
    y_pred = model(intercept, x, z, xz, NB, xNB)
    loss = lossfn(y_pred, y_true)
    l2 = torch.sum(model.a0 ** 2) + torch.sum(model.a1 ** 2)
    loss = loss + l2 * lamda
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses[i] = loss

    if i % 10 == 9 :
        print(f"Epoch: {i+1} | BCELoss: {loss:.4f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.title("BCE Loss")
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
loss_text = f"Epoch: {epochs} | BCE Loss: {losses[epochs-1]:.4f}"
plt.text(epochs/3, 1, loss_text)
plt.legend()
plt.show()

print(model.beta0)
print(model.beta1)
print(model.beta2)
print(model.beta3)
print(model.a0)
print(model.a1)

random1 = model.a0.squeeze().tolist()
random2 = model.a1.squeeze().tolist()

print(random1)

print(random2)
