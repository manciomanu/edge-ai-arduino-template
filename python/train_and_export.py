import numpy as np
from pathlib import Path

np.random.seed(42)

# Synthetic dataset (2 features, binary labels)
N = 200
class0 = np.random.normal([0.0, 0.0], [0.8, 0.8], (N, 2))
class1 = np.random.normal([1.5, 1.0], [0.8, 0.8], (N, 2))
X = np.vstack([class0, class1]).astype(np.float32)
y = np.hstack([np.zeros(N), np.ones(N)]).astype(np.float32)

# Add bias term
X_b = np.c_[np.ones((X.shape[0], 1)), X]
w = np.zeros(X_b.shape[1], dtype=np.float32)

def sigmoid(z): return 1 / (1 + np.exp(-z))

# Train (logistic regression via grad descent)
lr, epochs = 0.05, 600
for _ in range(epochs):
    z = X_b @ w
    p = sigmoid(z)
    grad = X_b.T @ (p - y) / len(y)
    w -= lr * grad

# Quantize weights to int8-like and export header
scale = 64.0 / max(1e-6, np.max(np.abs(w)))
w_q = np.clip(np.round(w * scale), -127, 127).astype(np.int8)

header = f"""
#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#define N_FEATURES 2
#define SCALE_FACTOR {scale:.6f}f
// [bias, w1, w2]
static const signed char WEIGHTS_Q[N_FEATURES+1] = {{
    {int(w_q[0])}, {int(w_q[1])}, {int(w_q[2])}
}};

#endif
""".strip()

Path(__file__).parent.joinpath("model_weights.h").write_text(header)
print("Exported model_weights.h")
