import numpy as np
from sklearn.preprocessing import OneHotEncoder

# =============================================================================
# ФУНКЦИИ АКТИВАЦИИ И ИХ ПРОИЗВОДНЫЕ
# =============================================================================

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    t = tanh(x)
    return 1 - t * t

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def softmax(x):
    shift = np.max(x, axis=0, keepdims=True)
    exps = np.exp(x - shift)
    return exps / np.sum(exps, axis=0, keepdims=True)

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_grad),
    'tanh': (tanh, tanh_grad),
    'relu': (relu, relu_grad),
    'leaky_relu': (leaky_relu, leaky_relu_grad),
    'softmax': (softmax, None),
    'linear': (lambda x: x, lambda x: np.ones_like(x))
}

def get_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}")
    return ACTIVATIONS[name]


# =============================================================================
# КЛАСС MyMLP
# =============================================================================

class MyMLP:
    def __init__(
        self,
        layer_sizes,
        activations,
        weight_init='xavier',
        optimizer='sgd',
        l2_lambda=0.0,
        dropout_rate=0.0,
        batch_norm=False,
        **opt_params
    ):
        """
        layer_sizes: list[int] — напр., [784, 128, 64, 10]
        activations: list[str] — длина = len(layer_sizes)-1, напр. ['relu', 'relu', 'softmax']
        weight_init: 'xavier' | 'he'
        optimizer: 'sgd' | 'momentum' | 'rmsprop' | 'adam'
        l2_lambda: float — коэффициент L2-регуляризации
        dropout_rate: float ∈ [0, 1) — вероятность отключения нейрона (только в скрытых слоях)
        batch_norm: bool — использовать упрощённый BatchNorm в скрытых слоях
        opt_params: lr, momentum, beta1, beta2, eps, rho и т.д.
        """
        assert len(activations) == len(layer_sizes) - 1
        assert activations[-1] == 'softmax', "Last activation must be 'softmax' для multi-class"
        
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weight_init = weight_init
        self.optimizer = optimizer.lower()
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.use_batch_norm = batch_norm

        # Гиперпараметры оптимизатора
        self.lr = opt_params.get('lr', 0.001)
        self.momentum = opt_params.get('momentum', 0.9)
        self.beta1 = opt_params.get('beta1', 0.9)
        self.beta2 = opt_params.get('beta2', 0.999)
        self.eps = opt_params.get('eps', 1e-8)
        self.rho = opt_params.get('rho', 0.9)

        # Инициализация весов и смещений
        self.W, self.b = [], []
        self._init_weights()

        # Параметры BatchNorm
        self.gamma = []
        self.beta  = []
        if self.use_batch_norm:
            for n_out in layer_sizes[1:-1]:
                self.gamma.append(np.ones((n_out, 1)))
                self.beta.append(np.zeros((n_out, 1)))
            self.gamma.append(None)
            self.beta.append(None)
        else:
            self.gamma = [None] * len(self.W)
            self.beta  = [None] * len(self.W)

        # Состояния оптимизатора
        zeros_like_W = [np.zeros_like(w) for w in self.W]
        zeros_like_b = [np.zeros_like(bi) for bi in self.b]
        self.vW = zeros_like_W[:]
        self.vb = zeros_like_b[:]
        self.mW = zeros_like_W[:]
        self.mb = zeros_like_b[:]
        self.sW = zeros_like_W[:]
        self.sb = zeros_like_b[:]

        # Кэш для forward/backward
        self.Z = []
        self.Z_bn = []
        self.A = []
        self.dropout_masks = []
        self._label_encoder = None

    def _init_weights(self):
        self.W.clear()
        self.b.clear()
        for i in range(len(self.layer_sizes) - 1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i+1]
            if self.weight_init == 'xavier':
                scale = np.sqrt(6.0 / (n_in + n_out))
                W = np.random.uniform(-scale, scale, (n_out, n_in))
            elif self.weight_init == 'he':
                scale = np.sqrt(2.0 / n_in)
                W = np.random.randn(n_out, n_in) * scale
            else:
                raise ValueError(f"Unknown weight_init: {self.weight_init}")
            b = np.zeros((n_out, 1))
            self.W.append(W)
            self.b.append(b)

    def _batch_norm(self, z, gamma, beta):
        if gamma is None or beta is None:
            return z
        mu = np.mean(z, axis=1, keepdims=True)
        var = np.var(z, axis=1, keepdims=True) + self.eps
        z_norm = (z - mu) / np.sqrt(var)
        return gamma * z_norm + beta

    def forward(self, X, training=True):
        self.Z.clear()
        self.Z_bn.clear()
        self.A.clear()
        self.dropout_masks.clear()

        A = X
        self.A.append(A)

        for l in range(len(self.W)):
            Z = self.W[l] @ A + self.b[l]
            self.Z.append(Z.copy())

            if self.use_batch_norm and l < len(self.W) - 1:
                Z = self._batch_norm(Z, self.gamma[l], self.beta[l])
            self.Z_bn.append(Z.copy())

            act_fn, _ = get_activation(self.activations[l])
            A = act_fn(Z)

            if training and l < len(self.W) - 1 and self.dropout_rate > 0:
                keep_prob = 1.0 - self.dropout_rate
                mask = (np.random.rand(*A.shape) < keep_prob).astype(np.float32)
                A = (A * mask) / keep_prob
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)

            self.A.append(A)

        return A

    def backward(self, y_true):
        m = y_true.shape[1]
        L = len(self.W)
        dW = [np.zeros_like(w) for w in self.W]
        db = [np.zeros_like(bi) for bi in self.b]
        dgamma = [np.zeros_like(g) if g is not None else None for g in self.gamma]
        dbeta  = [np.zeros_like(be) if be is not None else None for be in self.beta]

        dZ = self.A[-1] - y_true

        for l in reversed(range(L)):
            dW[l] = (dZ @ self.A[l].T) / m
            if self.l2_lambda > 0:
                dW[l] += self.l2_lambda * self.W[l] / m
            db[l] = np.sum(dZ, axis=1, keepdims=True) / m

            if self.use_batch_norm and l < L - 1 and self.gamma[l] is not None:
                Z = self.Z[l]
                mu = np.mean(Z, axis=1, keepdims=True)
                var = np.var(Z, axis=1, keepdims=True) + self.eps
                std = np.sqrt(var)
                z_norm = (Z - mu) / std

                dgamma[l] = np.sum(dZ * z_norm, axis=1, keepdims=True) / m
                dbeta[l]  = np.sum(dZ, axis=1, keepdims=True) / m

                N = Z.shape[1]
                dZ_norm = dZ * self.gamma[l]
                dZ = (N * dZ_norm - np.sum(dZ_norm, axis=1, keepdims=True)
                      - z_norm * np.sum(dZ_norm * z_norm, axis=1, keepdims=True)) / (N * std)

            if self.dropout_masks[l] is not None:
                mask = self.dropout_masks[l]
                dZ = dZ * mask / (1.0 - self.dropout_rate)

            if l > 0:
                dA = self.W[l].T @ dZ
                _, grad_fn = get_activation(self.activations[l-1])
                Z_prev = self.Z[l-1]
                dZ = dA * grad_fn(Z_prev)

        return dW, db, dgamma, dbeta

    def update_weights(self, dW, db, dgamma, dbeta, step=None):
        L = len(self.W)
        for l in range(L):
            if self.optimizer == 'sgd':
                self.W[l] -= self.lr * dW[l]
                self.b[l] -= self.lr * db[l]

            elif self.optimizer == 'momentum':
                self.vW[l] = self.momentum * self.vW[l] + self.lr * dW[l]
                self.vb[l] = self.momentum * self.vb[l] + self.lr * db[l]
                self.W[l] -= self.vW[l]
                self.b[l] -= self.vb[l]

            elif self.optimizer == 'rmsprop':
                self.vW[l] = self.rho * self.vW[l] + (1 - self.rho) * (dW[l] ** 2)
                self.vb[l] = self.rho * self.vb[l] + (1 - self.rho) * (db[l] ** 2)
                self.W[l] -= self.lr * dW[l] / (np.sqrt(self.vW[l]) + self.eps)
                self.b[l] -= self.lr * db[l] / (np.sqrt(self.vb[l]) + self.eps)

            elif self.optimizer == 'adam':
                if step is None:
                    raise ValueError("Adam: step (1-based) required")
                self.mW[l] = self.beta1 * self.mW[l] + (1 - self.beta1) * dW[l]
                self.mb[l] = self.beta1 * self.mb[l] + (1 - self.beta1) * db[l]
                self.sW[l] = self.beta2 * self.sW[l] + (1 - self.beta2) * (dW[l] ** 2)
                self.sb[l] = self.beta2 * self.sb[l] + (1 - self.beta2) * (db[l] ** 2)
                
                mW_hat = self.mW[l] / (1 - self.beta1 ** step)
                mb_hat = self.mb[l] / (1 - self.beta1 ** step)
                sW_hat = self.sW[l] / (1 - self.beta2 ** step)
                sb_hat = self.sb[l] / (1 - self.beta2 ** step)
                
                self.W[l] -= self.lr * mW_hat / (np.sqrt(sW_hat) + self.eps)
                self.b[l] -= self.lr * mb_hat / (np.sqrt(sb_hat) + self.eps)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")

            if self.use_batch_norm and l < L - 1 and self.gamma[l] is not None:
                if self.optimizer == 'sgd':
                    self.gamma[l] -= self.lr * dgamma[l]
                    self.beta[l]  -= self.lr * dbeta[l]
                elif self.optimizer == 'momentum':
                    if not hasattr(self, 'vgamma'):
                        self.vgamma = [np.zeros_like(g) for g in self.gamma if g is not None]
                        self.vbeta  = [np.zeros_like(be) for be in self.beta if be is not None]
                    idx = l
                    self.vgamma[idx] = self.momentum * self.vgamma[idx] + self.lr * dgamma[l]
                    self.vbeta[idx]  = self.momentum * self.vbeta[idx]  + self.lr * dbeta[l]
                    self.gamma[l] -= self.vgamma[idx]
                    self.beta[l]  -= self.vbeta[idx]
                else:
                    self.gamma[l] -= self.lr * dgamma[l]
                    self.beta[l]  -= self.lr * dbeta[l]

    def _categorical_crossentropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]

    def fit(self, X, y, X_val=None, y_val=None, epochs=50, batch_size=64, patience=10, output=True):
        X = np.asarray(X)
        n_samples = X.shape[0]
        y = np.asarray(y).ravel()
    
        enc = OneHotEncoder(sparse_output=False)
        y_onehot = enc.fit_transform(y.reshape(-1, 1)).T
        self._label_encoder = enc
    
        y_val_onehot = None
        if y_val is not None:
            y_val = np.asarray(y_val).ravel()
            y_val_onehot = enc.transform(y_val.reshape(-1, 1)).T
    
        n_batches = int(np.ceil(n_samples / batch_size))
        best_val_loss = np.inf
        patience_counter = 0
        losses_train, losses_val = [], []
        acc_train, acc_val = [], []
    
        step = 0
    
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y_onehot[:, idx]
            y_shuffled_labels = y[idx]
    
            epoch_loss = 0.0
            correct_train = 0
            total_train = 0
    
            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end].T
                y_batch = y_shuffled[:, start:end]
                y_batch_labels = y_shuffled_labels[start:end]
    
                y_pred = self.forward(X_batch, training=True)
                loss = self._categorical_crossentropy(y_batch, y_pred)
                epoch_loss += loss
    
                y_pred_labels = np.argmax(y_pred, axis=0)
                correct_train += np.sum(y_pred_labels == y_batch_labels)
                total_train += len(y_batch_labels)
    
                dW, db, dgamma, dbeta = self.backward(y_batch)
                step += 1
                self.update_weights(dW, db, dgamma, dbeta, step=step)
    
            avg_train_loss = epoch_loss / n_batches
            train_acc = correct_train / total_train
            losses_train.append(avg_train_loss)
            acc_train.append(train_acc)
    
            val_loss = val_acc = None
            if X_val is not None and y_val is not None and y_val_onehot is not None:
                y_val_pred = self.forward(X_val.T, training=False)
                val_loss = self._categorical_crossentropy(y_val_onehot, y_val_pred)
                y_val_pred_labels = np.argmax(y_val_pred, axis=0)
                val_acc = np.mean(y_val_pred_labels == y_val)
    
                losses_val.append(val_loss)
                acc_val.append(val_acc)
    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if output:
                            print(f"Early stopping at epoch {epoch}, val_loss={val_loss:.6f}, val_acc={val_acc:.4f}")
                        break
                    
            if output and (epoch % 5 == 0 or epoch == 1):
                msg = f"Epoch {epoch:3d}/{epochs} | train_loss: {avg_train_loss:.6f} | train_acc: {train_acc:.4f}"
                if val_loss is not None:
                    msg += f" | val_loss: {val_loss:.6f} | val_acc: {val_acc:.4f}"
                print(msg)
    
        return {
            'train_loss': losses_train,
            'val_loss': losses_val,
            'train_acc': acc_train,
            'val_acc': acc_val
        }
    
    def predict(self, X):
        probs = self.forward(X.T, training=False)
        return np.argmax(probs, axis=0)

    def predict_proba(self, X):
        return self.forward(X.T, training=False).T
