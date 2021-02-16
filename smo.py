from __future__ import division, print_function
import os
import numpy as np

filepath = os.path.dirname(os.path.abspath(__file__))


class SMO():
    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, tol=0.001):
        self.kernels = {
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.tol = tol
        self.b = 0

    def fit(self, X, y):
        print('x shape', X.shape)
        print('y shape', y.shape)
        # Initialization
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros(n)[:, np.newaxis]
        kernel = self.kernels[self.kernel_type]
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                # Compute Ej
                Ej = self.compute_error(X, y, alpha, j)
                # check kkt violation
                if ~self.check_kkt_violation(alpha[j], y[j] * Ej):
                    continue
                i = self.select_randomly(j, n)  # Get random int i~=j
                xi, xj, yi, yj = X[i, :], X[j, :], y[i], y[j]
                eta = kernel(xi, xi) + kernel(xj, xj) - 2 * kernel(xi, xj)
                if eta == 0:
                    continue

                # save old alphas
                alpha_j_old, alpha_i_old = alpha[j], alpha[i]
                (L, H) = self.find_bounds(self.C, alpha_j_old, alpha_i_old, yj, yi)



                # Compute Ei
                Ei = self.compute_error(X, y, alpha, i)

                # Set new alpha values
                alpha[j] = alpha_j_old + float(yj * (Ei - Ej)) / eta
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_i_old + yi * yj * (alpha_j_old - alpha[j])

                # find bias
                bi = self.b - Ei - y[i] * (alpha[i] - alpha_i_old) * kernel(xi, xi) - y[j] * (
                            alpha[j] - alpha_j_old) * kernel(xi, xj)
                bj = self.b - Ej - y[i] * (alpha[i] - alpha_i_old) * kernel(xj, xi) - y[j] * (
                            alpha[j] - alpha_j_old) * kernel(xj, xj)
                if 0 < alpha[i] < self.C: self.b = bi
                elif 0 < alpha[j] < self.C: self.b = bj
                else: self.b = (bi + bj)/2.0

            # Check the magnitude of change for convergence
            print('\nFor iteration %d alpha is: \n' % count, alpha)
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.tol:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % self.max_iter)
                return
        # Compute final model parameters
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, len(support_vectors), alpha

    def check_kkt_violation(self, alpha, yE):
        is_violated = (alpha < self.C and yE < -self.tol) or (alpha > 0 and yE > self.tol)
        return is_violated

    def predict(self, X):
        return self.h(X, self.w, self.b)

    def calc_b(self, X, y, w):
        return np.mean(y - np.dot(w.T, X.T))

    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha, y))

    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    # Prediction error
    def compute_error(self, X, y, alpha, i):
        return np.sum(np.multiply(y, alpha) * X * X[i].T) + self.b - y[i]

    def find_bounds(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if (y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

    def select_randomly(self, j, n):
        i = j
        while (i == j):
            i = int(np.random.uniform(0, n))
        return i

    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)
