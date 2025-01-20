from enum import Enum

import cvxpy as cp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class Model(Enum):
    QUADRATIC_SVM = 'quadratic_svm'
    SPARSE_QUADRATIC_SVM = 'sparse_quadratic_svm'
    LINEAR_KERNEL_SVM = 'linear_kernel_svm'
    QUADRATIC_KERNEL_SVM = 'quadratic_kernel_svm'
    POLYNOMIAL_KERNEL_SVM = 'polynomial_kernel_svm'
    QUADRATIC_SVM_NO_REG = 'quadratic_svm_no_reg'
    

class MinAlg(Enum):
    CVXPY = 'cvxpy'
    GD = 'gd'


class LossType(Enum):
    HINGE = 'hinge'
    QUADRATIC = 'quadratic'
    

def hinge_loss(x):
    return np.sum(np.maximum(0, x))

def quadratic_loss(x):
    return np.sum(x ** 2)


class QuadraticSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 mu=1.0, 
                 lambda_=1.0, 
                 alpha=0.0, 
                 max_iters=1000, 
                 min_alg=MinAlg.CVXPY.value, 
                 tol=1e-6, 
                 beta=0.5, 
                 verbose=False,
                 batch_size=None,
                 n_epochs=10):
        """
        Quadratic SVM with L1 and L2 regularization.
        
        Parameters
        ----------
        mu : float, default=1.0
            The regularization parameter for the hinge loss.
        lambda_ : float, default=1.0
            The regularization parameter for the L1 norm.
        alpha : float, default=0.0
            The regularization parameter for the L2 norm.
        max_iters : int, default=1000
            The maximum number of iterations for the cvxpy solver.
        min_alg : str, default='cvxpy'
            The algorithm to use for the minimization step. Supported types: 'cvxpy', 'gd'.
        tol : float, default=1e-6
            The tolerance for the gradient descent algorithm.
        beta : float, default=0.5
            The learning rate for the gradient descent algorithm.
        verbose : bool, default=False
            Whether to print additional information during training.
        batch_size : int, default=None
            The mini-batch size for the gradient descent algorithm.
        n_epochs : int, default=10
            The number of epochs for the gradient descent algorithm.
        """
        self.mu = mu
        self.lambda_ = lambda_
        self.alpha = alpha
        self.max_iters = max_iters
        self.min_alg = min_alg
        self.beta = beta
        self.tol = tol
        self.verbose = verbose
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
    @staticmethod
    def soft_threshold(x, thresh):
        """
        Component-wise soft-thresholding operator:
           S(x, thr) = sign(x) * max(|x| - thr, 0).
        """
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)
    
    def _f(self, X, W, b, c):
        """
        Compute:
          Z = X W + 1*b^T  -> shape (m, n)
          f = 0.5 * diag(X W X^T) + X b + c -> shape (m,)
        Returns (Z, f) for convenience.
        """
        # Z = (m x n)
        Z = X.dot(W) + b  # broadcasting b (shape (n,)) across rows
        # f = (m,)
        # 0.5 * sum_{j} [XW]_ij * X_ij = 0.5 * np.einsum('ij,ij->i', XW, X)
        XW = X.dot(W)  # shape (m, n)
        quad_terms = 0.5 * np.einsum('ij,ij->i', XW, X)  # shape (m,)
        linear_terms = X.dot(b)  # shape (m,)
        f = quad_terms + linear_terms + c
        return Z, f
    
    def _compute_loss(self, X, y, W, b, c):
        """
        Compute the full objective A + B + C + D.
        # (A) sum of squares: sum_i ||W x_i + b||^2
        # (B) hinge loss: mu * sum_i max(0, 1 - y_i f_i)
        # (C) L1 penalty: lambda_ * (||W||_1 + ||b||_1 + |c|)
        # (D) L2 penalty: alpha * (||W||_F^2 + ||b||_2^2 + c^2)
        returns A, B, C, D and obj_val.
        """
        Z, f = self._f(X, W, b, c)
        
        # (A) sum of squares
        sq_term = np.sum(Z ** 2)
        
        # (B) hinge loss
        rho = 1.0 - y * f  # shape (m,)
        active_mask = (rho > 0).astype(float)  # 1 if hinge is active, else 0
        hinge_val = np.sum(rho[active_mask > 0])  # sum of positive parts
        hinge_val *= self.mu
        
        # (C) L1 penalty
        l1_val = (0 if self.lambda_ == 0 else 
            self.lambda_ * (np.sum(np.abs(W)) + np.sum(np.abs(b)) + abs(c)))
        
        # (D) L2 penalty
        if self.alpha == 0:
            l2_val = 0
        else:
            l2_W = np.sum(W ** 2)
            l2_b = np.sum(b ** 2)
            l2_c = c ** 2
            l2_val = self.alpha * (l2_W + l2_b + l2_c)

        # Compute the objective value
        obj_val = sq_term + hinge_val + l1_val + l2_val
        
        loss = {
            'sq_term': sq_term,
            'hinge_val': hinge_val,
            'l1_val': l1_val,
            'l2_val': l2_val,
            'obj_val': obj_val
        }
        
        intermediary_vals = {
            'forward': {
                'Z': Z,
                'f': f,
            },
            'hing_loss': {
                'rho': rho,
                'active_mask': active_mask
            }
        }
        
        return loss, intermediary_vals
        
    def _compute_loss_and_smooth_gradients(self, X, y, W, b, c):
        """
        Compute the gradients of the gradient of the smooth part
        (everything except the L1 portion).
        """
        m = X.shape[0]
        loss, intermediary_vals = self._compute_loss(X, y, W, b, c)

        # Gradients wrt W and b for sum_i ||Z||^2 = sum_i ||W x_i + b||^2
        Z = intermediary_vals['forward']['Z']
        gradW_sq = 2.0 * X.T.dot(Z)       # shape (n,n)
        gradb_sq = 2.0 * np.sum(Z, axis=0)  # shape (n,)
        
        # Gradients wrt W and b for hinge loss term 
        # mu * sum_i active_mask_i = mu * sum_i max(0, 1 - y_i f_i)
        active_mask = intermediary_vals['hing_loss']['active_mask']
        gradW_hinge = np.zeros_like(W)
        gradb_hinge = np.zeros_like(b)
        gradc_hinge = 0.0
        
        # We can accumulate in a loop (for clarity):
        for i in range(m):
            if active_mask[i] > 0:
                # - y[i] factor
                # x_i is X[i,:], shape (n,)
                xi = X[i, :]
                yi = y[i]
                gradW_hinge += (-yi) * np.outer(xi, xi)
                gradb_hinge += (-yi) * xi
                gradc_hinge += (-yi)

        # Multiply by mu
        gradW_hinge *= self.mu
        gradb_hinge *= self.mu
        gradc_hinge *= self.mu
        
        # Gradients wrt W and b for L2 penalty alpha * (||W||_F^2 + ||b||_2^2 + c^2)
        gradW_l2 = 2.0 * self.alpha * W
        gradb_l2 = 2.0 * self.alpha * b
        gradc_l2 = 2.0 * self.alpha * c
        
        # Combine all "smooth" gradients
        gradW_smooth = gradW_sq + gradW_hinge + gradW_l2
        gradb_smooth = gradb_sq + gradb_hinge + gradb_l2
        gradc_smooth = gradc_hinge + gradc_l2
        
        gradients_smooth = {
            'gradW_smooth': gradW_smooth,
            'gradb_smooth': gradb_smooth,
            'gradc_smooth': gradc_smooth
        }
        
        return gradients_smooth, loss
    
    def _fit_with_gd(self, X, y, verbose):
        """
        Fit the model using gradient descent.
        """
        m, n = X.shape
        
        W = np.zeros((n, n))
        b = np.zeros(n)
        c = 0.0
        
        # If batch_size is None or > number of samples, treat as full batch
        if (self.batch_size is None) or (self.batch_size > m):
            effective_batch_size = m
        else:
            effective_batch_size = self.batch_size
        
        old_obj = np.inf
        
        for epoch in range(self.n_epochs):
            # Shuffle the data indices
            indices = np.arange(m)
            np.random.shuffle(indices)
            
            # Go through the mini-batches
            for start_idx in range(0, m, effective_batch_size):
                end_idx = start_idx + effective_batch_size
                batch_idx = indices[start_idx:end_idx]
                
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # 1) Compute objective and smooth gradients for the batch
                smooth_gradients, current_loss = self._compute_loss_and_smooth_gradients(X_batch, y_batch, W, b, c)
                
                gradW = smooth_gradients['gradW_smooth']
                gradb = smooth_gradients['gradb_smooth']
                gradc = smooth_gradients['gradc_smooth']
                
                # 2) Backtracking line search
                t = 1.0
                while True:
                    W_tilde = W - t * gradW
                    b_tilde = b - t * gradb
                    c_tilde = c - t * gradc
                    
                    thr = self.lambda_ * t
                    W_new = self.soft_threshold(W_tilde, thr)
                    b_new = self.soft_threshold(b_tilde, thr)
                    
                    # Scalar c proximal step
                    c_sign = np.sign(c_tilde)
                    c_abs = abs(c_tilde)
                    c_shrunk = max(c_abs - thr, 0.0)
                    c_new = c_sign * c_shrunk
                    
                    # Enforce symmetry
                    W_new = 0.5 * (W_new + W_new.T)
                    
                    # Compute objective with updated parameters
                    _, full_loss = self._compute_loss(X_batch, y_batch, W_new, b_new, c_new)
                    new_obj = full_loss['obj_val']
                    
                    if new_obj <= current_loss - 0.5 * t * (np.linalg.norm(gradW)**2 + np.linalg.norm(gradb)**2 + gradc**2):
                        break
                    t *= self.beta
                
                # 3) Update parameters
                W, b, c = W_new, b_new, c_new
                
            # After finishing one epoch, we can check convergence on the full dataset
            full_loss, _ = self._compute_loss(X, y, W, b, c)
            new_obj = full_loss['obj_val']
            
            if verbose:
                print(f"Epoch {epoch+1}/{self.n_epochs}, obj={new_obj:.6f}")
                
            # Convergence check
            rel_change = abs(old_obj - new_obj) / max(1.0, abs(old_obj))
            if rel_change < self.tol:
                if verbose:
                    print("Converged early!")
                break  
        
            old_obj = new_obj
        
        # Save final parameters
        self.W_ = W
        self.b_ = b
        self.c_ = c

    def _fit_with_cvxpy(self, X, y, verbose):
        m, n = X.shape 
        W = cp.Variable((n, n), symmetric=True)
        b = cp.Variable((n, 1))
        c = cp.Variable()

        # First term: sum_{i=1}^m ||W x_i + b||_2^2
        WX = X @ W  # Shape: (m, n)
        b_expanded = cp.matmul(cp.Constant(np.ones((m, 1))), b.T)  # Shape: (m, n)
        WX_plus_b = WX + b_expanded  # Shape: (m, n)
        norms_squared = cp.square(cp.norm(WX_plus_b, axis=1))
        first_term = cp.sum(norms_squared)

        # Second term: mu * sum_{i=1}^m max(0, 1 - y_i * f_{W,b,c}(x_i))
        quad_terms = cp.sum(cp.multiply(X @ W, X), axis=1)  # Shape: (m,)
        Xb = X @ b  # Shape: (m, 1)
        quad_terms = cp.reshape(quad_terms, (m, 1))  # Shape: (m, 1)
        f_vals = 0.5 * quad_terms + Xb + c  # Shape: (m, 1)

        y = y.reshape((m, 1))  # Shape: (m, 1)
        hinge_losses = cp.pos(1 - cp.multiply(y, f_vals))  # Shape: (m, 1)
        second_term = self.mu * cp.sum(hinge_losses)

        # Third term: lambda_ * (||W||_1 + ||b||_1 + |c|) + alpha * (||W||_2^2 + ||b||_2^2 + c^2)
        third_term = (self.lambda_ * (cp.norm1(W) + cp.norm1(b) + cp.abs(c)) +
                      self.alpha   * (cp.norm(W, 'fro') ** 2 + cp.norm2(b) ** 2 + c ** 2))

        # Objective function
        objective = cp.Minimize(first_term + second_term + third_term)

        # Problem definition
        prob = cp.Problem(objective)

        # Solve the problem
        prob.solve(solver=cp.SCS, max_iters=self.max_iters, verbose=verbose)

        # Check if the problem was solved successfully
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError("Solver did not converge!")

        # Save the parameters
        self.W_ = W.value
        self.b_ = b.value.flatten()
        self.c_ = c.value
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = y.astype(float)
        self.classes_ = np.unique(y)
        
        # If there's only 2 classes (and they are -1, +1), proceed as usual
        if len(self.classes_) == 2:
            # Just check they are indeed {-1, +1}
            unique_set = set(self.classes_)
            if unique_set != {-1.0, 1.0}:
                raise ValueError("For 2-class scenario, labels must be -1 or +1.")
            
            if self.min_alg == 'cvxpy':
                self._fit_with_cvxpy(X, y, self.verbose)
            elif self.min_alg == 'gd':
                self._fit_with_gd(X, y, self.verbose)
            else:
                raise ValueError("Invalid min_alg parameter. Must be 'cvxpy' or 'gd'.")

            # We do not maintain multiple classifiers
            self.ovr_classifiers_ = None

        # One-vs-All approach
        else:
            self.ovr_classifiers_ = []  # store sub-classifiers
            # We won't store single W_, b_, c_ on the parent
            self.W_ = None
            self.b_ = None
            self.c_ = None

            # For each class c, train a separate binary classifier
            for c in self.classes_:
                # Re-label: +1 if y_i == c, else -1
                y_bin = np.where(y == c, 1.0, -1.0)
                
                # Create a single "binary" Quadratic SVM
                #   (with same hyperparams except we override verbose or not)
                clf = QuadraticSvmClassifier(
                    mu=self.mu,
                    lambda_=self.lambda_,
                    alpha=self.alpha,
                    max_iters=self.max_iters,
                    min_alg=self.min_alg,
                    tol=self.tol,
                    beta=self.beta,
                    verbose=self.verbose,
                    batch_size=self.batch_size,
                    n_epochs=self.n_epochs
                )
                # Fit the sub-classifier
                clf.fit(X, y_bin)
                self.ovr_classifiers_.append(clf)

        return self

    def decision_function(self, X):
        """
        - If 2-class: return the (m,) decision function from the single model.
        - If multi-class: return an (m, n_classes) array of decision-values,
          where each column is the sub-classifier's decision function.
        """
        check_is_fitted(self, ['classes_'])
        X = np.asarray(X, dtype=float)

        # 2-class scenario
        if (len(self.classes_) == 2) and (self.ovr_classifiers_ is None):
            check_is_fitted(self, ['W_', 'b_', 'c_'])
            return self._f(X, self.W_, self.b_, self.c_)[1]

        # Multi-class scenario
        else:
            # We'll produce a (m, n_classes) matrix of decision values
            df_list = []
            for clf in self.ovr_classifiers_:
                # each sub-classifier's decision_function is shape (m,)
                df_list.append(clf.decision_function(X))

            # shape (n_classes, m) => transpose => (m, n_classes)
            return np.vstack(df_list).T

    def predict(self, X):
        """
        - If 2-class: sign of decision_function -> +1 or -1
        - If multi-class: pick the class with the largest decision_function
        """
        check_is_fitted(self, ['classes_'])
        
        # 2-class scenario
        if (len(self.classes_) == 2) and (self.ovr_classifiers_ is None):
            scores = self.decision_function(X)  # shape (m,)
            return np.where(scores >= 0, 1, -1)
        
        # Multi-class scenario
        else:
            # shape (m, n_classes)
            df_matrix = self.decision_function(X)
            # Pick the column with max decision
            # i.e. df_matrix[i, :] => pick argmax
            best_indices = np.argmax(df_matrix, axis=1)
            return self.classes_[best_indices]
    
    
class SparseQuadraticSvmClassifier(BaseEstimator, ClassifierMixin):
    """
    Sparse Quadratic Support Vector Machine with Zero-Norm Regularization.

    This classifier solves a non-convex optimization problem by penalty decomposition and solving
    convex sub-problems to minimize a loss function that includes a hinge loss term, a quadratic term, 
    and a zero-norm regularization term to promote sparsity.

    Parameters
    ----------
    rho : float, default=1.0
        Initial value for the penalty parameter in the augmented Lagrangian.
    beta : float, default=sqrt(5)
        Multiplicative factor to increase rho after each outer iteration.
    mu : float, default=5
        Regularization parameter for the hinge loss.
    k : int, default=10
        Sparsity level (number of non-zero elements in the solution).
    epsilon_inner : float, default=5e-2
        Tolerance for the inner loop convergence.
    epsilon_outer : float, default=5e-3
        Tolerance for the outer loop convergence.
    max_iters : int, default=1000
        Maximum number of iterations in the cvxpy solver.
    max_outer_iters : int, default=100
        Maximum number of outer iterations.
    max_inner_iters : int, default=100
        Maximum number of inner iterations.
    z_init_coef : int, default=-9
        Coefficient for initializing the z variable.
    c_init_coef : int, default=3234
        Coefficient for initializing the c variable.
    u_init_coef : int, default=-7
        Coefficient for initializing the u variable.
    solver : cvxpy solver, default=cvxpy.SCS
        The solver to use for the cvxpy problems.
    use_dual : bool, default=False
        Whether to use the dual formulation for the cvxpy problems.
    loss_type : str, default='hinge'
        The type of loss function to use. Supported types: 'hinge', 'quadratic'.
    min_alg : str, default='cvxpy'
        The algorithm to use for the minimization step. Supported types: 'cvxpy', 'gd'.
    gd_tol : float, default=1e-6
        Tolerance for the gradient descent algorithm.
    gd_n_epochs : int, default=100
        Number of epochs for the gradient descent algorithm.
    gd_beta : float, default=0.5
        Coefficient for linear backtracking in the gradient descent algorithm.
    verbose : bool, default=False
        Whether to print additional information during training.
    batch_size : int, default=None
        The mini-batch size for the gradient descent algorithm.
    n_epochs : int, default=10
        Number of epochs for the gradient descent algorithm. 


    Notes
    -----
    m = number of samples, n = number of features, d = n + n * (n + 1) // 2


    Attributes
    ----------
    X_ : Matrix of shape (m, n)
        The input training data.
    y_ : array-like of shape (m,)
        The target training labels.
    classes_ : array-like of shape (n_classes,)
        The unique classes in the target labels.
    z : Column vector of shape (d, 1)
        The half-vectorized W_ left concatenated with b_.
    u : Column vector of shape (d, 1)
        Auxiliary variable to make the problem convex. u, should be sparse.
    W_ : Matrix of shape (n, n)
        Weight matrix learned during training.
    b_ : Column vector of shape (n_features, 1)
        Bias vector learned during training.
    c_ : float
        Scalar bias term learned during training.

        
    Variables
    ---------
    q: float
        Objective value of the problem, denoted by q in the paper.
    G: array-like of shape (d, d) where d = n_features + n_features * (n_features + 1) // 2
        The matrix G in the paper.
    r: array-like of shape (n_samples, d) where d = n_features + n_features * (n_features + 1) // 2
        The variable r in the paper.
    """

    def __init__(self, 
                 rho=1.0, 
                 beta=np.sqrt(5), 
                 mu=5, 
                 k=10,
                 epsilon_inner=5e-2, 
                 epsilon_outer=5e-3, 
                 max_iters=1000,
                 max_outer_iters=10,
                 max_inner_iters=100,
                 z_init_coef=-9,
                 c_init_coef=3234,
                 u_init_coef=-7,
                 solver=cp.SCS, # For CVXPY
                 use_dual=False, # For CVXPY
                 loss_type=LossType.HINGE.value, # 'hinge' or 'quadratic'
                 min_alg=MinAlg.CVXPY.value, # 'cvxpy' or 'gd'
                 gd_tol=1e-8, # For GD
                 gd_n_epochs=1000, # For GD
                 gd_beta=0.5, # For GD
                 verbose=False,
                 batch_size=None, # For GD
                 ):
        self.rho = rho
        self.beta = beta
        self.mu = mu
        self.k = k
        self.epsilon_inner = epsilon_inner
        self.epsilon_outer = epsilon_outer
        self.max_iters = max_iters
        self.max_outer_iters = max_outer_iters
        self.max_inner_iters = max_inner_iters
        self.z_init_coef = z_init_coef
        self.c_init_coef = c_init_coef
        self.u_init_coef = u_init_coef
        self.solver = solver
        self.use_dual = use_dual
        self.loss_type = loss_type
        self.min_alg = min_alg
        self.gd_tol = gd_tol
        self.gd_n_epochs = gd_n_epochs
        self.gd_beta = gd_beta
        self.batch_size = batch_size
        self.verbose = verbose

        if loss_type == 'hinge':
            self.H = hinge_loss
        elif loss_type == 'quadratic':
            self.H = quadratic_loss
        else:
            raise ValueError("Invalid loss type. Supported types: 'hinge', 'quadratic'.")

    def _get_d(self, n):
        """Compute the dimensionality of the variable z."""
        return n + n * (n + 1) // 2
    
    def _get_duplication_matrix(self, n):
        """Create the duplication matrix D_n in a vectorized manner."""
        # Total number of elements in the upper triangle including the diagonal
        k = self._get_d(n) - n

        # Generate indices for the upper triangle
        i_upper, j_upper = np.triu_indices(n)

        # Create a mapping from upper triangle indices to unique IDs
        z = np.arange(k)

        # Initialize the j_indices matrix
        j_indices = np.zeros((n, n), dtype=int)
        j_indices[i_upper, j_upper] = z
        j_indices[j_upper, i_upper] = z  # Ensure symmetry

        # Flatten the j_indices matrix
        j_indices_flat = j_indices.flatten()

        # Create the duplication matrix using COO format
        i_indices = np.arange(n * n)
        data = np.ones(n * n, dtype=int)
        D_n = coo_matrix((data, (i_indices, j_indices_flat)),
                        shape=(n * n, k)).toarray()

        return D_n

    def _compute_G_and_r(self, X):
        """
        Compute the matrices G and r based on the data matrix X_ in a vectorized manner.
        """
        m, n = X.shape
        d = self._get_d(n)

        D_n = self._get_duplication_matrix(n)

        # Compute the half-vectorization of x_i x_i^T for all i
        X_outer = X[:, :, np.newaxis] * X[:, np.newaxis, :]  # Shape: (m, n, n)
        i_upper, j_upper = np.triu_indices(n)
        s = 0.5 * X_outer[:, i_upper, j_upper]  # Shape: (m, k)

        # Compute r in a vectorized manner
        r = np.hstack([s, X])  # Shape: (m, d)

        # Initialize G
        G = np.zeros((d, d))

        # Precompute identity matrix
        eye_n = np.eye(n)

        # Loop over m to compute H_i and accumulate G
        for i in range(m):
            x_i = X[i, :]

            # Compute X_i as Kronecker product
            X_i = np.kron(eye_n, x_i.reshape(1, -1))  # Shape: (n, n^2)

            # Compute M_i
            M_i = X_i @ D_n  # Shape: (n, k)

            # Construct H_i
            H_i = np.hstack([M_i, eye_n])  # Shape: (n, d)

            # Accumulate G
            G += 2 * H_i.T @ H_i  # Shape: (d, d)

        return G, r

    def _hvec(self, A):
        """
        Half-vectorization of a symmetric matrix A (taking upper triangular part including the diagonal).
        
        Parameters
        ----------
        A : ndarray of shape (n, n)
            The symmetric matrix to half-vectorize.
        
        Returns
        -------
        v : ndarray of shape (n * (n + 1) // 2,)
            The half-vectorized form of the symmetric matrix A.
        """
        n = A.shape[0]
        indices = np.triu_indices(n)
        return A[indices]
    
    def _hvec_inv(self, v):
        """
        Reconstructs the symmetric matrix A from its half-vectorized form v.

        Parameters
        ----------
        v : array-like of shape (n * (n + 1) // 2,)
            The half-vectorized form of the symmetric matrix A.

        Returns
        -------
        A : ndarray of shape (n, n)
            The reconstructed symmetric matrix.
        """
        vec_d = v.shape[0]
        n = 0.5 * (-1 + np.sqrt(1 + 8 * vec_d))
        n = int(n)
        A = np.zeros((n, n))
        indices = np.triu_indices(n)
        v = v.flatten()
        A[indices] = v
        v = v.reshape(-1, 1)
        A = A + A.T - np.diag(np.diag(A))
        
        return A

    def _q(self, G, r, y, z, c, u):
        """
        Compute 
        q_rho(z,c,u):= 0.5 * z^TGz + mu * sum_i H(1-y_i(z^Tr_i+c)) + 0.5 * rho * ||z-u||_2^2
        
        Parameters
        ----------
        Note: m is the number of samples, n is the number of features, and d = n + n * (n + 1) // 2.
        G : Matrix of shape (d, d)
            The matrix G in the objective function.
        r : Matrix of shape (m, d)
            The matrix r in the objective function.
        y : Column vector of shape (m,)
            The target labels.
        z : Column vector of shape (d, 1)
            The variable z in the objective function.
        c : float
            The variable c in the objective function.
        u : Column vector of shape (d, 1)
            The auxiliary variable u in the objective function.
        
        Returns
        -------
        q : float
        """
        square_loss = 0.5 * z.T @ G @ z
        regularization = 0.5 * self.rho * norm(z - u, 2)**2
        y_hat = r @ z + c

        q = square_loss + self.mu * self.H(1 - y.reshape(-1, 1)*y_hat) + regularization
        
        return q[0][0]

    def _optimize_alpha_dual(self, G, r, u):
        """
        Optimize over alpha (dual variables) while keeping u fixed.
        
        Parameters
        ----------
        G : Matrix of shape (d, d)
            The matrix G in the objective function.
        r : Matrix of shape (m, d)
            The matrix r in the objective function.
        u : Column vector of shape (d, 1)
            The auxiliary variable u in the objective function.
            
        Returns
        -------
        alpha : Column vector of shape (m, 1)
        """
        m, d = r.shape  # m samples, d features
        alpha = cp.Variable(m)

        M = G + self.rho * np.eye(d)  # (d x d) matrix
        M_inv = np.linalg.inv(M)  # (d x d) matrix

        # y_flat = self.y_.copy().flatten()  # (m,) vector
        u = u.reshape(d, 1)  # (d x 1) vector

        # Precompute constants for the quadratic term
        Dy = np.diag(self.y_)
        K = Dy.T @ r @ M_inv @ r.T @ Dy  # (m x m) matrix
        c = (self.rho * u.T @ M_inv @ r.T @ Dy).flatten() - np.ones(m)

        # Define the objective function
        objective = cp.Minimize(0.5 * cp.quad_form(alpha, K) + c @ alpha)
        
        constraints = [
            cp.sum(cp.multiply(self.y_, alpha)) == 0,
            alpha >= 0,
            alpha <= self.mu
        ]

        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.solver, verbose=False)

        return alpha.value.reshape(-1, 1)
    
    def _fit_z_c_with_dual(self, u):
        """
        Compute z and c from alpha using KKT conditions.
        
        Parameters
        ----------
        u : Column vector of shape (d, 1)
            The auxiliary variable u in the objective function.
            
        Returns
        -------
        z : Column vector of shape (d, 1)
        c : float
        """
        if not self.use_dual:
            raise ValueError("This method is only applicable for the dual formulation. To use it\
                             set use_dual=True.")
        G, r = self._compute_G_and_r(self.X_)  # (d, d), (m, d)
        
        # Step 1: Compute z
        M = G + self.rho * np.eye(G.shape[0])
        # Compute the right-hand side of the equation
        alpha = self._optimize_alpha_dual(G, r, u)
        rhs = self.rho * u + r.T @ (alpha * self.y_.reshape(-1, 1))  # (d x 1) vector
        # Solve for z
        z = np.linalg.solve(M, rhs)  # (d x 1) vector

        # Step 2: Compute c
        tolerance = 1e-5  # Tolerance for numerical stability

        # Identify support indices
        support_mask = (alpha > tolerance) & (alpha < self.mu - tolerance)
        support_indices = np.where(support_mask)[0]

        if len(support_indices) == 0:
            # No support vectors found; default c to zero or handle appropriately
            c = 0.0
        else:
            # Compute c for each support vector
            r_z = r[support_indices] @ z  # Shape: (k, 1)
            c_values = self.y_.reshape(-1, 1)[support_indices] - r_z  # Shape: (k, 1)
            c = np.mean(c_values)

        return z, c # (d x 1) vector, scalar
    
    def _fit_z_c_with_cvxpy(self, u):
        '''
        Compute z and c using cvxpy. Only works with Hinge loss. Quadratic loss is not supported.
        
        Parameters
        ----------
        u : Column vector of shape (d, 1)
            The auxiliary variable u in the objective function.
            
        Returns
        -------
        z_value : Column vector of shape (d, 1)
        c_value : float
        '''
        if self.loss_type != 'hinge':
            raise ValueError("CVXPY solver only supports hinge loss. \
                             Use _fit_z_c_with_quadratic() for quadratic loss.")
            
        if self.use_dual:
            raise ValueError("CVXPY solver does not support dual formulation. \
                             Use _fit_z_c_from_alpha for dual formulation.")
            
        G, r = self._compute_G_and_r(self.X_)  # (d, d), (m, d)
        
        # Ensure correct dimensions for variables
        z = cp.Variable(u.shape)  # (d, 1)
        c = cp.Variable()         # Scalar

        # Objective components
        square_loss = 0.5 * cp.quad_form(z, G)  # Quadratic term for z
        hinge_loss = cp.sum(
            cp.pos(1 - cp.multiply(self.y_.reshape(-1, 1), r@z + c))  # Hinge loss for each sample
        )
        regularization = 0.5 * cp.norm(z - u, 2)**2  # Regularization term

        # Define the objective function
        objective = cp.Minimize(square_loss + self.mu*hinge_loss + self.rho*regularization)

        # No additional constraints in this formulation
        constraints = []

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.solver, verbose=self.verbose, max_iters=self.max_iters)

        # Extract values, ensuring dimensions are preserved
        z_value = z.value.reshape(-1, 1)  # Ensure z is returned as (d, 1)
        c_value = float(c.value)          # Ensure c is a scalar

        return z_value, c_value # (d, 1) vector, scalar
    
    def _fit_z_c_with_gd(self, u):
        """
        Compute z and c using gradient descent.
        
        Parameters
        ----------
        u : Column vector of shape (d, 1)
            The auxiliary variable u in the objective function.
        
        Returns
        -------
        z : Column vector of shape (d, 1)
        c : float
        """
        if self.loss_type != 'hinge':
            raise ValueError("CVXPY solver only supports hinge loss. \
                             Use _fit_z_c_with_quadratic() for quadratic loss.")
            
        m, n = self.X_.shape
        d = self._get_d(n)
        z = np.zeros((d, 1))
        c = 0.0
        
        G, r = self._compute_G_and_r(self.X_) # (d, d), (m, d)
        
        # If batch_size is None or > number of samples, treat as full batch
        if (self.batch_size is None) or (self.batch_size > m):
            effective_batch_size = m
        else:
            effective_batch_size = self.batch_size
        
        prev_q = np.inf
        
        for epoch in range(self.gd_n_epochs):
            # Shuffle the data indices
            indices = np.arange(m)
            np.random.shuffle(indices)
            
            # Go through the mini-batches
            for start_idx in range(0, m, effective_batch_size):
                end_idx = start_idx + effective_batch_size
                batch_idx = indices[start_idx:end_idx]
                
                X_batch = self.X_[batch_idx] # (effective_batch_size, n)
                y_batch = self.y_[batch_idx].reshape(-1, 1) # (effective_batch_size, 1)
                
                G_batch, r_batch = self._compute_G_and_r(X_batch) # (d, d), (effective_batch_size, d)
                
                # ---------- 1) Compute gradients ----------

                # (A) Gradient wrt z of 0.5 * z^T G z is G z
                grad_z_sq = G_batch @ z # shape (d, 1)

                # (B) Gradient wrt z, c for hinge:
                #     hinge = mu * sum_i max(0, 1 - y_i(r_i^T z + c))
                # Only samples i with margin[i] = 1 - y_i * y_hat[i] > 0 are active
                y_hat = r_batch@z + c
                margin = 1.0 - y_batch * y_hat  # shape (effective_batch_size, 1)
                active_mask = (margin > 0).astype(float) 

                # grad wrt z is: - mu * sum_{i in active} y_i * r_i
                # note r_i is row i of self.r (shape (n,))
                # we want a column vector for z
                grad_z_hinge = np.zeros((d, 1))
                for i in range(effective_batch_size):
                    if active_mask[i] > 0:
                        # derivative is -mu * y_i * r_i
                        grad_z_hinge += (-self.mu * y_batch[i]) * r_batch[i, :].reshape(d, 1)

                # grad wrt c is: - mu * sum_{i in active} y_i
                grad_c_hinge = 0.0
                for i in range(effective_batch_size):
                    if active_mask[i] > 0:
                        grad_c_hinge += (-self.mu * y_batch[i])

                # (C) Gradient wrt z for reg: 0.5*rho||z-u||^2 => grad is rho(z-u)/2 * 2 => rho(z-u)
                grad_z_reg = self.rho * (z - u)

                # ---------- 2) Total gradients ----------
                grad_z = grad_z_sq + grad_z_hinge + grad_z_reg
                grad_c = grad_c_hinge

                # ---------- 3) Backtracking line search ----------
                t = 1.0
                while True:
                    z_new = z - t * grad_z
                    c_new = c - t * grad_c
                    current_q = self._q(G_batch, r_batch, y_batch, z_new, c_new, u)
                    if current_q <= prev_q - 0.5 * t * (np.linalg.norm(grad_z)**2 + grad_c**2):
                        break
                    t *= self.gd_beta
                # ---------- 4) Update parameters ----------
                z, c = z_new, c_new
                
            # ------------- 5) check convergence on the full dataset -------------
            current_q = self._q(G, r, self.y_, z, c, u)
            
            # if self.verbose:
            #     print(f"Epoch {epoch+1}/{self.gd_n_epochs}: {current_q:.6f}")
                
            # Convergence check
            if prev_q not in [np.inf]:
                rel_change = abs(prev_q - current_q) / max(1.0, abs(prev_q))
                if rel_change < self.gd_tol:
                    if self.verbose:
                        print("Converged early!")
                    break
            
            prev_q = current_q
            
        return z, c[0] # (d, 1) vector, scalar
        
    def _fit_z_c_with_quadratic(self, u):
        """
        Solve the subproblem:

        minimize_{z in R^d, c in R}
            0.5 * z^T G z
            + mu * sum_i [1 - y_i(r_i^T z + c)]^2
            + 0.5 * rho * ||z - u||^2

        using the closed-form block matrix solution:

        [ z ] = [ M11   M12 ]^-1  *  [ b1 ]
        [ c ]   [ M21   M22 ]      [ b2 ]

        with:
        M11 = G + rho I + 2 mu A^T (D^2) A
        M12 = 2 mu A^T D 1
        M21 = 2 mu 1^T (D^2) A
        M22 = 2 mu 1^T (D^2) 1
        b1  = 2 mu A^T D 1 + rho u
        b2  = 2 mu 1^T D 1

        Here y in {+1, -1}, so D^2 = I.

        Parameters
        ----------
        u   : ndarray of shape (d, 1)

        Returns
        -------
        z : ndarray of shape (d, 1)   (the optimal z)
        c : float                  (the optimal c)
        """
        if self.loss_type != 'quadratic':
            raise ValueError("This method is only applicable for the quadratic loss. \
                             Use _fit_z_c_with_cvxpy() for hinge loss.")
        
        if self.use_dual:
            raise ValueError("This method is only applicable for the primal formulation. \
                             Use _fit_z_c_from_alpha for dual formulation.")
        
        G, r = self._compute_G_and_r(self.X_)
        
        # Dimensions
        m, d = r.shape

        # Ensure shapes
        u = u.reshape(-1)  # Ensure u is (d,)
        y = self.y_.reshape(-1)  # Ensure y is (m,)
        one = np.ones(m)  # Vector of ones, shape (m,)

        # (1) Construct block matrix M
        # -- M11 = G + rho I + 2 mu A^T A
        M11 = G + self.rho * np.eye(d) + 2.0 * self.mu * (r.T @ r)  # Shape (d, d)

        # -- M12 = 2 mu A^T y
        M12 = 2.0 * self.mu * (r.T @ y)  # Shape (d,)

        # -- M21 = 2 mu 1^T A
        M21 = 2.0 * self.mu * (one @ r)  # Shape (d,)

        # -- M22 = 2 mu * m
        M22 = 2.0 * self.mu * m  # Scalar

        # Make the big (d+1)x(d+1) matrix
        M12_mat = M12[:, np.newaxis]          # Shape (d, 1)
        M21_mat = M21[np.newaxis, :]          # Shape (1, d)
        big_M = np.block([
            [M11,       M12_mat],
            [M21_mat,   M22      ]
        ])  # Shape (d+1, d+1)

        # (2) Construct right-hand side b = [b1; b2]
        # b1 = 2 mu A^T y + rho u
        b1 = 2.0 * self.mu * (r.T @ y) + self.rho * u  # Shape (d,)

        # b2 = 2 mu sum(y)
        b2 = 2.0 * self.mu * np.sum(y)  # Scalar

        # Make big b shape (d+1,)
        big_b = np.concatenate([b1, [b2]], axis=0)  # Shape (d+1,)

        # (3) Solve the linear system big_M x = big_b
        sol = np.linalg.solve(big_M, big_b)  # Shape (d+1,)

        # The first d entries => z, the last => c
        z = sol[:d].reshape(-1, 1)  # Ensure z is (d, 1)
        c = float(sol[d])  # Ensure c is scalar

        return z, c
    
    def _optimize_z_c(self, u):
        """
        Optimize over z and c while keeping u fixed.
        
        Parameters
        ----------
        u : Column vector of shape (d, 1)
            The auxiliary variable u in the objective function.
            
        Returns
        -------
        z : Column vector of shape (d, 1)
        c : float
        """
        if self.use_dual:
            return self._fit_z_c_with_dual(u)
        if self.loss_type == 'hinge':
            if self.min_alg == 'cvxpy':
                return self._fit_z_c_with_cvxpy(u)
            elif self.min_alg == 'gd':
                return self._fit_z_c_with_gd(u)
        elif self.loss_type == 'quadratic':
            return self._fit_z_c_with_quadratic(u)
        
    def _optimize_u(self, z):
        """
        Optimize over u to promote sparsity, keeping z fixed.
        
        Parameters
        ----------
        z : Column vector of shape (d, 1)
            The variable z in the objective function.
            
        Returns
        -------
        u : Column vector of shape (d, 1)
        """
        u = np.zeros(z.shape)
        idx = np.argpartition(np.abs(z), -self.k, axis=0)[-self.k:]
        u[idx] = z[idx]
        return u
    
    def _inner_loop_converged(self, z_lplus1, c_lplus1, u_lplus1):
        """
        Check if the inner loop has converged.
        
        Parameters
        ----------
        z_lplus1 : Column vector of shape (d, 1)
            The updated z variable.
        c_lplus1 : float
            The updated c variable.
        u_lplus1 : Column vector of shape (d, 1)
            The updated u variable.
            
        Returns
        -------
        converged : bool
        """
        term_z = norm(z_lplus1 - self.z, np.inf) / max(1, norm(z_lplus1, np.inf))
        term_c = abs(c_lplus1 - self.c) / max(1, abs(c_lplus1))
        term_u = norm(u_lplus1 - self.u, np.inf) / max(1, norm(u_lplus1, np.inf))
        max_term = max(term_z, term_c, term_u)
        
        if self.verbose: print("Max term: ", max_term)
        
        return max_term <= self.epsilon_inner
    
    def _run_inner_loop(self, X, y, X_test=None, y_test=None):
        """
        Run the inner loop of the optimization algorithm.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        X_test : array-like of shape (n_samples, n_features)
            Test vectors.
        y_test : array-like of shape (n_samples,)
            Test values.
        """
        z_plus1, c_plus1 = self._optimize_z_c(self.u)
        u_plus1 = self._optimize_u(z_plus1)
        
        if self.verbose: print("⚙️ Train accuracy: ", self.score(X, y))
        if X_test is not None and y_test is not None:
            if self.verbose: print("🧪 Test accuracy: ", self.score(X_test, y_test))
            
        converged = self._inner_loop_converged(z_plus1, c_plus1, u_plus1)
            
        self.z, self.c, self.u = z_plus1, c_plus1, u_plus1
        
        return converged
    
    def _extract_w_b(self):
        """
        Extract the weight matrix W and bias vector b from the variables z and c.
        
        Returns
        -------
        W : ndarray of shape (n_features, n_features)
        b : ndarray of shape (n_features,)
        """
        n = self.X_.shape[1]
        d = self._get_d(n)
        self.W_ = self._hvec_inv(self.u[:d - n])
        self.b_ = self.u[d - n:].flatten()
        
    def fit(self, X, y, X_test=None, y_test=None):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples, 1)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        # One-vs-All approach for multi-class classification
        if len(self.classes_) > 2:
            print("------------Training multi-class classifier using One-vs-All approach-----------------")
            self.ovr_classifiers_ = []  # Store sub-classifiers
            self.W_ = None
            self.b_ = None
            self.c_ = None

            # Train a binary classifier for each class
            for c in self.classes_:
                print(f"------------Training binary classifier for class----------------- {c}")
                # Re-label: +1 for the current class, -1 for all others
                y_bin = np.where(y == c, 1.0, -1.0)

                # Create a new classifier instance
                clf = self.__class__(
                    rho=self.rho,
                    beta=self.beta,
                    mu=self.mu,
                    k=self.k,
                    epsilon_inner=self.epsilon_inner,
                    epsilon_outer=self.epsilon_outer,
                    max_iters=self.max_iters,
                    max_outer_iters=self.max_outer_iters,
                    max_inner_iters=self.max_inner_iters,
                    z_init_coef=self.z_init_coef,
                    c_init_coef=self.c_init_coef,
                    u_init_coef=self.u_init_coef,
                    solver=self.solver,
                    use_dual=self.use_dual,
                    loss_type=self.loss_type,
                    min_alg=self.min_alg,
                    gd_tol=self.gd_tol,
                    gd_n_epochs=self.gd_n_epochs,
                    gd_beta=self.gd_beta,
                    verbose=self.verbose,
                    batch_size=self.batch_size
                )

                # Fit the binary classifier
                clf.fit(X, y_bin)
                self.ovr_classifiers_.append(clf)

            return self
        
        # Binary classification case
        if set(self.classes_) != {-1, 1}:
            raise ValueError("Labels must be -1 or +1 for binary classification.")

        self.X_ = X
        self.y_ = y.reshape(-1, 1)

        n = X.shape[1]
        d = self._get_d(n)
        
        self.z = self.z_init_coef * np.random.rand(d, 1)
        self.c = self.c_init_coef * np.random.random()
        self.u = self.u_init_coef * np.random.rand(d, 1)

        # Outer loop
        for j in range(self.max_outer_iters):
            # Inner loop
            for l in range(self.max_inner_iters):
                converged = self._run_inner_loop(X, y, X_test, y_test)
                if converged:
                    break

            self.rho *= self.beta

            self._extract_w_b()

            if norm(self.z - self.u, np.inf) <= self.epsilon_outer:
                break

        return self            

    def decision_function(self, X):
        """
        Compute the decision function for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        decision : array-like of shape (n_samples,) or (n_samples, n_classes)
            The decision function of the input samples.
        """
        check_is_fitted(self, ['classes_'])
        X = check_array(X)

        if len(self.classes_) > 2:
            # Multi-class case: combine decisions from OvA classifiers
            decisions = np.zeros((X.shape[0], len(self.classes_)))
            for i, clf in enumerate(self.ovr_classifiers_):
                decisions[:, i] = clf.decision_function(X).flatten()
            return decisions

        # Binary classification case
        check_is_fitted(self, ['u', 'c'])
        m, n = X.shape
        d = self._get_d(n)
        r = np.zeros((m, d))

        for i in range(m):
            x_i = X[i, :]
            x_i_x_i_T = np.outer(x_i, x_i)
            s_i = 0.5 * self._hvec(x_i_x_i_T)
            r[i] = np.concatenate([s_i, x_i])

        decision = r @ self.u + self.c
        return decision.flatten()

    def predict(self, X):
        """
        Perform classification using decision function
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class label per sample.
        """
        check_is_fitted(self, ['classes_'])

        if len(self.classes_) > 2:
            # Multi-class case: choose class with highest decision value
            decisions = self.decision_function(X)
            class_indices = np.argmax(decisions, axis=1)
            y_pred = self.classes_[class_indices]
            return y_pred

        # Binary classification case
        scores = self.decision_function(X)
        y_pred = np.where(scores >= 0, 1, -1)
        return y_pred
