from utils.spectral_utils import tridiag_to_eigv
from tqdm import tqdm
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import eigsh
from warnings import warn
import torch
import time
import logging
from utils import sampling_utils as sample
import numpy as np

def hvp(loss, model, vec):
    """
    Returns H*vec where H is the hessian of the loss w.r.t.
    the vectorized model parameters
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.zero_()
    hessian_vec_prod = None
    
    grad_dict = torch.autograd.grad(
        loss, model.parameters(), create_graph=True
    )
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
    grad_grad = torch.autograd.grad(
        grad_vec, model.parameters(), grad_outputs=vec, only_inputs=True, retain_graph=True
    )
    if hessian_vec_prod is not None:
        hessian_vec_prod += torch.cat([g.contiguous().view(-1) for g in grad_grad])
    else:
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in grad_grad])

    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.zero_()

    return hessian_vec_prod


def lanczos(loss, model, dim, max_itr, use_cuda=False, verbose=False):
    r'''
        Lanczos iteration following the wikipedia article here
            https://en.wikipedia.org/wiki/Lanczos_algorithm
        Inputs:
            func   : model functional class
            dim    : dimensions
            max_itr: max iteration
            use_gpu: Use Gpu
            verbose: print extra details
        Outputs:
            eigven values
            weights
    '''
    float_dtype = torch.float64

    # Initializing empty arrays for storing
    tridiag = torch.zeros((max_itr, max_itr), dtype=float_dtype)
    vecs = torch.zeros((dim, max_itr), dtype=float_dtype)

    # intialize a random unit norm vector
    init_vec = torch.zeros((dim), dtype=float_dtype).uniform_(-1, 1)
    init_vec /= torch.norm(init_vec)
    vecs[:, 0] = init_vec

    # placeholders for data
    beta = 0.0
    v_old = torch.zeros((dim), dtype=float_dtype)

    for k in range(max_itr):
        t = time.time()

        v = vecs[:, k]
        if use_cuda:
            v = v.type(torch.float32).cuda()
        time_mvp = time.time()
        w = hvp(loss, model, v)
        if use_cuda:
            v = v.cpu().type(float_dtype)
            w = w.cpu().type(float_dtype)
        time_mvp = time.time() - time_mvp

        w -= beta * v_old
        alpha = np.dot(w, v)
        tridiag[k, k] = alpha
        w -= alpha*v

        # Reorthogonalization
        for j in range(k):
            tau = vecs[:, j]
            coeff = np.dot(w, tau)
            w -= coeff * tau

        beta = np.linalg.norm(w)

        if beta < 1e-6:
            raise ZeroDivisionError
            quit()

        if k + 1 < max_itr:
            tridiag[k, k+1] = beta
            tridiag[k+1, k] = beta
            vecs[:, k+1] = w / beta

        v_old = v

        info = f"Iteration {k} / {max_itr} done in {time.time()-t:.2f}s (MVP: {time_mvp:.2f}s)"
        if (verbose) and (k%10 == 0):
            logger = logging.getLogger('my_log')
            logger.info(info)

    return vecs, tridiag

def eig_trace(loss, model, max_itr, draws, use_cuda=False):
    dim  = sum(p.numel() for p in model.parameters())

    tri = np.zeros((draws, max_itr, max_itr))
    for num_draws in tqdm(range(draws)):
        _, tridiag = lanczos( loss, model, dim, max_itr, use_cuda)
        tri[num_draws, :, :] = tridiag.numpy()

    e, w = tridiag_to_eigv(tri)
    e = np.mean(e, 0)
    return e