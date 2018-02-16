from numpy.core.numerictypes import typecodes
import numpy
from scipy import linalg


# taken from pyriemann.base
def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    if Ci.dtype.char in typecodes['AllFloat'] and not numpy.isfinite(Ci).all():
        raise ValueError("Covariance matrices must be positive definite. Add regularization to avoid this error.")
    eigvals, eigvects = linalg.eigh(Ci, check_finite=False)
    eigvals = numpy.diag(operator(eigvals))
    Out = numpy.dot(numpy.dot(eigvects, eigvals), eigvects.T)
    return Out


def sqrtm(Ci):
    """Return the matrix square root of a covariance matrix defined by :
    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{1/2} \mathbf{V}^T
    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`
    :param Ci: the coavriance matrix
    :returns: the matrix square root
    """
    return _matrix_operator(Ci, numpy.sqrt)


def logm(Ci):
    """Return the matrix logarithm of a covariance matrix defined by :
    .. math::
            \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T
    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`
    :param Ci: the coavriance matrix
    :returns: the matrix logarithm
    """
    return _matrix_operator(Ci, numpy.log)


def expm(Ci):
    """Return the matrix exponential of a covariance matrix defined by :
    .. math::
            \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T
    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`
    :param Ci: the coavriance matrix
    :returns: the matrix exponential
    """
    return _matrix_operator(Ci, numpy.exp)


def invsqrtm(Ci):
    """Return the inverse matrix square root of a covariance matrix defined by :
    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2} \mathbf{V}^T
    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`
    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root
    """
    isqrt = lambda x: 1. / numpy.sqrt(x)
    return _matrix_operator(Ci, isqrt)


# taken from pyriemann.utils.mean
def _get_sample_weight(sample_weight, data):
    """Get the sample weights.
    If none provided, weights init to 1. otherwise, weights are normalized.
    """
    if sample_weight is None:
        sample_weight = numpy.ones(data.shape[0])
    if len(sample_weight) != data.shape[0]:
        raise ValueError("len of sample_weight must be equal to len of data.")
    sample_weight /= numpy.sum(sample_weight)
    return sample_weight


def mean_riemann(covmats, tol=10e-9, maxiter=50, init=None,
                 sample_weight=None):
    """Return the mean covariance matrix according to the Riemannian metric.
    The procedure is similar to a gradient descent minimizing the sum of
    riemannian distance to the mean.
    .. math::
            \mathbf{C} = \\arg\min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}  # noqa
    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the gradient descent. If None the Arithmetic mean is used
    :param sample_weight: the weight of each sample
    :returns: the mean covariance matrix
    """
    # init
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = numpy.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    nu = 1.0
    tau = numpy.finfo(numpy.float64).max
    crit = numpy.finfo(numpy.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        J = numpy.zeros((Ne, Ne))

        for index in range(Nt):
            tmp = numpy.dot(numpy.dot(Cm12, covmats[index, :, :]), Cm12)
            J += sample_weight[index] * logm(tmp)

        crit = numpy.linalg.norm(J, ord='fro')
        h = nu * crit
        C = numpy.dot(numpy.dot(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C