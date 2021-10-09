import minitorch
import pytest
from hypothesis import given
import numba
from hypothesis.strategies import floats, integers, lists, data, permutations

from .strategies import tensors, shaped_tensors, assert_close


small_floats = floats(min_value=-100, max_value=100, allow_nan=False)
v = 4.524423
one_arg = [
    ("neg", lambda a: -a),
    ("addconstant", lambda a: a + v),
    ("lt", lambda a: a < v),
    ("subconstant", lambda a: a - v),
    ("mult", lambda a: 5 * a),
    ("div", lambda a: a / v),
    ("sig", lambda a: a.sigmoid()),
    ("log", lambda a: (a + 100000).log()),
    ("relu", lambda a: (a + 2).relu()),
    ("exp", lambda a: (a - 200).exp()),
]

reduce = [
    ("sum", lambda a: a.sum()),
    ("mean", lambda a: a.mean()),
    ("sum2", lambda a: a.sum(0)),
    ("mean2", lambda a: a.mean(0)),
]
two_arg = [
    ("add", lambda a, b: a + b),
    ("mul", lambda a, b: a * b),
    ("lt", lambda a, b: a < b + v),
]


# Create different backends.
TensorBackend = minitorch.make_tensor_backend(minitorch.TensorOps)
FastTensorBackend = minitorch.make_tensor_backend(minitorch.FastOps)
matmul_tests = [pytest.param(FastTensorBackend, marks=pytest.mark.task3_2)]
backend_tests = [pytest.param(FastTensorBackend, marks=pytest.mark.task3_1)]


if numba.cuda.is_available():
    CudaTensorBackend = minitorch.make_tensor_backend(minitorch.CudaOps, is_cuda=True)
    matmul_tests.append(pytest.param(CudaTensorBackend, marks=pytest.mark.task3_4))
    backend_tests.append(pytest.param(CudaTensorBackend, marks=pytest.mark.task3_3))


@pytest.mark.parametrize("backend", backend_tests)
@given(lists(floats(allow_nan=False)))
def test_create(backend, t1):
    "Create different tensors."
    t2 = minitorch.tensor(t1)
    for i in range(len(t1)):
        assert t1[i] == t2[i]


@given(data())
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_one_args(fn, backend, data):
    "Run forward for all one arg functions above."
    # backend=TensorBackend #add by wfy to debug
    t1 = data.draw(tensors(backend=backend))
    t2 = fn[1](t1)
    for ind in t2._tensor.indices():
        print(f"{t2[ind]},{fn[1](minitorch.Scalar(t1[ind])).data}")
        assert_close(t2[ind], fn[1](minitorch.Scalar(t1[ind])).data)


@given(data())
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_two_args(fn, backend, data):
    "Run forward for all two arg functions above."
    t1, t2 = data.draw(shaped_tensors(2, backend=backend))
    t3 = fn[1](t1, t2)
    for ind in t3._tensor.indices():
        assert (
            t3[ind] == fn[1](minitorch.Scalar(t1[ind]), minitorch.Scalar(t2[ind])).data
        )


@given(data())
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_one_derivative(fn, backend, data):
    "Run backward for all one arg functions above."
    t1 = data.draw(tensors(backend=backend))
    minitorch.grad_check(fn[1], t1)


@given(data())
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_two_grad(fn, backend, data):
    "Run backward for all two arg functions above."
    t1, t2 = data.draw(shaped_tensors(2, backend=backend))
    minitorch.grad_check(fn[1], t1, t2)


@given(data())
@pytest.mark.parametrize("fn", reduce)
@pytest.mark.parametrize("backend", backend_tests)
def test_reduce(fn, backend, data):
    "Run backward for all reduce functions above."
    t1 = data.draw(tensors(backend=backend))
    minitorch.grad_check(fn[1], t1)


@given(data())
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_two_grad_broadcast(fn, backend, data):
    "Run backward for all two arg functions above with broadcast."

    t1, t2 = data.draw(shaped_tensors(2, backend=backend))
    minitorch.grad_check(fn[1], t1, t2)

    # broadcast check
    minitorch.grad_check(fn[1], t1.sum(0), t2)
    minitorch.grad_check(fn[1], t1, t2.sum(0))


@given(data())
@pytest.mark.parametrize("backend", backend_tests)
def test_permute(backend, data):
    "Check permutations for all backends."
    t1 = data.draw(tensors(backend=backend))
    permutation = data.draw(permutations(range(len(t1.shape))))

    def permute(a):
        return a.permute(*permutation)

    minitorch.grad_check(permute, t1)


@pytest.mark.task3_2
def test_mm2():
    a = minitorch.rand((2, 3), backend=FastTensorBackend)
    b = minitorch.rand((3, 4), backend=FastTensorBackend)
    c = a @ b

    c2 = (a.view(2, 3, 1) * b.view(1, 3, 4)).sum(1).view(2, 4)

    for ind in c._tensor.indices():
        assert_close(c[ind], c2[ind])


# Matrix Multiplication
@given(data())
@pytest.mark.parametrize("backend", matmul_tests)
def test_mm(backend, data):
    small_ints = integers(min_value=2, max_value=4)
    A, B, C, D = (
        data.draw(small_ints),
        data.draw(small_ints),
        data.draw(small_ints),
        data.draw(small_ints),
    )
    a = data.draw(tensors(backend=backend, shape=(D, A, B)))
    b = data.draw(tensors(backend=backend, shape=(1, B, C)))

    c = a @ b
    c2 = (
        (a.contiguous().view(D, A, B, 1) * b.contiguous().view(1, 1, B, C))
        .sum(2)
        .view(D, A, C)
    )
    for ind in c._tensor.indices():
        assert_close(c[ind], c2[ind])
