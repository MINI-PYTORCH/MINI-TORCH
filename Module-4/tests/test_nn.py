import minitorch
from hypothesis import given
from .strategies import tensors, assert_close
import pytest

@pytest.mark.task4_3
def test_tile():
    t = minitorch.tensor_fromlist(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).view(1, 1, 4, 4)
    tiled, _, _ = minitorch.tile(t, (2, 2))
    assert tiled[0, 0, 0, 0, 0] == 1
    assert tiled[0, 0, 0, 0, 1] == 2
    assert tiled[0, 0, 0, 0, 2] == 5
    assert tiled[0, 0, 0, 0, 3] == 6

@pytest.mark.task4_3
def test_tile_2():
    t = minitorch.tensor_fromlist(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).view(1, 1, 4, 4)
    tiled, _, _ = minitorch.tile(t, (1, 2))
    assert tiled[0, 0, 0, 0, 0] == 1
    assert tiled[0, 0, 0, 0, 1] == 2
    assert tiled[0, 0, 0, 1, 0] == 3
    assert tiled[0, 0, 0, 1, 1] == 4

@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t):
    out = minitorch.avgpool2d(t, (2, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t):
    # TODO: Implement for Task 4.4.
    pass
    # raise NotImplementedError('Need to implement for Task 4.4')


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t):
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t):
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t):
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)
