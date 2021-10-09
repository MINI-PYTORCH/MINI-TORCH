from minitorch.operators import inv
from .autodiff import FunctionBase, Variable, History
from . import operators
import numpy as np


# ## Task 1.1
# Central Difference calculation


def central_difference(f, *vals, arg=0, epsilon=1e-6):
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals (list of floats): n-float values :math:`x_0 \ldots x_{n-1}`
        arg (int): the number :math:`i` of the arg to compute the derivative
        epsilon (float): a small constant

    Returns:
        float : An approximation of :math:`f'_i(x_0, \ldots, x_{n-1})`
    """
    # TODO: Implement for Task 1.1.
    vals_incre=list(vals)
    vals_deces=list(vals)
    vals_incre[arg]+=epsilon
    vals_deces[arg]-=epsilon
    #assert vals_incre!=vals_deces 
    return (f(*vals_incre)-f(*vals_deces))/(2*epsilon)
    # raise NotImplementedError('Need to implement for Task 1.1')


# ## Task 1.2 and 1.4
# Scalar Forward and Backward


class Scalar(Variable):
    """
    A reimplementation of scalar values for autodifferentiation
    tracking.  Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    :class:`ScalarFunction`.

    Attributes:
        data (float): The wrapped scalar value.
    """

    def __init__(self, v, back=History(), name=None):
        super().__init__(back, name=name)
        self.data = float(v)

    def __repr__(self):
        return "Scalar(%f)" % self.data

    def __mul__(self, b):
        return Mul.apply(self, b)

    def __truediv__(self, b):# 表示除法self作被除数
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b):#表示除法self作除数。我们使用乘法和inv来巧妙的避免了除法运算的定义。
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b):
        # TODO: Implement for Task 1.2.
        return Add.apply(self,b)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def __lt__(self, b):
        # TODO: Implement for Task 1.2.
        return LT.apply(self,b)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def __gt__(self, b):
        # TODO: Implement for Task 1.2.
        return LT.apply(b,self)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def __eq__(self, b):
        # TODO: Implement for Task 1.2.
        return EQ.apply(self,b)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def __sub__(self, b):
        # TODO: Implement for Task 1.2.
        return Add.apply(self,Neg.apply(b))
        # raise NotImplementedError('Need to implement for Task 1.2')

    def __neg__(self):
        # TODO: Implement for Task 1.2.
        return Neg.apply(self)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def log(self):
        # TODO: Implement for Task 1.2.
        return Log.apply(self)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def exp(self):
        # TODO: Implement for Task 1.2.
        return Exp.apply(self)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def sigmoid(self):
        # TODO: Implement for Task 1.2.
        return Sigmoid.apply(self)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def relu(self):
        # TODO: Implement for Task 1.2.
        return ReLU.apply(self)
        # raise NotImplementedError('Need to implement for Task 1.2')

    def get_data(self):
        "Returns the raw float value"
        return self.data


class ScalarFunction(FunctionBase):
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @staticmethod
    def forward(ctx, *inputs):
        r"""
        Forward call, compute :math:`f(x_0 \ldots x_{n-1})`.

        Args:
            ctx (:class:`Context`): A container object to save
                                    any information that may be needed
                                    for the call to backward.
            *inputs (list of floats): n-float values :math:`x_0 \ldots x_{n-1}`.

        Should return float the computation of the function :math:`f`.
        """
        pass  # pragma: no cover

    @staticmethod
    def backward(ctx, d_out):
        r"""
        Backward call, computes :math:`f'_{x_i}(x_0 \ldots x_{n-1}) \times d_{out}`.

        Args:
            ctx (Context): A container object holding any information saved during in the corresponding `forward` call.
            d_out (float): :math:`d_out` term in the chain rule.

        Should return the computation of the derivative function
        :math:`f'_{x_i}` for each input :math:`x_i` times `d_out`.

        """
        pass  # pragma: no cover

    # Checks.
    variable = Scalar
    data_type = float

    @staticmethod
    def data(a):
        return a


# Examples
class Add(ScalarFunction):
    "Addition function :math:`f(x, y) = x + y`"

    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, d_output):
        return d_output, d_output


class Log(ScalarFunction):
    "Log function :math:`f(x) = log(x)`"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx, a, b):
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a,b)
        return operators.mul(a,b)
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # TODO: Implement for Task 1.4.
        a,b=ctx.saved_values
        return d_output*b,d_output*a
        # raise NotImplementedError('Need to implement for Task 1.4')


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return operators.inv(a)
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # TODO: Implement for Task 1.4.
        a=ctx.saved_values
        return operators.inv_back(a,d_output)
        # raise NotImplementedError('Need to implement for Task 1.4')


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement for Task 1.2.
        # with open("logs/record.txt",'a') as fp:
        #     fp.write(f"NegYse{operators.neg(a)}\n" if type(operators.neg(a))==int else "NO\n")
        if type(a)==int:
           a=float(a)
        assert type(a)==float #严格讲每个方法都需要检查。
        return operators.neg(a)
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # TODO: Implement for Task 1.4.
        return operators.neg(d_output)
        # raise NotImplementedError('Need to implement for Task 1.4')


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement for Task 1.2.
        sig_a=operators.sigmoid(a)
        ctx.save_for_backward(sig_a)
        return sig_a
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # TODO: Implement for Task 1.4.
        sig_a=ctx.saved_values
        return sig_a*(1-sig_a)*d_output
        # raise NotImplementedError('Need to implement for Task 1.4')


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(1.0 if a>0 else 0.0)

        '''
        for debug
        '''
        # with open("record.txt",'a') as fp:
        #     fp.write(f"ReluYes{operators.relu(a)}\n" if type(operators.relu(a))==int else "NO\n")
        # return float(operators.relu(a))
        return operators.relu(a)
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # TODO: Implement for Task 1.4.
        ans=ctx.saved_values
        return ans*d_output
        # raise NotImplementedError('Need to implement for Task 1.4')


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx, a):
        # TODO: Implement for Task 1.2.
        exp_a=operators.exp(a)
        ctx.save_for_backward(exp_a)
        return exp_a
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod
    def backward(ctx, d_output):
        # TODO: Implement for Task 1.4.
        exp_a=ctx.saved_values
        return exp_a*d_output
        # raise NotImplementedError('Need to implement for Task 1.4')


class LT(ScalarFunction):
    "Less-than function :math:`f(x) =` 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx, a, b):

        return operators.lt(a,b)
        # TODO: Implement for Task 1.2.
        raise NotImplementedError('Need to implement for Task 1.2')
        # TODO: Implement for Task 1.2.
        raise NotImplementedError('Need to implement for Task 1.2')
    @staticmethod
    def backward(ctx, d_out):
        # TODO: Implement for Task 1.4.
        return 0.0,0.0
        # raise NotImplementedError('Need to implement for Task 1.4')


class EQ(ScalarFunction):
    "Equal function :math:`f(x) =` 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx, a, b):
        # TODO: Implement for Task 1.2.
        return operators.eq(a,b)
        # raise NotImplementedError('Need to implement for Task 1.2')

    @staticmethod 
    def backward(ctx, d_output):
        # TODO: Implement for Task 1.4.
        return 0.0,0.0
        # pass
        # raise NotImplementedError('Need to implement for Task 1.4')


def derivative_check(f, *scalars):
    """
    Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters:
        f (function) : function from n-scalars to 1-scalar.
        *scalars (list of :class:`Scalar`) : n input scalar values.
    """
    for x in scalars:
        x.requires_grad_(True)
    out = f(*scalars)
    out.backward()

    vals = [v for v in scalars]
    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *vals, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
