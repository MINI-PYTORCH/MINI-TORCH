class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules (dict of name x :class:`Module`): Storage of the child modules
        _parameters (dict of name x :class:`Parameter`): Storage of the module's parameters
        training (bool): Whether the module is in training mode or evaluation mode

    """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self):
        "Return the direct child modules of this module."
        return self.__dict__["_modules"].values()

    def train(self):
        "Set the mode of this module and all descendent modules to `train`."
        # TODO: Implement for Task 0.4.
        def train_recursion(module):
            module.training=True
            for submodule in module.modules():
                train_recursion(submodule)
        train_recursion(self)
        # raise NotImplementedError('Need to implement for Task 0.4')

    def eval(self):
        "Set the mode of this module and all descendent modules to `eval`."
        # TODO: Implement for Task 0.4.
        def eval_recursion(module):
            module.training=False
            for submodule in module.modules():
                eval_recursion(submodule)
        eval_recursion(self)
        # raise NotImplementedError('Need to implement for Task 0.4')

    def named_parameters(self):
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            list of pairs: Contains the name and :class:`Parameter` of each ancestor parameter.
        """
        # TODO: Implement for Task 0.4.

        ans=[]
        def named_parameters_recursion(module,prefix):
            tmp=prefix + '.' if prefix else ''
            for name,value in module.__dict__["_parameters"].items():
                ans.append((tmp+name,value))
            for name,submodule in module.__dict__["_modules"].items():
                prefix=tmp+name
                named_parameters_recursion(submodule,prefix)
        named_parameters_recursion(self,'')
        return ans
        # raise NotImplementedError('Need to implement for Task 0.4')

    def parameters(self):
        "Enumerate over all the parameters of this module and its descendents."
        # TODO: Implement for Task 0.4.

        return list(map(lambda x: x[1],self.named_parameters()))
        # raise NotImplementedError('Need to implement for Task 0.4')

    def add_parameter(self, k, v):
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k (str): Local name of the parameter.
            v (value): Value for the parameter.

        Returns:
            Parameter: Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key, val):#每次类的实例进行属性赋值都会调用该函数。
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):#当使用某个不存在的属性时，如module.a,a不是其中的属性就会将a作为参数执行该函数。
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        assert False, "Not Implemented"

    def __repr__(self):#用于自定义对象的输出信息
        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x=None, name=None):
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x):
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)
