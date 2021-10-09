import streamlit as st
from interface.streamlit_utils import get_img_tag
from interface.train import render_train_interface
import sys
from run_torch import TorchTrain

module_num = int(sys.argv[1])

st.set_page_config(page_title="interactive minitorch")
st.sidebar.markdown(
    """
<h1 style="font-size:30pt; float: left; margin-right: 20px; margin-top: 1px;">MiniTorch</h1>{}
""".format(
        get_img_tag("https://minitorch.github.io/_images/match.png", width="40")
    ),
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    [Documentation](https://minitorch.github.io/)
"""
)

module_selection = st.sidebar.radio(
    "Modle",
    ["Module 0", "Module 1", "Module 2", "Module 3", "Module 4"][: module_num + 1],
)


PAGES = {}

if module_selection == "Module 0":
    from run_manual import ManualTrain
    from module_interface import render_module_sandbox
    from math_interface import render_math_sandbox

    def render_run_manual_interface():
        st.header("Module 0 - Manual")
        render_train_interface(ManualTrain, False, False, True)

    def render_m0_sandbox():
        return render_math_sandbox(False)

    PAGES["Math Sandbox"] = render_m0_sandbox
    PAGES["Module Sandbox"] = render_module_sandbox

    def render_run_torch_interface():
        st.header("Demo - Torch")
        render_train_interface(TorchTrain, False)

    PAGES["Torch Example"] = render_run_torch_interface
    PAGES["Module 0: Manual"] = render_run_manual_interface

if module_selection == "Module 1":
    from run_scalar import ScalarTrain
    from show_expression_interface import render_show_expression

    def render_m1_sandbox():
        return render_math_sandbox(True)

    def render_run_scalar_interface():
        st.header("Module 1 - Scalars")
        render_train_interface(ScalarTrain)

    PAGES["Scalar Sandbox"] = render_m1_sandbox
    PAGES["Autodiff Sandbox"] = render_show_expression
    PAGES["Module 1: Scalar"] = render_run_scalar_interface

if module_selection == "Module 2":
    from run_tensor import TensorTrain

    def render_run_tensor_interface():
        st.header("Module 2 - Tensors")
        render_train_interface(TensorTrain)

    PAGES["Module 2: Tensor"] = render_run_tensor_interface


if module_selection == "Module 3":
    from run_fast_tensor import FastTrain

    def render_run_fast_interface():
        st.header("Module 3 - Efficient")
        render_train_interface(FastTrain, False)

    PAGES["Module 3: Efficient"] = render_run_fast_interface

if module_selection == "Module 4":
    from run_mnist_interface import render_run_image_interface

    PAGES["Module 4: Images"] = render_run_image_interface


PAGE_OPTIONS = list(PAGES.keys())

page_selection = st.sidebar.radio("Pages", PAGE_OPTIONS)
page = PAGES[page_selection]
page()
