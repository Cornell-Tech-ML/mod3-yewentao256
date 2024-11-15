# MiniTorch Module 3

![MiniTorch Logo](https://minitorch.github.io/minitorch.svg)

* Docs: [https://minitorch.github.io/]

* Overview: [https://minitorch.github.io/module3.html]

You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```bash
python run_tests.py
```

```powershell
$env:NUMBA_DISABLE_JIT = "1"; pytest tests/ -m task3_1
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

```bash
        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py
```

## Task 3.1

Diagnostics output from `python project/parallel_check.py`:

Todo here
