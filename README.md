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
pytest tests/ -m task3_1
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

- [MiniTorch Module 3](#minitorch-module-3)
  - [Task 3.1 \& 3.2](#task-31--32)
  - [Task 3.3](#task-33)
      - [Unit test result](#unit-test-result)
  - [Task 3.4](#task-34)
    - [Unit Test Result](#unit-test-result-1)
    - [Document of MM](#document-of-mm)
      - [MM\_practice](#mm_practice)
      - [tensor\_matrix\_multiply](#tensor_matrix_multiply)
  - [Task 3.5](#task-35)
    - [CPU, Hidden 100](#cpu-hidden-100)
      - [Split](#split)
      - [Xor](#xor)
      - [Simple](#simple)
    - [CPU Hidden 200](#cpu-hidden-200)
    - [GPU Hidden 100](#gpu-hidden-100)
      - [Split](#split-1)
      - [Xor](#xor-1)
      - [Simple](#simple-1)
    - [GPU Hidden 200](#gpu-hidden-200)

## Task 3.1 & 3.2

Diagnostics output from `python project/parallel_check.py`:

```bash
python project/parallel_check.py
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
C:\Users\Peter\Desktop\mod3-yewentao256\minitorch\fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, C:\Users\Peter\Desktop\mod3-yewentao256\minitorch\fast_ops.py (163)
-----------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                            |
        out: Storage,                                                                    |
        out_shape: Shape,                                                                |
        out_strides: Strides,                                                            |
        in_storage: Storage,                                                             |
        in_shape: Shape,                                                                 |
        in_strides: Strides,                                                             |
    ) -> None:                                                                           |
        out_size = len(out)                                                              |
        if np.array_equal(out_strides, in_strides) and np.array_equal(                   |
            out_shape, in_shape                                                          |
        ):                                                                               |
            # Stride-aligned case: apply function directly to corresponding elements.    |
            for i in prange(out_size):---------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                               |
        else:                                                                            |
            # Non-aligned case: use indexing                                             |
            out_index = np.empty(len(out_shape), dtype=np.int32)                         |
            in_index = np.empty(len(in_shape), dtype=np.int32)                           |
            for i in prange(out_size):---------------------------------------------------| #1
                to_index(i, out_shape, out_index)                                        |
                broadcast_index(out_index, out_shape, in_shape, in_index)                |
                out_pos = index_to_position(out_index, out_strides)                      |
                in_pos = index_to_position(in_index, in_strides)                         |
                out[out_pos] = fn(in_storage[in_pos])                                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
C:\Users\Peter\Desktop\mod3-yewentao256\minitorch\fast_ops.py (215)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\Peter\Desktop\mod3-yewentao256\minitorch\fast_ops.py (215)
--------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                   |
        out: Storage,                                                           |
        out_shape: Shape,                                                       |
        out_strides: Strides,                                                   |
        a_storage: Storage,                                                     |
        a_shape: Shape,                                                         |
        a_strides: Strides,                                                     |
        b_storage: Storage,                                                     |
        b_shape: Shape,                                                         |
        b_strides: Strides,                                                     |
    ) -> None:                                                                  |
        out_size = len(out)                                                     |
        aligned = True                                                          |
        if len(out_shape) != len(a_shape) or len(out_shape) != len(b_shape):    |
            aligned = False                                                     |
        else:                                                                   |
            for i in range(len(out_shape)):                                     |
                if (                                                            |
                    out_shape[i] != a_shape[i]                                  |
                    or out_shape[i] != b_shape[i]                               |
                    or out_strides[i] != a_strides[i]                           |
                    or out_strides[i] != b_strides[i]                           |
                ):                                                              |
                    aligned = False                                             |
                    break                                                       |
        if aligned:                                                             |
            # Strides and shapes are aligned; avoid indexing                    |
            for i in prange(out_size):------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                         |
        else:                                                                   |
            out_index = np.empty(len(out_shape), dtype=np.int32)                |
            a_index = np.empty(len(a_shape), dtype=np.int32)                    |
            b_index = np.empty(len(b_shape), dtype=np.int32)                    |
            for i in prange(out_size):------------------------------------------| #3
                to_index(i, out_shape, out_index)                               |
                broadcast_index(out_index, out_shape, a_shape, a_index)         |
                broadcast_index(out_index, out_shape, b_shape, b_index)         |
                out_pos = index_to_position(out_index, out_strides)             |
                a_pos = index_to_position(a_index, a_strides)                   |
                b_pos = index_to_position(b_index, b_strides)                   |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])           |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
C:\Users\Peter\Desktop\mod3-yewentao256\minitorch\fast_ops.py (281)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\Peter\Desktop\mod3-yewentao256\minitorch\fast_ops.py (281)
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   |
        out: Storage,                                              |
        out_shape: Shape,                                          |
        out_strides: Strides,                                      |
        a_storage: Storage,                                        |
        a_shape: Shape,                                            |
        a_strides: Strides,                                        |
        reduce_dim: int,                                           |
    ) -> None:                                                     |
        out_index = np.empty(len(out_shape), dtype=np.int32)       |
        for i in prange(len(out)):---------------------------------| #4
            to_index(i, out_shape, out_index)                      |
            out_pos = index_to_position(out_index, out_strides)    |
            total = out[out_pos]                                   |
                                                                   |
            a_pos = index_to_position(out_index, a_strides)        |
            stride = a_strides[reduce_dim]                         |
            size = a_shape[reduce_dim]                             |
                                                                   |
            for _ in range(size):                                  |
                total = fn(total, a_storage[a_pos])                |
                a_pos += stride                                    |
                                                                   |
            out[out_pos] = total                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
C:\Users\Peter\Desktop\mod3-yewentao256\minitorch\fast_ops.py (309)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\Peter\Desktop\mod3-yewentao256\minitorch\fast_ops.py (309)
------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                            |
    out: Storage,                                                       |
    out_shape: Shape,                                                   |
    out_strides: Strides,                                               |
    a_storage: Storage,                                                 |
    a_shape: Shape,                                                     |
    a_strides: Strides,                                                 |
    b_storage: Storage,                                                 |
    b_shape: Shape,                                                     |
    b_strides: Strides,                                                 |
) -> None:                                                              |
    """NUMBA tensor matrix multiply function.                           |
                                                                        |
    Should work for any tensor shapes that broadcast as long as         |
                                                                        |
    ```                                                                 |
    assert a_shape[-1] == b_shape[-2]                                   |
    ```                                                                 |
                                                                        |
    Optimizations:                                                      |
                                                                        |
    * Outer loop in parallel                                            |
    * No index buffers or function calls                                |
    * Inner loop should have no global writes, 1 multiply.              |
                                                                        |
                                                                        |
    Args:                                                               |
    ----                                                                |
        out (Storage): storage for `out` tensor                         |
        out_shape (Shape): shape for `out` tensor                       |
        out_strides (Strides): strides for `out` tensor                 |
        a_storage (Storage): storage for `a` tensor                     |
        a_shape (Shape): shape for `a` tensor                           |
        a_strides (Strides): strides for `a` tensor                     |
        b_storage (Storage): storage for `b` tensor                     |
        b_shape (Shape): shape for `b` tensor                           |
        b_strides (Strides): strides for `b` tensor                     |
                                                                        |
    Returns:                                                            |
    -------                                                             |
        None : Fills in `out`                                           |
                                                                        |
    """                                                                 |
    B, M, N = out_shape                                                 |
    K = a_shape[2]  # Must be equal to b_shape[1]                       |
    assert a_shape[-1] == b_shape[-2]                                   |
                                                                        |
    b_stride_out, m_stride_out, n_stride_out = out_strides              |
                                                                        |
    b_stride_a = a_strides[0] if a_shape[0] != 1 else 0                 |
    m_stride_a = a_strides[1] if a_shape[1] != 1 else 0                 |
    k_stride_a = a_strides[2] if a_shape[2] != 1 else 0                 |
                                                                        |
    b_stride_b = b_strides[0] if b_shape[0] != 1 else 0                 |
    k_stride_b = b_strides[1] if b_shape[1] != 1 else 0                 |
    n_stride_b = b_strides[2] if b_shape[2] != 1 else 0                 |
                                                                        |
    for b in prange(B):-------------------------------------------------| #5
        b_a = b * b_stride_a                                            |
        b_b = b * b_stride_b                                            |
        b_out = b * b_stride_out                                        |
                                                                        |
        for m in range(M):                                              |
            a_m = b_a + m * m_stride_a                                  |
            out_index = b_out + m * m_stride_out                        |
            for n in range(N):                                          |
                total = 0.0                                             |
                                                                        |
                a_index = a_m                                           |
                b_index = b_b + n * n_stride_b                          |
                for _ in range(K):                                      |
                    total += a_storage[a_index] * b_storage[b_index]    |
                    a_index += k_stride_a                               |
                    b_index += k_stride_b                               |
                                                                        |
                out[out_index] = total                                  |
                out_index += n_stride_out                               |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

## Task 3.3

#### Unit test result

```bash
======================================= test session starts ========================================
platform linux -- Python 3.10.12, pytest-8.3.3, pluggy-1.5.0
rootdir: /content/drive/MyDrive/mod3-yewentao256
configfile: pyproject.toml
plugins: hypothesis-6.119.3, typeguard-4.4.1, anyio-3.7.1
collected 117 items / 60 deselected / 57 selected

tests/test_tensor_general.py .........................................................       [100%]

========================================= warnings summary =========================================
tests/test_tensor_general.py: 16 warnings
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py: 69 warnings
  /usr/local/lib/python3.10/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py: 13 warnings
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_args[cuda-fn12]
tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
tests/test_tensor_general.py::test_two_grad[cuda-fn4]
tests/test_tensor_general.py::test_sum_practice2
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 3 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 9 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 18 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 6 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 27 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn1]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn1]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_sum_practice_other_dims
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=================== 57 passed, 60 deselected, 110 warnings in 278.49s (0:04:38) ====================
```

## Task 3.4

### Unit Test Result

```bash
======================================= test session starts ========================================
platform linux -- Python 3.10.12, pytest-8.3.3, pluggy-1.5.0
rootdir: /content/drive/MyDrive/mod3-yewentao256
configfile: pyproject.toml
plugins: hypothesis-6.119.3, typeguard-4.4.1, anyio-3.7.1
collected 117 items / 110 deselected / 7 selected

tests/test_tensor_general.py .......                                                         [100%]

========================================= warnings summary =========================================
tests/test_tensor_general.py::test_mul_practice1
tests/test_tensor_general.py::test_mul_practice3
tests/test_tensor_general.py::test_mul_practice3
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py: 12 warnings
  /usr/local/lib/python3.10/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_mul_practice4
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_mul_practice4
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_mul_practice5
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 3 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 5 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 36 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 18 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 24 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 27 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 64 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 6 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.10/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 48 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================= 7 passed, 110 deselected, 38 warnings in 10.80s ==========================
```

### Document of MM

#### MM_practice

```py
def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # load data into shared memory
    if tx < size and ty < size:
        a_shared[tx, ty] = a[tx * size + ty]
        b_shared[tx, ty] = b[tx * size + ty]

    cuda.syncthreads()

    # compute the matrix multiplication
    if tx < size and ty < size:
        sum = 0.0
        for k in range(size):
            sum += a_shared[tx, k] * b_shared[k, ty]
        out[tx * size + ty] = sum
```

* `tx` (`threadIdx.x`): Represents the thread's row index within the block.
* `ty` (`threadIdx.y`): Represents the thread's col index within the block.
* `i` and `j` are not explicitly used in this function because all can be mapped to `tx` and `ty` for indexing.

**Example:**

For a `size = 4` matrix, each thread `(tx, ty)` computes the element at row `ty` and column `tx` of the output matrix by iterating over `k` from `0` to `3` and accumulating the product `a_shared[ty, k] * b_shared[k, tx]`.

#### tensor_matrix_multiply

```py
def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    # a_shape = (B, M, K), b_shape = (B, K, N), out_shape = (B, M, N)
    # declare the variables to reduce index operations
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    M = a_shape[1]
    K = a_shape[2]
    assert K == b_shape[1]
    N = b_shape[2]
    a_m_strides = a_strides[1]
    a_k_strides = a_strides[2]
    b_k_strides = b_strides[1]
    b_n_strides = b_strides[2]
    batch = cuda.blockIdx.z
    BLOCK_DIM = 32

    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    result = 0.0
    # number of tiles to cover the K dimension
    tiles = (K + BLOCK_DIM - 1) // BLOCK_DIM
    a_m = i
    a_k = ty
    b_k = tx
    b_n = j
    # here we precompute the start to reduce the number of operations
    a_start = batch * a_batch_stride + a_m * a_m_strides
    b_start = batch * b_batch_stride + b_n * b_n_strides
    for t in range(tiles):
        # guard: check if the current tile is within the bounds of the matrix
        if a_m < M and a_k < K:
            a_index = a_start + a_k * a_k_strides
            a_shared[tx, ty] = a_storage[a_index]

        if b_k < K and b_n < N:
            b_index = b_start + b_k * b_k_strides
            # trick: to improve reading cache hit later
            b_shared[ty, tx] = b_storage[b_index]

        # sync threads to make sure all data are loaded
        cuda.syncthreads()

        # multiplication for the current tile
        for k in range(BLOCK_DIM):
            if (t * BLOCK_DIM + k) < K:
                result += a_shared[tx, k] * b_shared[ty, k]

        # we add block_dim to move to the next tile
        a_k += BLOCK_DIM
        b_k += BLOCK_DIM

    if i < M and j < N:
        # finally we write the result to the output
        out_index = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_index] = result
```

Understanding of each line could be seen in the code comments.

**Key Components**:

1. **Tiling (`tiles`):**
   * `tiles = (K + BLOCK_DIM - 1) // BLOCK_DIM`: Number of tiles needed to cover the `K` dimension.
   * The `K` dimension is processed in chunks (tiles) of size `BLOCK_DIM` to fit into shared memory.

2. **Shared Memory Utilization:**
   * `a_shared` and `b_shared` store tiles of matrices `a` and `b` respectively.
   * Transposing `b_shared` (`b_shared[ty, tx]`) improves memory access patterns and cache performance.

**Example**:

**`a_shape = (16, 1024, 512)`**, **`b_shape = (16, 512, 2048)`**, **`out_shape = (16, 1024, 2048)`**

Compute the element `(i=100, j=1500)` in `out[5][100][1500]` for batch `5`:

1. **Initialize Variables:**
   * `batch = 5`, `i = 100`, `j = 1500`
   * `BLOCK_DIM = 32`
   * `tiles = 16` (since `(512 + 32 - 1) // 32 = 16`)

2. **Precompute Start Positions:**
   * `a_start = 5 * a_batch_stride + 100 * a_m_strides`
   * `b_start = 5 * b_batch_stride + 1500 * b_n_strides`

3. **Iterate Over Each Tile (`t` from `0` to `15`):**
   * **Load Tiles into Shared Memory:**

     ```python
     if a_m < 1024 and a_k < 512:
         a_shared[tx, ty] = a_storage[a_start + a_k * a_k_strides]

     if b_k < 512 and b_n < 2048:
         b_shared[ty, tx] = b_storage[b_start + b_k * b_k_strides]
     ```

   * **Synchronize Threads:**

     ```python
     cuda.syncthreads()
     ```

   * **Compute Partial Results:**

     ```python
     for k in range(32):
         if (t * 32 + k) < 512:
             result += a_shared[tx, k] * b_shared[ty, k]
     ```

   * **Move to Next Tile:**

     ```python
     a_k += 32
     b_k += 32
     ```

4. **Write the Result:**

   ```python
   if 100 < 1024 and 1500 < 2048:
       out[out_index] = result
   ```

---

Performance Benefits

* **Parallelism:** Utilizes thousands of GPU threads to perform computations concurrently, drastically reducing computation time for large batches.
* **Shared Memory:** Minimizes slow global memory accesses by storing frequently used data in fast shared memory.
* **Tiling Strategy:** Efficiently handles large `K` dimensions by processing data in manageable chunks, ensuring scalability.
* **Optimized Memory Access:** Transposing one of the matrices enhances cache utilization and memory bandwidth.

---

Speed-Up Comparison

```bash
Timing summary
Size: 64
    fast: 0.00234
    gpu: 0.00833
Size: 128
    fast: 0.01068
    gpu: 0.01678
Size: 256
    fast: 0.06967
    gpu: 0.05102
Size: 512
    fast: 0.32548
    gpu: 0.18282
Size: 1024
    fast: 3.74572
    gpu: 0.82267
```

## Task 3.5

### CPU, Hidden 100

Note: Run on colab.

#### Split

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch 10 | Loss: 27.4078 | Correct: 31 | Time: 1.93 sec
Epoch 20 | Loss: 25.4781 | Correct: 45 | Time: 0.12 sec
Epoch 30 | Loss: 21.6217 | Correct: 50 | Time: 0.12 sec
Epoch 40 | Loss: 15.9520 | Correct: 45 | Time: 0.12 sec
Epoch 50 | Loss: 13.9351 | Correct: 49 | Time: 0.11 sec
Epoch 60 | Loss: 10.8578 | Correct: 50 | Time: 0.20 sec
Epoch 70 | Loss: 9.5381 | Correct: 50 | Time: 0.21 sec
Epoch 80 | Loss: 9.8857 | Correct: 47 | Time: 0.12 sec
Epoch 90 | Loss: 7.2538 | Correct: 48 | Time: 0.12 sec
Epoch 100 | Loss: 6.8134 | Correct: 50 | Time: 0.12 sec
Epoch 110 | Loss: 6.4941 | Correct: 50 | Time: 0.12 sec
Epoch 120 | Loss: 6.5543 | Correct: 48 | Time: 0.12 sec
Epoch 130 | Loss: 5.9851 | Correct: 50 | Time: 0.12 sec
Epoch 140 | Loss: 4.5269 | Correct: 50 | Time: 0.11 sec
Epoch 150 | Loss: 4.0810 | Correct: 50 | Time: 0.12 sec
Epoch 160 | Loss: 4.3591 | Correct: 50 | Time: 0.20 sec
Epoch 170 | Loss: 3.4304 | Correct: 50 | Time: 0.20 sec
Epoch 180 | Loss: 4.1339 | Correct: 50 | Time: 0.12 sec
Epoch 190 | Loss: 3.2895 | Correct: 50 | Time: 0.12 sec
Epoch 200 | Loss: 3.0836 | Correct: 50 | Time: 0.12 sec
Epoch 210 | Loss: 3.0192 | Correct: 50 | Time: 0.14 sec
Epoch 220 | Loss: 2.6809 | Correct: 50 | Time: 0.13 sec
Epoch 230 | Loss: 2.8596 | Correct: 50 | Time: 0.12 sec
Epoch 240 | Loss: 2.4759 | Correct: 50 | Time: 0.12 sec
Epoch 250 | Loss: 2.3459 | Correct: 50 | Time: 0.12 sec
Epoch 260 | Loss: 2.2033 | Correct: 50 | Time: 0.22 sec
Epoch 270 | Loss: 2.1722 | Correct: 50 | Time: 0.18 sec
Epoch 280 | Loss: 2.2868 | Correct: 50 | Time: 0.12 sec
Epoch 290 | Loss: 2.0230 | Correct: 50 | Time: 0.12 sec
Epoch 300 | Loss: 1.8129 | Correct: 50 | Time: 0.12 sec
Epoch 310 | Loss: 1.8888 | Correct: 50 | Time: 0.12 sec
Epoch 320 | Loss: 1.7843 | Correct: 50 | Time: 0.12 sec
Epoch 330 | Loss: 1.5268 | Correct: 50 | Time: 0.11 sec
Epoch 340 | Loss: 1.5397 | Correct: 50 | Time: 0.12 sec
Epoch 350 | Loss: 1.8162 | Correct: 50 | Time: 0.11 sec
Epoch 360 | Loss: 1.7482 | Correct: 50 | Time: 0.21 sec
Epoch 370 | Loss: 1.5181 | Correct: 50 | Time: 0.19 sec
Epoch 380 | Loss: 1.6860 | Correct: 50 | Time: 0.12 sec
Epoch 390 | Loss: 1.4895 | Correct: 50 | Time: 0.13 sec
Epoch 400 | Loss: 1.4166 | Correct: 50 | Time: 0.12 sec
Epoch 410 | Loss: 1.4108 | Correct: 50 | Time: 0.12 sec
Epoch 420 | Loss: 1.1040 | Correct: 50 | Time: 0.12 sec
Epoch 430 | Loss: 1.2211 | Correct: 50 | Time: 0.12 sec
Epoch 440 | Loss: 1.0695 | Correct: 50 | Time: 0.12 sec
Epoch 450 | Loss: 1.0950 | Correct: 50 | Time: 0.11 sec
Epoch 460 | Loss: 1.0860 | Correct: 50 | Time: 0.22 sec
Epoch 470 | Loss: 0.9513 | Correct: 50 | Time: 0.18 sec
Epoch 480 | Loss: 0.9493 | Correct: 50 | Time: 0.12 sec
Epoch 490 | Loss: 1.1880 | Correct: 50 | Time: 0.12 sec
Epoch 500 | Loss: 1.0895 | Correct: 50 | Time: 0.12 sec
```

#### Xor

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05  Epoch 10 | Loss: 26.1128 | Correct: 43 | Time: 1.90 sec
Epoch 20 | Loss: 22.1774 | Correct: 43 | Time: 0.12 sec
Epoch 30 | Loss: 18.3871 | Correct: 45 | Time: 0.11 sec
Epoch 40 | Loss: 16.0804 | Correct: 49 | Time: 0.12 sec
Epoch 50 | Loss: 14.1895 | Correct: 47 | Time: 0.13 sec
Epoch 60 | Loss: 12.0619 | Correct: 49 | Time: 0.24 sec
Epoch 70 | Loss: 10.5581 | Correct: 50 | Time: 0.15 sec
Epoch 80 | Loss: 10.0349 | Correct: 49 | Time: 0.12 sec
Epoch 90 | Loss: 9.1916 | Correct: 49 | Time: 0.12 sec
Epoch 100 | Loss: 7.7193 | Correct: 50 | Time: 0.12 sec
Epoch 110 | Loss: 8.3861 | Correct: 49 | Time: 0.12 sec
Epoch 120 | Loss: 6.3123 | Correct: 49 | Time: 0.13 sec
Epoch 130 | Loss: 6.2717 | Correct: 50 | Time: 0.13 sec
Epoch 140 | Loss: 5.3559 | Correct: 49 | Time: 0.12 sec
Epoch 150 | Loss: 5.4213 | Correct: 50 | Time: 0.16 sec
Epoch 160 | Loss: 4.4314 | Correct: 50 | Time: 0.21 sec
Epoch 170 | Loss: 4.4081 | Correct: 50 | Time: 0.14 sec
Epoch 180 | Loss: 4.5179 | Correct: 50 | Time: 0.12 sec
Epoch 190 | Loss: 4.1289 | Correct: 50 | Time: 0.12 sec
Epoch 200 | Loss: 3.5164 | Correct: 50 | Time: 0.12 sec
Epoch 210 | Loss: 3.3055 | Correct: 50 | Time: 0.12 sec
Epoch 220 | Loss: 3.0784 | Correct: 50 | Time: 0.12 sec
Epoch 230 | Loss: 3.2190 | Correct: 50 | Time: 0.12 sec
Epoch 240 | Loss: 2.7236 | Correct: 50 | Time: 0.12 sec
Epoch 250 | Loss: 2.8258 | Correct: 50 | Time: 0.14 sec
Epoch 260 | Loss: 2.5071 | Correct: 50 | Time: 0.21 sec
Epoch 270 | Loss: 2.6128 | Correct: 50 | Time: 0.15 sec
Epoch 280 | Loss: 2.3742 | Correct: 50 | Time: 0.12 sec
Epoch 290 | Loss: 2.1312 | Correct: 50 | Time: 0.12 sec
Epoch 300 | Loss: 2.0916 | Correct: 50 | Time: 0.12 sec
Epoch 310 | Loss: 2.0191 | Correct: 50 | Time: 0.12 sec
Epoch 320 | Loss: 1.9421 | Correct: 50 | Time: 0.12 sec
Epoch 330 | Loss: 1.7482 | Correct: 50 | Time: 0.11 sec
Epoch 340 | Loss: 1.8802 | Correct: 50 | Time: 0.11 sec
Epoch 350 | Loss: 1.7060 | Correct: 50 | Time: 0.12 sec
Epoch 360 | Loss: 1.6244 | Correct: 50 | Time: 0.21 sec
Epoch 370 | Loss: 1.6074 | Correct: 50 | Time: 0.18 sec
Epoch 380 | Loss: 1.4285 | Correct: 50 | Time: 0.11 sec
Epoch 390 | Loss: 1.3624 | Correct: 50 | Time: 0.11 sec
Epoch 400 | Loss: 1.3379 | Correct: 50 | Time: 0.11 sec
Epoch 410 | Loss: 1.2829 | Correct: 50 | Time: 0.11 sec
Epoch 420 | Loss: 1.2054 | Correct: 50 | Time: 0.11 sec
Epoch 430 | Loss: 1.2555 | Correct: 50 | Time: 0.11 sec
Epoch 440 | Loss: 1.1083 | Correct: 50 | Time: 0.11 sec
Epoch 450 | Loss: 1.1390 | Correct: 50 | Time: 0.11 sec
Epoch 460 | Loss: 1.0251 | Correct: 50 | Time: 0.18 sec
Epoch 470 | Loss: 0.9988 | Correct: 50 | Time: 0.21 sec
Epoch 480 | Loss: 1.0158 | Correct: 50 | Time: 0.15 sec
Epoch 490 | Loss: 0.9522 | Correct: 50 | Time: 0.22 sec
Epoch 500 | Loss: 0.9599 | Correct: 50 | Time: 0.14 sec
```

#### Simple

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
Epoch 10 | Loss: 7.4525 | Correct: 49 | Time: 1.86 sec
Epoch 20 | Loss: 4.6225 | Correct: 50 | Time: 0.18 sec
Epoch 30 | Loss: 3.4083 | Correct: 50 | Time: 0.25 sec
Epoch 40 | Loss: 2.9572 | Correct: 50 | Time: 0.16 sec
Epoch 50 | Loss: 2.2034 | Correct: 50 | Time: 0.12 sec
Epoch 60 | Loss: 1.9473 | Correct: 50 | Time: 0.12 sec
Epoch 70 | Loss: 1.5705 | Correct: 50 | Time: 0.12 sec
Epoch 80 | Loss: 1.5642 | Correct: 50 | Time: 0.12 sec
Epoch 90 | Loss: 1.3389 | Correct: 50 | Time: 0.12 sec
Epoch 100 | Loss: 1.1674 | Correct: 50 | Time: 0.12 sec
Epoch 110 | Loss: 1.0466 | Correct: 50 | Time: 0.12 sec
Epoch 120 | Loss: 1.0110 | Correct: 50 | Time: 0.12 sec
Epoch 130 | Loss: 0.8894 | Correct: 50 | Time: 0.21 sec
Epoch 140 | Loss: 0.8415 | Correct: 50 | Time: 0.19 sec
Epoch 150 | Loss: 0.8120 | Correct: 50 | Time: 0.12 sec
Epoch 160 | Loss: 0.7765 | Correct: 50 | Time: 0.12 sec
Epoch 170 | Loss: 0.7300 | Correct: 50 | Time: 0.12 sec
Epoch 180 | Loss: 0.6857 | Correct: 50 | Time: 0.12 sec
Epoch 190 | Loss: 0.6436 | Correct: 50 | Time: 0.12 sec
Epoch 200 | Loss: 0.6257 | Correct: 50 | Time: 0.11 sec
Epoch 210 | Loss: 0.5999 | Correct: 50 | Time: 0.12 sec
Epoch 220 | Loss: 0.5787 | Correct: 50 | Time: 0.12 sec
Epoch 230 | Loss: 0.5556 | Correct: 50 | Time: 0.18 sec
Epoch 240 | Loss: 0.5200 | Correct: 50 | Time: 0.21 sec
Epoch 250 | Loss: 0.4932 | Correct: 50 | Time: 0.12 sec
Epoch 260 | Loss: 0.4936 | Correct: 50 | Time: 0.12 sec
Epoch 270 | Loss: 0.4579 | Correct: 50 | Time: 0.12 sec
Epoch 280 | Loss: 0.4403 | Correct: 50 | Time: 0.12 sec
Epoch 290 | Loss: 0.4341 | Correct: 50 | Time: 0.12 sec
Epoch 300 | Loss: 0.4054 | Correct: 50 | Time: 0.11 sec
Epoch 310 | Loss: 0.3906 | Correct: 50 | Time: 0.11 sec
Epoch 320 | Loss: 0.3828 | Correct: 50 | Time: 0.13 sec
Epoch 330 | Loss: 0.3643 | Correct: 50 | Time: 0.19 sec
Epoch 340 | Loss: 0.3649 | Correct: 50 | Time: 0.22 sec
Epoch 350 | Loss: 0.3528 | Correct: 50 | Time: 0.11 sec
Epoch 360 | Loss: 0.3366 | Correct: 50 | Time: 0.11 sec
Epoch 370 | Loss: 0.3231 | Correct: 50 | Time: 0.12 sec
Epoch 380 | Loss: 0.3234 | Correct: 50 | Time: 0.13 sec
Epoch 390 | Loss: 0.3132 | Correct: 50 | Time: 0.12 sec
Epoch 400 | Loss: 0.3023 | Correct: 50 | Time: 0.12 sec
Epoch 410 | Loss: 0.2955 | Correct: 50 | Time: 0.12 sec
Epoch 420 | Loss: 0.2873 | Correct: 50 | Time: 0.12 sec
Epoch 430 | Loss: 0.2832 | Correct: 50 | Time: 0.17 sec
Epoch 440 | Loss: 0.2572 | Correct: 50 | Time: 0.21 sec
Epoch 450 | Loss: 0.2685 | Correct: 50 | Time: 0.13 sec
Epoch 460 | Loss: 0.2454 | Correct: 50 | Time: 0.12 sec
Epoch 470 | Loss: 0.2511 | Correct: 50 | Time: 0.12 sec
Epoch 480 | Loss: 0.2345 | Correct: 50 | Time: 0.12 sec
Epoch 490 | Loss: 0.2494 | Correct: 50 | Time: 0.12 sec
Epoch 500 | Loss: 0.2434 | Correct: 50 | Time: 0.12 sec
```

### CPU Hidden 200

Split dataset

```bash
python run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET split --RATE 0.05
Epoch 10 | Loss: 24.1994 | Correct: 35 | Time: 1.89 sec
Epoch 20 | Loss: 20.0673 | Correct: 43 | Time: 0.38 sec
Epoch 30 | Loss: 15.8197 | Correct: 41 | Time: 0.26 sec
Epoch 40 | Loss: 12.9544 | Correct: 40 | Time: 0.26 sec
Epoch 50 | Loss: 12.8685 | Correct: 50 | Time: 0.27 sec
Epoch 60 | Loss: 10.5187 | Correct: 50 | Time: 0.42 sec
Epoch 70 | Loss: 8.7661 | Correct: 50 | Time: 0.26 sec
Epoch 80 | Loss: 8.2845 | Correct: 46 | Time: 0.26 sec
Epoch 90 | Loss: 7.6957 | Correct: 49 | Time: 0.26 sec
Epoch 100 | Loss: 7.4358 | Correct: 49 | Time: 0.30 sec
Epoch 110 | Loss: 7.8498 | Correct: 49 | Time: 0.38 sec
Epoch 120 | Loss: 6.7870 | Correct: 48 | Time: 0.26 sec
Epoch 130 | Loss: 5.7294 | Correct: 49 | Time: 0.27 sec
Epoch 140 | Loss: 5.3800 | Correct: 50 | Time: 0.26 sec
Epoch 150 | Loss: 6.1493 | Correct: 46 | Time: 0.42 sec
Epoch 160 | Loss: 5.0862 | Correct: 50 | Time: 0.26 sec
Epoch 170 | Loss: 6.1619 | Correct: 49 | Time: 0.26 sec
Epoch 180 | Loss: 4.5684 | Correct: 48 | Time: 0.26 sec
Epoch 190 | Loss: 3.7476 | Correct: 50 | Time: 0.32 sec
Epoch 200 | Loss: 4.2238 | Correct: 49 | Time: 0.37 sec
Epoch 210 | Loss: 4.4334 | Correct: 49 | Time: 0.26 sec
Epoch 220 | Loss: 4.2650 | Correct: 50 | Time: 0.26 sec
Epoch 230 | Loss: 3.5689 | Correct: 49 | Time: 0.26 sec
Epoch 240 | Loss: 3.5686 | Correct: 50 | Time: 0.42 sec
Epoch 250 | Loss: 4.1673 | Correct: 48 | Time: 0.26 sec
Epoch 260 | Loss: 3.4727 | Correct: 50 | Time: 0.26 sec
Epoch 270 | Loss: 3.3432 | Correct: 46 | Time: 0.26 sec
Epoch 280 | Loss: 2.8414 | Correct: 50 | Time: 0.32 sec
Epoch 290 | Loss: 2.5420 | Correct: 49 | Time: 0.37 sec
Epoch 300 | Loss: 3.2396 | Correct: 48 | Time: 0.35 sec
Epoch 310 | Loss: 3.2856 | Correct: 50 | Time: 0.34 sec
Epoch 320 | Loss: 3.0530 | Correct: 50 | Time: 0.36 sec
Epoch 330 | Loss: 2.9170 | Correct: 50 | Time: 0.33 sec
Epoch 340 | Loss: 2.8132 | Correct: 49 | Time: 0.26 sec
Epoch 350 | Loss: 2.9966 | Correct: 50 | Time: 0.26 sec
Epoch 360 | Loss: 3.3600 | Correct: 50 | Time: 0.26 sec
Epoch 370 | Loss: 2.3896 | Correct: 48 | Time: 0.42 sec
Epoch 380 | Loss: 2.8050 | Correct: 48 | Time: 0.26 sec
Epoch 390 | Loss: 2.4231 | Correct: 50 | Time: 0.26 sec
Epoch 400 | Loss: 2.3368 | Correct: 50 | Time: 0.26 sec
Epoch 410 | Loss: 6.0217 | Correct: 50 | Time: 0.37 sec
Epoch 420 | Loss: 2.1862 | Correct: 50 | Time: 0.32 sec
Epoch 430 | Loss: 2.2058 | Correct: 50 | Time: 0.26 sec
Epoch 440 | Loss: 2.1760 | Correct: 50 | Time: 0.26 sec
Epoch 450 | Loss: 2.4019 | Correct: 49 | Time: 0.26 sec
Epoch 460 | Loss: 2.9468 | Correct: 50 | Time: 0.42 sec
Epoch 470 | Loss: 1.9698 | Correct: 50 | Time: 0.26 sec
Epoch 480 | Loss: 1.9879 | Correct: 50 | Time: 0.26 sec
Epoch 490 | Loss: 1.7219 | Correct: 50 | Time: 0.27 sec
Epoch 500 | Loss: 2.2363 | Correct: 50 | Time: 0.39 sec
```

### GPU Hidden 100

T4 (colab) is used for this task.

#### Split

```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch 10 | Loss: 30.1928 | Correct: 40 | Time: 2.11 sec
Epoch 20 | Loss: 27.1207 | Correct: 41 | Time: 1.75 sec
Epoch 30 | Loss: 22.3531 | Correct: 43 | Time: 1.84 sec
Epoch 40 | Loss: 16.3042 | Correct: 49 | Time: 1.76 sec
Epoch 50 | Loss: 13.6659 | Correct: 48 | Time: 1.92 sec
Epoch 60 | Loss: 11.4513 | Correct: 48 | Time: 1.80 sec
Epoch 70 | Loss: 8.8212 | Correct: 48 | Time: 1.75 sec
Epoch 80 | Loss: 7.6644 | Correct: 49 | Time: 1.83 sec
Epoch 90 | Loss: 8.2058 | Correct: 48 | Time: 1.78 sec
Epoch 100 | Loss: 5.7483 | Correct: 49 | Time: 1.71 sec
Epoch 110 | Loss: 4.8286 | Correct: 49 | Time: 1.80 sec
Epoch 120 | Loss: 4.6781 | Correct: 49 | Time: 1.76 sec
Epoch 130 | Loss: 4.2799 | Correct: 49 | Time: 1.73 sec
Epoch 140 | Loss: 3.7783 | Correct: 48 | Time: 1.84 sec
Epoch 150 | Loss: 3.8407 | Correct: 49 | Time: 1.74 sec
Epoch 160 | Loss: 3.3631 | Correct: 49 | Time: 1.72 sec
Epoch 170 | Loss: 3.4508 | Correct: 50 | Time: 1.94 sec
Epoch 180 | Loss: 2.9571 | Correct: 49 | Time: 1.75 sec
Epoch 190 | Loss: 2.7262 | Correct: 49 | Time: 1.72 sec
Epoch 200 | Loss: 2.4790 | Correct: 49 | Time: 1.79 sec
Epoch 210 | Loss: 2.3714 | Correct: 50 | Time: 1.77 sec
Epoch 220 | Loss: 2.8776 | Correct: 49 | Time: 1.72 sec
Epoch 230 | Loss: 2.9465 | Correct: 49 | Time: 1.79 sec
Epoch 240 | Loss: 2.1923 | Correct: 50 | Time: 1.77 sec
Epoch 250 | Loss: 2.9157 | Correct: 50 | Time: 1.72 sec
Epoch 260 | Loss: 2.9066 | Correct: 49 | Time: 1.78 sec
Epoch 270 | Loss: 2.3109 | Correct: 49 | Time: 1.78 sec
Epoch 280 | Loss: 2.6698 | Correct: 49 | Time: 1.72 sec
Epoch 290 | Loss: 3.1939 | Correct: 49 | Time: 1.94 sec
Epoch 300 | Loss: 2.1042 | Correct: 49 | Time: 1.74 sec
Epoch 310 | Loss: 2.3558 | Correct: 49 | Time: 1.72 sec
Epoch 320 | Loss: 1.3060 | Correct: 49 | Time: 1.78 sec
Epoch 330 | Loss: 1.7049 | Correct: 49 | Time: 1.76 sec
Epoch 340 | Loss: 2.7171 | Correct: 50 | Time: 1.73 sec
Epoch 350 | Loss: 1.8982 | Correct: 49 | Time: 1.77 sec
Epoch 360 | Loss: 2.6878 | Correct: 49 | Time: 1.77 sec
Epoch 370 | Loss: 1.5462 | Correct: 49 | Time: 1.72 sec
Epoch 380 | Loss: 1.9661 | Correct: 49 | Time: 1.76 sec
Epoch 390 | Loss: 2.1700 | Correct: 50 | Time: 1.78 sec
Epoch 400 | Loss: 1.6307 | Correct: 49 | Time: 1.71 sec
Epoch 410 | Loss: 3.7232 | Correct: 50 | Time: 1.87 sec
Epoch 420 | Loss: 1.7536 | Correct: 50 | Time: 1.80 sec
Epoch 430 | Loss: 1.6631 | Correct: 49 | Time: 1.74 sec
Epoch 440 | Loss: 1.8169 | Correct: 49 | Time: 1.73 sec
Epoch 450 | Loss: 1.6273 | Correct: 50 | Time: 1.83 sec
Epoch 460 | Loss: 1.8059 | Correct: 49 | Time: 1.71 sec
Epoch 470 | Loss: 2.1243 | Correct: 50 | Time: 1.72 sec
Epoch 480 | Loss: 1.4871 | Correct: 49 | Time: 1.83 sec
Epoch 490 | Loss: 1.9283 | Correct: 50 | Time: 1.71 sec
Epoch 500 | Loss: 2.2122 | Correct: 50 | Time: 1.72 sec
```

#### Xor

```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch 10 | Loss: 26.1405 | Correct: 40 | Time: 2.21 sec
Epoch 20 | Loss: 23.1663 | Correct: 40 | Time: 1.77 sec
Epoch 30 | Loss: 19.6417 | Correct: 43 | Time: 1.86 sec
Epoch 40 | Loss: 17.1206 | Correct: 47 | Time: 1.75 sec
Epoch 50 | Loss: 16.2361 | Correct: 43 | Time: 1.75 sec
Epoch 60 | Loss: 13.3775 | Correct: 47 | Time: 1.86 sec
Epoch 70 | Loss: 12.8198 | Correct: 49 | Time: 1.75 sec
Epoch 80 | Loss: 11.4565 | Correct: 50 | Time: 1.74 sec
Epoch 90 | Loss: 9.3858 | Correct: 47 | Time: 1.88 sec
Epoch 100 | Loss: 9.0118 | Correct: 49 | Time: 1.74 sec
Epoch 110 | Loss: 8.6768 | Correct: 49 | Time: 1.74 sec
Epoch 120 | Loss: 7.3349 | Correct: 48 | Time: 1.87 sec
Epoch 130 | Loss: 7.7030 | Correct: 49 | Time: 1.74 sec
Epoch 140 | Loss: 7.1751 | Correct: 50 | Time: 1.75 sec
Epoch 150 | Loss: 7.0377 | Correct: 49 | Time: 1.87 sec
Epoch 160 | Loss: 7.5784 | Correct: 49 | Time: 1.74 sec
Epoch 170 | Loss: 6.1692 | Correct: 49 | Time: 1.76 sec
Epoch 180 | Loss: 6.6558 | Correct: 50 | Time: 1.85 sec
Epoch 190 | Loss: 5.5081 | Correct: 49 | Time: 1.77 sec
Epoch 200 | Loss: 5.2057 | Correct: 50 | Time: 1.80 sec
Epoch 210 | Loss: 5.7303 | Correct: 50 | Time: 1.81 sec
Epoch 220 | Loss: 4.3243 | Correct: 50 | Time: 1.74 sec
Epoch 230 | Loss: 4.4612 | Correct: 48 | Time: 1.82 sec
Epoch 240 | Loss: 4.4152 | Correct: 50 | Time: 1.79 sec
Epoch 250 | Loss: 4.5216 | Correct: 50 | Time: 1.75 sec
Epoch 260 | Loss: 4.4504 | Correct: 50 | Time: 1.85 sec
Epoch 270 | Loss: 4.8802 | Correct: 49 | Time: 1.75 sec
Epoch 280 | Loss: 3.4956 | Correct: 50 | Time: 1.71 sec
Epoch 290 | Loss: 3.0892 | Correct: 50 | Time: 1.81 sec
Epoch 300 | Loss: 4.1555 | Correct: 49 | Time: 1.73 sec
Epoch 310 | Loss: 2.8927 | Correct: 47 | Time: 1.72 sec
Epoch 320 | Loss: 3.1200 | Correct: 49 | Time: 1.82 sec
Epoch 330 | Loss: 3.5981 | Correct: 49 | Time: 1.74 sec
Epoch 340 | Loss: 3.1047 | Correct: 50 | Time: 1.72 sec
Epoch 350 | Loss: 3.4179 | Correct: 49 | Time: 1.78 sec
Epoch 360 | Loss: 2.8670 | Correct: 49 | Time: 1.77 sec
Epoch 370 | Loss: 2.8069 | Correct: 50 | Time: 1.72 sec
Epoch 380 | Loss: 2.8529 | Correct: 49 | Time: 1.78 sec
Epoch 390 | Loss: 2.8525 | Correct: 50 | Time: 1.77 sec
Epoch 400 | Loss: 2.5870 | Correct: 49 | Time: 1.71 sec
Epoch 410 | Loss: 2.7870 | Correct: 49 | Time: 1.79 sec
Epoch 420 | Loss: 2.5646 | Correct: 50 | Time: 1.76 sec
Epoch 430 | Loss: 2.4482 | Correct: 49 | Time: 1.72 sec
Epoch 440 | Loss: 2.4900 | Correct: 50 | Time: 1.76 sec
Epoch 450 | Loss: 2.8942 | Correct: 49 | Time: 1.78 sec
Epoch 460 | Loss: 2.1721 | Correct: 50 | Time: 1.72 sec
Epoch 470 | Loss: 2.8714 | Correct: 50 | Time: 1.76 sec
Epoch 480 | Loss: 2.5369 | Correct: 50 | Time: 1.79 sec
Epoch 490 | Loss: 2.1846 | Correct: 50 | Time: 1.71 sec
Epoch 500 | Loss: 2.8915 | Correct: 50 | Time: 1.75 sec
```

#### Simple

```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05

Epoch 10 | Loss: 15.6878 | Correct: 47 | Time: 1.91 sec
Epoch 20 | Loss: 10.2712 | Correct: 49 | Time: 1.71 sec
Epoch 30 | Loss: 9.0778 | Correct: 49 | Time: 1.80 sec
Epoch 40 | Loss: 9.0857 | Correct: 45 | Time: 1.71 sec
Epoch 50 | Loss: 6.2408 | Correct: 45 | Time: 1.71 sec
Epoch 60 | Loss: 5.7252 | Correct: 49 | Time: 1.82 sec
Epoch 70 | Loss: 6.0364 | Correct: 47 | Time: 1.70 sec
Epoch 80 | Loss: 4.5177 | Correct: 47 | Time: 1.69 sec
Epoch 90 | Loss: 4.4819 | Correct: 43 | Time: 1.82 sec
Epoch 100 | Loss: 4.2399 | Correct: 50 | Time: 1.70 sec
Epoch 110 | Loss: 4.7823 | Correct: 48 | Time: 1.70 sec
Epoch 120 | Loss: 4.9820 | Correct: 42 | Time: 1.80 sec
Epoch 130 | Loss: 3.7613 | Correct: 50 | Time: 1.71 sec
Epoch 140 | Loss: 3.7973 | Correct: 49 | Time: 1.71 sec
Epoch 150 | Loss: 3.2758 | Correct: 49 | Time: 1.77 sec
Epoch 160 | Loss: 3.7799 | Correct: 47 | Time: 1.77 sec
Epoch 170 | Loss: 2.8369 | Correct: 50 | Time: 1.70 sec
Epoch 180 | Loss: 3.6446 | Correct: 46 | Time: 1.74 sec
Epoch 190 | Loss: 3.5484 | Correct: 50 | Time: 1.79 sec
Epoch 200 | Loss: 2.5516 | Correct: 50 | Time: 1.71 sec
Epoch 210 | Loss: 4.2206 | Correct: 50 | Time: 1.71 sec
Epoch 220 | Loss: 2.2312 | Correct: 50 | Time: 1.83 sec
Epoch 230 | Loss: 1.4340 | Correct: 50 | Time: 1.72 sec
Epoch 240 | Loss: 1.5934 | Correct: 50 | Time: 1.71 sec
Epoch 250 | Loss: 2.4083 | Correct: 50 | Time: 1.84 sec
Epoch 260 | Loss: 2.7828 | Correct: 50 | Time: 1.71 sec
Epoch 270 | Loss: 1.5928 | Correct: 50 | Time: 1.72 sec
Epoch 280 | Loss: 1.9113 | Correct: 49 | Time: 1.82 sec
Epoch 290 | Loss: 2.5118 | Correct: 50 | Time: 1.70 sec
Epoch 300 | Loss: 1.5144 | Correct: 50 | Time: 1.71 sec
Epoch 310 | Loss: 1.2457 | Correct: 50 | Time: 1.82 sec
Epoch 320 | Loss: 1.4595 | Correct: 50 | Time: 1.71 sec
Epoch 330 | Loss: 1.2756 | Correct: 50 | Time: 1.70 sec
Epoch 340 | Loss: 1.6981 | Correct: 50 | Time: 1.82 sec
Epoch 350 | Loss: 1.8870 | Correct: 50 | Time: 1.69 sec
Epoch 360 | Loss: 3.1722 | Correct: 50 | Time: 1.70 sec
Epoch 370 | Loss: 1.0833 | Correct: 50 | Time: 1.78 sec
Epoch 380 | Loss: 1.1344 | Correct: 50 | Time: 1.75 sec
Epoch 390 | Loss: 0.9360 | Correct: 50 | Time: 1.71 sec
Epoch 400 | Loss: 0.7021 | Correct: 50 | Time: 1.74 sec
Epoch 410 | Loss: 0.7742 | Correct: 50 | Time: 1.78 sec
Epoch 420 | Loss: 1.3345 | Correct: 50 | Time: 1.70 sec
Epoch 430 | Loss: 0.8926 | Correct: 50 | Time: 1.71 sec
Epoch 440 | Loss: 0.9616 | Correct: 50 | Time: 1.82 sec
Epoch 450 | Loss: 0.7540 | Correct: 50 | Time: 1.71 sec
Epoch 460 | Loss: 0.6519 | Correct: 50 | Time: 1.70 sec
Epoch 470 | Loss: 0.6030 | Correct: 50 | Time: 1.83 sec
Epoch 480 | Loss: 0.6455 | Correct: 50 | Time: 1.70 sec
Epoch 490 | Loss: 0.4826 | Correct: 50 | Time: 1.69 sec
Epoch 500 | Loss: 0.5692 | Correct: 50 | Time: 1.82 sec
```

### GPU Hidden 200

```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET split --RATE 0.05
Epoch 10 | Loss: 30.8957 | Correct: 28 | Time: 2.00 sec
Epoch 20 | Loss: 26.8407 | Correct: 40 | Time: 1.89 sec
Epoch 30 | Loss: 16.2436 | Correct: 47 | Time: 1.79 sec
Epoch 40 | Loss: 18.4552 | Correct: 45 | Time: 1.92 sec
Epoch 50 | Loss: 19.8446 | Correct: 45 | Time: 1.80 sec
Epoch 60 | Loss: 10.1113 | Correct: 49 | Time: 1.80 sec
Epoch 70 | Loss: 9.1220 | Correct: 48 | Time: 1.91 sec
Epoch 80 | Loss: 9.3210 | Correct: 45 | Time: 1.79 sec
Epoch 90 | Loss: 11.0167 | Correct: 46 | Time: 1.92 sec
Epoch 100 | Loss: 6.5397 | Correct: 46 | Time: 1.79 sec
Epoch 110 | Loss: 5.9404 | Correct: 50 | Time: 1.80 sec
Epoch 120 | Loss: 5.0966 | Correct: 49 | Time: 1.90 sec
Epoch 130 | Loss: 6.5283 | Correct: 50 | Time: 1.79 sec
Epoch 140 | Loss: 14.0586 | Correct: 46 | Time: 1.89 sec
Epoch 150 | Loss: 5.8544 | Correct: 50 | Time: 1.82 sec
Epoch 160 | Loss: 5.5700 | Correct: 48 | Time: 1.80 sec
Epoch 170 | Loss: 6.3776 | Correct: 50 | Time: 1.90 sec
Epoch 180 | Loss: 5.5956 | Correct: 48 | Time: 1.80 sec
Epoch 190 | Loss: 16.5181 | Correct: 48 | Time: 1.84 sec
Epoch 200 | Loss: 7.2125 | Correct: 48 | Time: 1.85 sec
Epoch 210 | Loss: 5.4328 | Correct: 46 | Time: 1.79 sec
Epoch 220 | Loss: 5.0422 | Correct: 49 | Time: 1.90 sec
Epoch 230 | Loss: 5.8281 | Correct: 46 | Time: 1.81 sec
Epoch 240 | Loss: 3.3254 | Correct: 50 | Time: 1.82 sec
Epoch 250 | Loss: 3.2292 | Correct: 48 | Time: 1.88 sec
Epoch 260 | Loss: 5.4160 | Correct: 46 | Time: 1.79 sec
Epoch 270 | Loss: 4.8069 | Correct: 50 | Time: 1.92 sec
Epoch 280 | Loss: 4.7437 | Correct: 49 | Time: 1.78 sec
Epoch 290 | Loss: 5.7562 | Correct: 49 | Time: 1.79 sec
Epoch 300 | Loss: 4.2427 | Correct: 50 | Time: 1.91 sec
Epoch 310 | Loss: 5.1272 | Correct: 50 | Time: 1.79 sec
Epoch 320 | Loss: 5.0257 | Correct: 48 | Time: 1.91 sec
Epoch 330 | Loss: 3.2155 | Correct: 48 | Time: 1.78 sec
Epoch 340 | Loss: 4.3389 | Correct: 49 | Time: 1.79 sec
Epoch 350 | Loss: 3.0518 | Correct: 50 | Time: 1.90 sec
Epoch 360 | Loss: 28.0202 | Correct: 50 | Time: 1.80 sec
Epoch 370 | Loss: 2.2172 | Correct: 50 | Time: 1.87 sec
Epoch 380 | Loss: 2.8317 | Correct: 50 | Time: 1.81 sec
Epoch 390 | Loss: 5.3835 | Correct: 49 | Time: 1.79 sec
Epoch 400 | Loss: 6.7927 | Correct: 45 | Time: 1.92 sec
Epoch 410 | Loss: 2.6798 | Correct: 46 | Time: 1.80 sec
Epoch 420 | Loss: 4.0187 | Correct: 50 | Time: 1.84 sec
Epoch 430 | Loss: 2.6103 | Correct: 50 | Time: 1.87 sec
Epoch 440 | Loss: 2.3851 | Correct: 50 | Time: 1.79 sec
Epoch 450 | Loss: 4.0618 | Correct: 46 | Time: 1.92 sec
Epoch 460 | Loss: 2.8539 | Correct: 48 | Time: 1.81 sec
Epoch 470 | Loss: 2.5741 | Correct: 48 | Time: 1.84 sec
Epoch 480 | Loss: 3.4684 | Correct: 50 | Time: 1.91 sec
Epoch 490 | Loss: 4.1346 | Correct: 49 | Time: 1.80 sec
Epoch 500 | Loss: 1.6641 | Correct: 50 | Time: 1.92 sec
```
