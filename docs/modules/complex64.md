# Complex64 元素类型设计

`DLKernel/complex.py` 提供原生的 `complex64` 元素类型。`complex64` 面向
CuTe-DSL kernel，不改动 C++ MLIR 绑定。tvm-ffi 边界约定：kernel ABI
按 f64 存储。调用处用 `complex_storage(t)` 把 `torch.complex64` 张量转成
`torch.float64` 视图。

`tests/test_complex.py` 的 44 个用例验证该类型。覆盖构造分支、算术矩阵、
方法、类不变量与辅助函数，也覆盖 `__c_pointers__` 边界、特殊值、自动提升和
smem 往返。

本文记录设计取舍、已验证边界和使用限制。

## 根本约束

`cute.MemRefType.get(value_type, layout_type)` 是 C++ MLIR 绑定调用。
`make_tensor`、`make_rmem_tensor`、`recast_tensor` 和 smem 的
`allocate_tensor` 都经过它。它对元素类型实行白名单，只接受四种：`int`、
`float`、`ptr`、`sparse_elem`。直接选 `complex<f32>` 会得到：

```
TypeError: expects value type to be int, float, ptr or sparse_elem,
           but got 'complex<f32>'
```

检查在 `_cute_ir`（编译后的绑定）里执行。Python 层无法绕过。

白名单同样拒绝其他"独占"的 MLIR 类型候选：

| MLIR 元素类型 | 在白名单内 | 是否 Complex64 独占 |
| --- | :---: | --- |
| `complex<f32>` | 否 | 独占，但被拒 |
| `tuple<f32, f32>` | 否 | 独占，但被拒 |
| `f64` | 是 | 与 `Float64` 共享 |
| `i64` | 是 | 与 `Int64` / `Uint64` 共享 |
| `vector<2xf32>` 等 | 否（同族全拒） | 独占，但被拒 |

可行的存储类型只剩其他 Python `Numeric` 已占用的几种。本实现选 `f64`。

## Python 类型身份的恢复

存储类型是 `f64`。kernel 作者要求 `tensor[i]` 返回 `Complex64`，不是
`Float64`。JIT 侧 `_Tensor` 经 `Numeric.from_mlir_type` 从 MLIR memref 推导
`element_type`。该查找是多对一：f64 永远映射回 `Float64`。构造时声明的
Python 类型身份必须在往返后保留。

两条构造路径，结果不同：

1. `make_fake_tensor` / `make_rmem_tensor` 显式传入 dtype。运行时 tensor 把
   dtype 存进 `_dtype` 属性。只要 `Complex64.mlir_type == T.f64()`，MLIR
   memref 是 f64，`_dtype` 保持 `Complex64`。
2. `SmemAllocator.allocate_tensor`、`cute.recast_tensor` 和指针转 tensor 的
   `make_tensor` 从 MLIR 值构造 JIT tensor，不带 dtype。`_dtype` 默认取
   `Numeric.from_mlir_type(f64)` = `Float64`，此时需要重新打标。

修复是一行包装：`allocate_smem_complex(...)` 与 `recast_to_complex64(...)`。
两者内部执行 `t._dtype = Complex64; return t`。

不打标的后果如下。`smem[i] = c`（`c: Complex64`）这类标量写入经
`_cvt_to_dest`（`cute/tensor.py`）调用 `data.to(self.element_type)`，即
`Complex64.to(Float64)`。该转换把 f64 打包位当实数做 cvtf 式转换，截断到
f32，然后存储。输出静默损坏：高 32 位清零，低 32 位存位打包值的 f32 截断。

这是包装类唯一抹不平的边界问题，恰好出现在 MLIR 边界上。

## 为什么继承 Float32

`Complex64` 的算术运算符拆出两个 Float32 lane，计算后重新打包。
`Complex64` 在左侧时简单：Python 先调用 `Complex64.__mul__(other)`。

`Float32 * Complex64` 时，Python 先调用 `Float32.__mul__(complex)`。三种继承
选择的后果：

| 继承自 | mlir_type | 后果 |
| --- | --- | --- |
| `Int64` | i64 | kind 不匹配，存储时报 `type mismatch, store f64 to CxI64`。报错响亮，数据不损坏 |
| `Float64` | f64 | kind 与宽度都相同。提升选 Complex64 作结果类型，`op(lhs.value, rhs.value)` 对两个 f64 SSA 执行 `arith.mulf`。存储位是合法 f64，但把打包位模式当实数解释。静默损坏 |
| `Float32` | f64（width=64 覆盖） | `Complex64` 是 `Float32` 的严格子类。Python 子类优先规则让 `Complex64.__rmul__(Float32)` 先于 `Float32.__mul__(Complex64)` 触发。分发权在本实现手里 |

三者中只有 `Float32` 继承让混合算术两个方向都正确，且不需要 monkey-patch。
结构代价：`isinstance(complex_val, Float32) is True`。cutlass 里只有一个位置
检查它（`nvvm_wrappers.py` 的 warp-reduce fmax 辅助），实际无碍。

## 同类型初始化必须先短路

`_cvt_to_dest` 对每次 tensor 写入调用 `data.to(element_type)`。两侧都是
`Complex64` 时，它变成 `Complex64(complex_instance)`。`__init__` 若落进泛型
`Numeric` 分支（先 `Float32(x)` 再重打包），就把已有打包位当实数 Float 处理：
取位模式的 f32 截断，按 `(real_view, 0)` 打包。

修复是 `__init__` 顶部的一次类型检查：

```python
if isinstance(x, Complex64):
    Numeric.__init__(self, x.value)
    return
```

## 已验证内容

`tests/test_complex.py`（44 个用例）覆盖：

- 构造全部分支：Python `complex`、双参 `(re, im)`、同类型拷贝、f64/f32 的
  `ir.Value`、`Float32` 实例、其他 `Numeric`（Int32、Float64）、Python
  int/float、非法输入的 `TypeError`。
- 算术矩阵：`Complex64 OP X`，`X` 取 `Complex64`、`Float32`、`int`、
  `float`，两个方向；`__neg__`；对 Python complex 参考的随机正确性扫描。
- 方法：`conj()`（对合，且 `c * c.conj()` 为实数）、`real()`、`imag()`、
  `from_re_im` 别名。
- 类不变量：`width=64`、`is_float=True`、`mlir_type == T.f64()`、
  `isinstance(c, Float32) is True`、`is_same_kind(Float32 / Float64) is True`。
- 辅助函数：`complex_storage(t)`（f64 直通、complex64 转 view、其他 dtype 抛
  `TypeError`）、`recast_to_complex64`、`allocate_smem_complex`。
- `__c_pointers__`：静态值路径可用，动态 SSA 路径抛错。
- 边缘值经位转换管线精确往返：0、-0.0、±inf、NaN、denormal、极大模、混合符号。
- `_cvt_to_dest` 自动提升：把 `Float32` 或 `Float64` 写入 `Complex64` tensor
  落为 `(value, 0)`。该行为已用用例钉死，防止 `Numeric` 提升规则变化破坏它。
- smem 往返（gmem → rmem → smem → sync → smem → rmem → gmem），使用
  `allocate_smem_complex`。该用例回归 smem 分配边界的 dtype 丢失问题。

## 未验证内容

- TMA（SM90 批量加载 / cp.async.bulk）携带 Complex64 元素。
- copy atom 类型为 Complex64 的 `cute.copy`。
- Tensor-core（MMA）操作。
- 类型化 Complex64 smem 布局的收益。`allocate_smem_complex` 可用，标量 f32
  布局与它孰优未测。
- `Float32` 为 tensor（非标量）时的混合算术。例如 Complex64 寄存器乘另一个
  rmem tensor 的 Float32 lane。`__rmul__` 路径应当覆盖，未直接测试。

## 使用注意

1. **MLIR 边界丢失 dtype。** 凡从 MLIR 值构造 JIT tensor 的路径（smem 分配、
   recast、指针转 tensor）都丢 Complex64 标记。使用 `DLKernel/complex.py` 的
   包装，或手动调用 `_retag_as_complex64(t)`。
2. `isinstance(c, Float32) is True`。cutlass 只有一个位置检查它
   （`nvvm_wrappers.py` 的 fmax 辅助）。行为可疑时先 grep 这个位置。
3. **`is_same_kind(Complex64, Float32) is True`**（两者 `is_float=True`）。
   `tensor[i] = some_float32` 经 `_cvt_to_dest` 自动提升为
   `Complex64(float32_instance)`，落为 `(re=value, im=0)`。标量实数写入因此
   方便；想存"复数位"而误存 f32，自动提升会静默包装它。要加固特定 kernel，
   在 `__setitem__` 加显式类型检查。
4. **`Float32 OP Complex64` 依赖反射算符优先。** 前提是 `Complex64` 保持
   `Float32` 的严格子类。不要破坏该继承。
5. **`Numeric.from_mlir_type` 是单一全局注册表。** 为 Complex64 到 Float64 的
   映射做 monkey-patch 时，不得扰动既有 Float64 映射。当前实现不打补丁，按
   tensor 重打标。
6. **动态 Complex64 SSA 值不能作 kernel 参数。** `self.value` 是 SSA 时
   `__c_pointers__` 抛错。静态（Python complex 字面量）值可用。这与
   `Float32` 的行为一致。

## 文件布局

```
DLKernel/complex.py
  class Complex64(Float32, width=64, mlir_type=T.f64)
    __init__(x, im=None)        # complex / Float32 / Numeric / ir.Value(f32|f64)
    _pack_ssa(re, im)           # static: two f32 SSAs -> packed f64 SSA
    _unpack(self)               # f64 -> (Float32, Float32)
    from_re_im(re, im)          # public alias of _pack_ssa-based ctor
    real() / imag() / conj()
    __add__/__radd__/__sub__/__rsub__/__mul__/__rmul__/__neg__
    __c_pointers__              # 8-byte scalar-arg path

  allocate_smem_complex(allocator, layout)
  recast_to_complex64(src)
  complex_storage(t)            # torch.complex64 -> torch.float64 view

  _register_with_tvm_ffi()      # called at import time
```

```
tests/test_complex.py         # 44 tests for Complex64 the type
docs/modules/complex64.md  # this file (design rationale + sharp edges)
```
