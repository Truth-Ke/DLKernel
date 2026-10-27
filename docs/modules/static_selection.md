# 静态选择与 TuneMode(TuneMode 子系统)

> 阅读对象:需要控制 GEMM 冷启动行为——"什么时候允许在线实测、什么时候走解析
> 兜底"——的框架作者与算子维护者。
> 范围:`TuneMode` 三模式、固定契约 F 与 winner key W 的双身份、解析兜底
> (`gemm_heuristic`)、离线工具(`static_tune`)。进程内调优协议本体见
> [autotuning_subsystem.md](autotuning_subsystem.md);本文不重复其内容。
> 实施依据与全部历史记录:`gemm_static_tune_master.md`(仓库根,卷一为 v4 计划)。

---

## 1. 问题与契约

调优解决"哪个 GemmConfig 最快";TuneMode 解决"**什么时候允许花 GPU 去测**"。

三模式(`DLKernel/gemm_config.py:35`,`IntEnum`,过 torch.library schema 为
plain int,同 `SplitKMode` 先例)。**默认是 `TUNED`——TuneMode 之前的原行为,
miss 即实测**(2026-10-21 用户决定);"一个 miss 不必然触发 sweep"的稳态契约由
框架在调用点显式传 `RUNNING` 获得,不是库的默认:

| 模式 | W 命中实测冠军 | W miss(未被 REQUIRE 拒绝) |
|---|---|---|
| `TUNED = 0`(默认) | 用冠军 | 实测当前 W |
| `STATIC = 1` | 用冠军 | 复用本 F 已测冠军;F 从未测则解析兜底,绝不测量 |
| `RUNNING = 2`(显式 opt-in) | 用冠军,不消耗额度 | F 未测则实测一次;F 已测则**复用该 F 的已测冠军**(跨 token 桶迁移,autotuner `_contract_winners`) |

规则(`DLKernel/autotuner.py` miss 分支):

- 实测冠军跨模式共享;只有**本进程成功完成的 sweep** 消耗 RUNNING 的额度
  (磁盘命中、REQUIRE 拒绝、失败、空调用均不消耗)。
- `DLKERNEL_REQUIRE_TUNE_CACHE=1` 对"无冠军的自动选择调用"仍报错——解析兜底
  不满足断言。
- `DLKERNEL_FORCE_CACHE_UPDATE` 使磁盘条目不被采用,但**不**把 STATIC/已测
  RUNNING 变成 sweep,也不重置 F 记录。
- 2026-10-22 设计:F 已测时的新 W miss **复用该 F 的已测冠军**(显式 config 执行,
  冠军语义:plan/recipe 可缓存;不落入本 W 的内存表,他进程发布的本 W 冠军下一
  调用仍可升级;磁盘启用时冠军经 `contract_winners/` 文件跨进程迁移)。`-1` 兜底
  仅剩"F 从未测过"的纯冷场景:不产生冠军、不写任何缓存、miss 不记负结果。
- 未接 `tune_mode` 的算子(rmsnorm 等)行为不变:miss 照旧 sweep。

## 2. 双身份:F 与 W(`DLKernel/varlen_selector.py`)

- **W(winner key)**:既有 bucket 塌缩 key(`gemm_token_bucket_v5`)——识别
  "哪次测量"。
- **F(固定契约,`gemm_contract_key`/`gemm_contract_v1`)**:同一遍元数据扫描,
  声明的 token 维折叠为常量标记、剔除 avg-expert 桶/selector leaf/bench 预算
  ——识别"哪个算子契约"(权重几何、专家数、dtype、布局、epilogue 身份)。
  RUNNING 的"一次"按 F 计:Qwen 与 DeepSeek 的不同 expert 几何各自拥有首测;
  token 总量变化不换 F。
- `tune_mode` 被两条 key 收集路径显式排除(`_NON_IDENTITY_NAMES`):模式是
  miss 行为,不是 workload 身份。
- dense 无声明动态维:F = 完整默认 key。

## 3. 解析兜底(`DLKernel/gemm_heuristic.py`)

- 接口:`rank(desc, candidates) -> GemmConfig | None`。只排序不发明;候选来自
  算子自己的 prepare 阶段(含 varlen-K dynamic twins);SM 数在调用边界解析
  (`DLKERNEL_SM_COUNT` → `get_num_sms`(含 `set_num_sms` cap)→ 设备 → per-arch
  常数)后显式进 desc;不查 CUDA、不留 tensor、不读值。
- 接线点:`gemm_interface` 的 `config=None` 分支(SM90 先 rank,未覆盖落回
  nvMMH → default_config 链);epi wrapper 的 `config=None` 分支在访问
  config/分配 scratch 之前整体委托 `mod.gemm`;epi `__call__` 的非 tuner
  链(`DLKernel/epilogue/frontend.py:1685`)解析 config 时同样先问
  `gemm_heuristic.rank`,未覆盖落 default。
- 代价模型(v4,2026-10-21 冻结,1584-case 离线研究,逐步消融每项实测):
  `T = L1周期/wave_eff`,L1 取 DeepGEMM sm90.hpp:222-230 的完整三项式
  (l1_ab + l1_tc + C/D 双份),wave 分母为 **SM 数(CTA 计数)**;L2 多播摊薄
  项实测为死代码(47036 对全败)已删;varlen_k 对 dynamic-persistent 候选
  `wave_eff' = min(1, +0.1)`——**开关非拟合常数**(0 是断崖,0.1/0.2/0.5
  等价);tile 按**二维输出 tile 单元**(post-swap (tm,tn),三段 avg 阈值,
  四折一致)剪枝:m 族 (128,8192)→(128,256)/(256,128)/(192,128),k 族
  (170,2048)→(128,256)/(128,128)/(256,128);单元空回退全池;gather_A
  (A_idx)不覆盖。holdout:varlen_m top-5 **80.9%**(≤1.05 78.1%,中位
  1.010),varlen_k **59.5%**(≤1.05 75.8%,中位 1.021)。被否形态与消融
  阶梯归档于 `gemm_static_tune_master.md` 卷二第五部分与研究区。
- **dense 不由本 ranker 覆盖**:dense 的 miss 直接走既有 nvMMH/default 链
  (warn-once 照发)。
- 结果不进任何 winner 表或 JSON。

三条链刻意不同:低层 `mod.gemm` 本身没有 config 参数,epi wrapper 的 `-1`
委托只能在 `mod.gemm` 内部解析 config,接不上 ranker;前两条高层链持有
config 入口,能够且确实接入。

## 4. 与执行缓存的关系

自动选择**先于**依赖 config 的执行 plan 查询发生;`_gemm_iface_plan_cache` 的
key 含 `tune_mode`(`gemm_interface.py:1045`)——dense
下 STATIC 先行不会把兜底 config 钉死给后续 TUNED 调用。config 确定后,原有
执行缓存照常复用。

## 5. 离线工具(`python -m DLKernel.static_tune`)

| 子命令 | 语义 | 产物 |
|---|---|---|
| `study` | TUNED + `DLKERNEL_VARLEN_SELECTOR=0` sweep prepared 全集;每 raw case 独立 `DLKERNEL_CACHE_DIR`,artifact 目录共享 | timings 数据集(研究用;varlen 条目 key 含 fallback leaf,**不**用于部署) |
| `production` | TUNED + 部署语义 selector/预算 | 可部署 measured 冠军 |
| `prefill` | GEMM 族(dense/varlen_m/varlen_k)经 rank 解析合法 config(不测量);算子自有族(`fused_up`/`fused_down`/`epi`)走算子自身 STATIC 默认链,ranker 不覆盖 | 声明覆盖的 `.o`(GPU-blind 需 `DLKERNEL_ARCH` + `DLKERNEL_SM_COUNT`) |

## 6. 观测

- `DLKERNEL_DEBUG_TUNE=1` 新增一行 `AUTOTUNE_FALLBACK_HEURISTIC key=<digest>`;
  其余沿用既有 `TUNE_START/DISK_HIT/DISK_MISS`。
- 每个 (fn, key) 首次兜底打一条 `RuntimeWarning`(FlashInfer perf-cliff 同款),
  此后静默。

## 7. 测试索引

- 身份:`tests/test_autotuner.py::test_gemm_contract_*`、
  `test_gemm_tune_and_contract_keys_exclude_tune_mode`。
- 链语义:`tests/test_autotuner.py::test_static_miss_*`、
  `test_measured_winners_are_shared_across_modes`、`test_running_tunes_once_per_contract`、
  `test_tuned_still_benches_new_w_and_shares_the_record`、
  `test_missing_tune_mode_keeps_legacy_bench`。
- 兜底:`tests/test_gemm_heuristic.py`(CPU-only)。
- 离线 case 机制:`tests/test_static_tune.py`(CPU-only;meta 构建器、slug、
  prefill 的算子自有族 STATIC 解析)。
- 编译边界:`tests/test_gemm_functional.py::test_gemm_fullgraph_modes_match_eager`、
  `test_fake_trace_never_tunes_or_consumes_running_quota`。
