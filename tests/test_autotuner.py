import inspect

import pytest

from DLKernel.autotuner import AutotuneConfig, TunePolicy


@pytest.fixture(autouse=True)
def _cpu_arch_for_metadata_key(monkeypatch):
    """Give metadata-only key tests a deterministic SM90 capability on CPU."""
    monkeypatch.setenv("DLKERNEL_ARCH", "90")


def _select_via_policy(base, call):
    """The tuner's prepare -> shortlist composition for varlen candidate tests."""
    from DLKernel.varlen_selector import VarlenGemmTunePolicy

    policy = VarlenGemmTunePolicy()
    prepared = policy.prepare_candidates(base, call, {})
    return policy.shortlist(prepared, call, {})


def _bucket_key(args, kwargs=None):
    """Varlen tune key, with the ``gemm_tuned`` signature supplied for tests.

    The key is built from positional arguments, so the policy needs to know the
    parameter names; in production ``Autotuner`` passes its own ``arg_names``.
    """
    from DLKernel.gemm_interface import gemm_tuned
    from DLKernel.varlen_selector import gemm_tune_key

    return gemm_tune_key(args, kwargs or {}, gemm_tuned.arg_names)


def test_autotune_config_supports_multi_kwarg_hash_and_equality():
    config_a = AutotuneConfig(block_m=128, num_warps=4)
    config_b = AutotuneConfig(block_m=128, num_warps=4)
    config_c = AutotuneConfig(block_m=64, num_warps=4)

    assert config_a == config_b
    assert hash(config_a) == hash(config_b)
    assert config_a != config_c

    timings = {config_a: 1.25, config_c: 2.5}
    assert timings[config_b] == 1.25
    assert len({config_a, config_b, config_c}) == 2


def test_gemm_token_bucket_buckets_gather_indices_with_rows():
    import torch

    # MetaMoE EP8 routed-row sizes observed in training.  258012 is in the
    # 262144 bucket while 512303 is deliberately in the next bucket.
    common = dict(
        A=torch.empty((4096, 2048)),
        B=torch.empty((2048, 768)),
        cu_seqlens_m=torch.zeros((769,), dtype=torch.int32),
    )
    key_a = _bucket_key(
        (
            common["A"],
            common["B"],
            torch.empty((258012, 768)),
            None,
            None,
            1.0,
            1.0,
            common["cu_seqlens_m"],
            None,
            torch.empty((258012,), dtype=torch.int32),
        ),
        {},
    )
    key_b = _bucket_key(
        (
            common["A"],
            common["B"],
            torch.empty((258013, 768)),
            None,
            None,
            1.0,
            1.0,
            common["cu_seqlens_m"],
            None,
            torch.empty((258013,), dtype=torch.int32),
        ),
        {},
    )
    assert key_a == key_b

    key_a2 = _bucket_key(
        (
            common["A"],
            common["B"],
            torch.empty((300000, 768)),
            None,
            None,
            1.0,
            1.0,
            common["cu_seqlens_m"],
            None,
            torch.empty((300000,), dtype=torch.int32),
        ),
        {},
    )
    assert key_a2 == key_a

    key_c = _bucket_key(
        (
            common["A"],
            common["B"],
            torch.empty((512303, 768)),
            None,
            None,
            1.0,
            1.0,
            common["cu_seqlens_m"],
            None,
            torch.empty((512303,), dtype=torch.int32),
        ),
        {},
    )
    assert key_c != key_a


def test_gemm_token_bucket_ep8_gather_rows_do_not_collide():
    import torch

    # With eight local experts, the routed row count is much larger than the
    # source table A. The key must use A_idx length for average-M bucketing.
    cu = torch.zeros((9,), dtype=torch.int32)
    source_a = torch.empty((4096, 2048))
    b = torch.empty((8, 2048, 768))

    def key_for(total_m):
        return _bucket_key(
            (
                source_a,
                b,
                torch.empty((total_m, 768)),
                None,
                None,
                1.0,
                1.0,
                cu,
                None,
                torch.empty((total_m,), dtype=torch.int32),
            ),
            {},
        )

    assert key_for(258012) != key_for(512303)


def test_gemm_token_bucket_preserves_static_n_for_varlen_k():
    import torch

    # For varlen-K, only A's K dimension and B's K dimension are dynamic.
    # out's N dimension is static and must remain exact in the key.
    a = torch.empty((2048, 258012))
    b = torch.empty((258012, 768))
    out_768 = torch.empty((2048, 768))
    out_1024 = torch.empty((2048, 1024))
    cu = torch.tensor([0, 258012], dtype=torch.int32)
    args = (a, b, out_768, None, None, 1.0, 1.0, None, cu, None)
    key_768 = _bucket_key(args, {})
    key_1024 = _bucket_key((a, b, out_1024, None, None, 1.0, 1.0, None, cu, None), {})
    assert key_768 != key_1024


def test_gemm_token_bucket_keeps_quantized_output_scale_shape():
    import torch

    # Quantized-output scale factors are named tensors, not operands. Two
    # different output-scale layouts must not share one tuning result.
    cu = torch.zeros((9,), dtype=torch.int32)
    base = (
        torch.empty((258012, 2048)),
        torch.empty((768, 2048, 1536)),
        torch.empty((258012, 1536)),
        None,
        None,
        1.0,
        1.0,
        cu,
        None,
        None,
    )
    key_a = _bucket_key(base, {"SFD": torch.empty((128, 4, 32, 4, 4))})
    key_b = _bucket_key(base, {"SFD": torch.empty((128, 8, 32, 4, 4))})
    assert key_a is not None
    assert key_a != key_b


def test_gemm_token_bucket_keeps_quantized_output_scale_shape_for_positional_tail():
    import torch

    # cu_seqlens_m passed positionally: the hook has to name the positional
    # arguments itself to recognise a varlen call at all, and the positional
    # tail (SFD at index 23) still has to reach the key.
    cu = torch.zeros((9,), dtype=torch.int32)
    prefix = (
        torch.empty((258012, 2048)),
        torch.empty((768, 2048, 1536)),
        torch.empty((258012, 1536)),
        None,
        None,
        1.0,
        1.0,
        cu,
        None,
        None,
        None,
        False,
        False,
        None,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        1,
        0,
    )
    assert len(prefix) == 23
    key_a = _bucket_key(prefix + (torch.empty((128, 4, 32, 4, 4)), None, None), {})
    key_b = _bucket_key(prefix + (torch.empty((128, 8, 32, 4, 4)), None, None), {})
    assert key_a is not None
    assert key_a != key_b


def test_gemm_token_bucket_keeps_average_expert_rows_bucket():
    import torch

    # Both totals are in the 262144 total-row bucket, but their average rows
    # per expert cross an average-M bucket and should not share tuning.
    cu = torch.empty((769,), dtype=torch.int32)
    common = (None, None, 1.0, 1.0, cu, None, None)
    key_small = _bucket_key(
        (
            torch.empty((100000, 2048)),
            torch.empty((768, 2048, 1536)),
            torch.empty((100000, 1536)),
            *common,
        ),
        {},
    )
    key_large = _bucket_key(
        (
            torch.empty((200000, 2048)),
            torch.empty((768, 2048, 1536)),
            torch.empty((200000, 1536)),
            *common,
        ),
        {},
    )
    assert key_small != key_large


def test_gemm_token_bucket_separates_same_bucket_across_selector_leaves():
    import torch

    # Both totals round to the same average-rows bucket (64 experts), but
    # K/N=78.125 and K/N=79.1015625 land on opposite K-group leaves.
    def key_for(total_k):
        a = torch.empty((2048, total_k), dtype=torch.bfloat16)
        b = torch.empty((total_k, 2048), dtype=torch.bfloat16)
        out = torch.empty((2048, 2048), dtype=torch.bfloat16)
        return _bucket_key(
            (a, b, out, None, None, 1.0, 1.0, None, torch.zeros((65,), dtype=torch.int32), None),
            {},
        )

    assert key_for(160000) != key_for(162000)


def test_gemm_token_key_uses_capability_not_device_ordinal(monkeypatch):
    """Equivalent CUDA ordinals share a winner; architecture still isolates it."""
    import torch

    import DLKernel.varlen_selector as key_module

    a = torch.empty((4096, 2048), dtype=torch.bfloat16)
    b = torch.empty((2048, 1536), dtype=torch.bfloat16)
    out = torch.empty((4096, 1536), dtype=torch.bfloat16)
    cu = torch.empty((65,), dtype=torch.int32)
    monkeypatch.setattr(key_module, "_device_capability", lambda tensor: (9, 0))
    sm90 = _bucket_key((a, b, out, None, None, 1.0, 1.0, None, cu, None), {})
    assert ("device_capability", (9, 0)) in sm90
    assert not any(entry[0] == "device" for entry in sm90 if isinstance(entry, tuple))

    monkeypatch.setattr(key_module, "_device_capability", lambda tensor: (10, 0))
    sm100 = _bucket_key((a, b, out, None, None, 1.0, 1.0, None, cu, None), {})
    assert sm100 != sm90


def test_gemm_token_key_tracks_selector_switch_and_bench_budget(monkeypatch):
    import torch

    def key_for():
        a = torch.empty((2048, 160000), dtype=torch.bfloat16)
        b = torch.empty((160000, 2048), dtype=torch.bfloat16)
        out = torch.empty((2048, 2048), dtype=torch.bfloat16)
        cu = torch.zeros((65,), dtype=torch.int32)
        return _bucket_key((a, b, out, None, None, 1.0, 1.0, None, cu, None), {})

    enabled = key_for()
    monkeypatch.setenv("DLKERNEL_VARLEN_SELECTOR", "0")
    disabled = key_for()
    assert enabled != disabled

    monkeypatch.setenv("DLKERNEL_VARLEN_SELECTOR", "1")
    monkeypatch.setenv("DLKERNEL_TUNE_TIMED_CALLS", "64")
    budget = key_for()
    assert budget != enabled


def test_gemm_token_key_normalizes_omitted_and_explicit_defaults():
    import torch

    from DLKernel.gemm_interface import gemm_tuned
    from DLKernel.varlen_selector import gemm_tune_key

    a = torch.empty((4096, 2048))
    b = torch.empty((2048, 1536))
    out = torch.empty((4096, 1536))
    cu = torch.zeros((65,), dtype=torch.int32)
    names = gemm_tuned.arg_names
    defaults = gemm_tuned.arg_defaults
    positional = gemm_tune_key((a, b, out), {"cu_seqlens_m": cu}, names, defaults)
    explicit = {
        name: default
        for name, default in zip(names, defaults)
        if default is not inspect.Signature.empty
    }
    explicit.update({"A": a, "B": b, "out": out, "cu_seqlens_m": cu})
    keyword = gemm_tune_key((), explicit, names, defaults)
    assert positional == keyword


def test_large_sm90_varlen_m_prunes_to_the_candidate_budget(monkeypatch):
    """End to end through the pruning hook: an in-envelope varlen call is
    reduced to the shipped budget, as a subset of the structural pool."""
    import torch
    import DLKernel.gemm_interface as gi
    from DLKernel.gemm_config import _get_sm90_configs
    from DLKernel.varlen_selector import KEEP

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    configs = [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()]
    structural = set(range(len(configs)))
    kept = _select_via_policy(
        configs,
        {
            "A": torch.empty((258012, 2048), dtype=torch.bfloat16),
            "B": torch.empty((768, 2048, 1536), dtype=torch.bfloat16),
            "out": torch.empty((258012, 1536), dtype=torch.bfloat16),
            "cu_seqlens_m": torch.zeros((769,), dtype=torch.int32),
        },
    )
    assert len(kept) == KEEP
    kept_ids = {id(conf) for conf in kept}
    assert kept_ids <= {id(conf) for conf in configs}
    assert len(structural) == len(configs)


def test_large_sm90_varlen_k_prunes_to_the_candidate_budget(monkeypatch):
    import torch
    import DLKernel.gemm_interface as gi
    from DLKernel.gemm_config import _get_sm90_configs
    from DLKernel.varlen_selector import KEEP

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    configs = [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()]
    kept = _select_via_policy(
        configs,
        {
            "A": torch.empty((2048, 258012), dtype=torch.bfloat16),
            "B": torch.empty((258012, 768), dtype=torch.bfloat16),
            "out": torch.empty((2048, 768), dtype=torch.bfloat16),
            # Only prefix-sum length is inspected: 32 experts are in coverage.
            # Routed K comes from B.shape[0], not the stored values.
            "cu_seqlens_k": torch.arange(0, 33, dtype=torch.int32) * 1000,
        },
    )
    assert len(kept) == KEEP


def test_large_sm90_varlen_out_of_coverage_keeps_the_structural_pool(monkeypatch):
    """float32 is outside the measured coverage, so the selector must decline
    and hand the structural pool (22 M-group / 44 K-group + 2 dynamic twins)
    to the autotuner."""
    import torch
    import DLKernel.gemm_interface as gi
    from DLKernel.gemm_config import _get_sm90_configs

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    configs = [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()]
    kept_m = _select_via_policy(
        configs,
        {
            "A": torch.empty((196608, 2048)),
            "B": torch.empty((768, 2048, 1536)),
            "cu_seqlens_m": torch.zeros((769,), dtype=torch.int32),
        },
    )
    kept_k = _select_via_policy(
        configs,
        {
            "A": torch.empty((2048, 393216)),
            "B": torch.empty((393216, 768)),
            "cu_seqlens_k": torch.zeros((769,), dtype=torch.int32),
        },
    )
    assert len(kept_m) == 22  # varlen_m never gains the dynamic twins
    assert len(kept_k) == 46  # 44 static + 2 dynamic twins


def test_large_sm90_varlen_k_high_keeps_the_dynamic_twins(monkeypatch):
    """The k-high prune keeps the 6 static matches plus the measured
    dynamic-persistent twins; every other leaf rejects dynamic configs."""
    import torch
    import DLKernel.gemm_interface as gi
    from DLKernel.gemm_config import _get_sm90_configs, get_sm90_dynamic_varlen_configs
    from DLKernel.varlen_selector import KEEP

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    # the hook itself appends the twins to the static pool (production path)
    twin_configs = get_sm90_dynamic_varlen_configs()
    assert len(twin_configs) == 2
    kept = _select_via_policy(
        [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()],
        {
            "A": torch.empty((2048, 258012), dtype=torch.bfloat16),
            "B": torch.empty((258012, 768), dtype=torch.bfloat16),
            "out": torch.empty((2048, 768), dtype=torch.bfloat16),
            "cu_seqlens_k": torch.arange(0, 33, dtype=torch.int32) * 1000,
        },
    )
    assert len(kept) == KEEP
    kept_dynamic = [conf for conf in kept if conf.kwargs["config"].is_dynamic_persistent]
    assert {
        (conf.kwargs["config"].tile_m, conf.kwargs["config"].swap_ab) for conf in kept_dynamic
    } == {(256, False), (256, True)}


def test_large_sm90_varlen_selector_kill_switch(monkeypatch):
    """DLKERNEL_VARLEN_SELECTOR=0 restores the full structural pool."""
    import torch
    import DLKernel.gemm_interface as gi
    from DLKernel.gemm_config import _get_sm90_configs
    from DLKernel.varlen_selector import KEEP

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    monkeypatch.setenv("DLKERNEL_VARLEN_SELECTOR", "0")
    configs = [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()]
    kept = _select_via_policy(
        configs,
        {
            "A": torch.empty((2048, 258012), dtype=torch.bfloat16),
            "B": torch.empty((258012, 768), dtype=torch.bfloat16),
            "out": torch.empty((2048, 768), dtype=torch.bfloat16),
            "cu_seqlens_k": torch.arange(0, 33, dtype=torch.int32) * 1000,
        },
    )
    assert len(kept) > KEEP


def test_large_sm90_gather_prune_keeps_structural_filters(monkeypatch):
    """gather_A is outside the selector's coverage, so the structural rules are
    the only thing pruning: cluster_n == 1, no swap_ab, and use_tma_gather is
    rejected below SM100."""
    import torch
    from dataclasses import replace
    import DLKernel.gemm_interface as gi
    from DLKernel.gemm_config import _get_sm90_configs

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    configs = [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()]
    structural = next(
        c for c in configs if c.kwargs["config"].cluster_n == 1 and not c.kwargs["config"].swap_ab
    )
    configs.append(
        gi.AutotuneConfig(config=replace(structural.kwargs["config"], use_tma_gather=True))
    )
    kept = _select_via_policy(
        configs,
        {
            "A": torch.empty((4096, 2048)),
            "cu_seqlens_m": torch.zeros((9,), dtype=torch.int32),
            "A_idx": torch.empty((258012,), dtype=torch.int32),
        },
    )
    assert kept
    assert all(c.kwargs["config"].cluster_n == 1 for c in kept)
    assert all(not c.kwargs["config"].swap_ab for c in kept)
    assert all(not c.kwargs["config"].use_tma_gather for c in kept)


def test_autotune_bench_loop_defers_and_retries(monkeypatch):
    """A config whose kernel raises CompilePending is rotated to the back and
    retried once its sha polls done; all configs end up benchmarked exactly
    as if they had been warm.

    This encodes the compile-only rip-out contract: the autotuner no longer
    precompiles via fake tensors — the bench loop discovers cold keys with
    the real tensors in-process and overlaps compilation via the pool.
    """
    from DLKernel.autotuner import Autotuner, AutotuneConfig
    from DLKernel.cache import async_compile
    from DLKernel.cache.async_compile import CompilePending

    class _StubPool:
        """poll() reports 'pending' once per sha, then 'done'."""

        def __init__(self):
            self.polls = {}

        def poll(self, sha):
            n = self.polls.get(sha, 0) + 1
            self.polls[sha] = n
            return ("pending" if n == 1 else "done"), None

    stub = _StubPool()
    monkeypatch.setattr(async_compile, "_active_pool", stub)

    bench_order = []
    raised_once = set()

    def kernel(x, block: int = 0):
        # config block=1 is "cold": its first invocation defers.
        if block == 1 and 1 not in raised_once:
            raised_once.add(1)
            raise CompilePending("f" * 64, "fake._compile_kernel")
        bench_order.append(block)

    def do_bench(fn, quantiles=None, **kw):
        fn()
        return [1.0 + bench_order[-1], 1.0, 1.0]  # block=0 fastest

    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1, 2)],
        do_bench=do_bench,
    )
    import torch

    x = torch.empty(4, device="cuda")
    tuner(x)

    # Bench order: block=1 deferred, so it benched AFTER block 2 (exactly
    # once). The trailing 0 is __call__'s real invocation with the winner.
    assert bench_order == [0, 2, 1, 0], bench_order
    assert stub.polls == {"f" * 64: 2}  # one rotation, one release
    assert len(tuner.configs_timings) == 3
    best = tuner.cache[next(iter(tuner.cache))]
    assert best.kwargs["block"] == 0  # timings intact despite the deferral


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
def test_autotune_wedged_pool_falls_back_in_process(monkeypatch):
    """A sha that never resolves must not hang the sweep: past the attempt
    cap the config is benched with the pool suppressed (in-process compile),
    so autotuning always terminates.
    """
    import DLKernel.autotuner as at
    from DLKernel.autotuner import Autotuner, AutotuneConfig
    from DLKernel.cache import async_compile
    from DLKernel.cache.async_compile import CompilePending, get_active_pool

    class _WedgedPool:
        def poll(self, sha):
            return "pending", None  # never completes

    monkeypatch.setattr(async_compile, "_active_pool", _WedgedPool())
    monkeypatch.setattr(at, "_POOL_WEDGE_TIMEOUT_S", 0.2)

    benched = []

    def kernel(x, block: int = 0):
        # Defer as long as a pool is visible; succeed once suppressed.
        if block == 1 and get_active_pool() is not None:
            raise CompilePending("e" * 64, "fake._compile_kernel")
        benched.append(block)

    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1)],
        do_bench=lambda fn, quantiles=None, **kw: (fn(), [1.0, 1.0, 1.0])[1],
    )
    import torch

    tuner(torch.empty(4, device="cuda"))
    assert benched.count(1) == 1  # eventually ran, via suppress_pool
    assert len(tuner.configs_timings) == 2


def test_gemm_tune_key_declines_for_dense_calls():
    import torch

    from DLKernel.gemm_interface import gemm_tuned
    from DLKernel.varlen_selector import gemm_tune_key

    # No cu_seqlens anywhere: the hook must opt out so the autotuner keeps its
    # own default key for dense GEMM.
    args = (
        torch.empty((128, 64)),
        torch.empty((64, 128)),
        torch.empty((128, 128)),
        None,
        None,
        1.0,
        1.0,
        None,
        None,
        None,
    )
    assert gemm_tune_key(args, {}, gemm_tuned.arg_names) is None
    assert (
        gemm_tune_key(args[:7], {"cu_seqlens_m": None, "cu_seqlens_k": None}, gemm_tuned.arg_names)
        is None
    )


def test_gemm_tune_key_buckets_varlen_arguments_passed_by_keyword():
    import torch

    from DLKernel.gemm_interface import gemm_tuned
    from DLKernel.varlen_selector import gemm_tune_key

    # DLKernel/gemm.py and the epilogue frontend pass cu_seqlens by keyword, so
    # the hook must recognise that form and bucket it exactly like the
    # positional one.
    def key_for(total_k, total_m=None):
        a = torch.empty((2048, total_k))
        b = torch.empty((total_k, 768))
        out = torch.empty((2048 if total_m is None else total_m, 768))
        kwargs = {"cu_seqlens_k": torch.zeros((9,), dtype=torch.int32)}
        return gemm_tune_key((a, b, out, None, None, 1.0, 1.0), kwargs, gemm_tuned.arg_names)

    assert key_for(258012) is not None
    # 8 experts and 258012 routed rows is 32252 rows/expert, which lands in the
    # 32768 bucket: every total from 131073 to 262144 shares one tune.
    assert key_for(258012) == key_for(258013)
    assert key_for(258012) == key_for(200000)
    # 512303 is 64038 rows/expert: the next bucket, so it tunes separately.
    assert key_for(512303) != key_for(258012)


def test_bench_budget_override_is_gated_on_the_varlen_key_hook(monkeypatch):
    """Only a call that opted into the coarse key may change the bench budget.

    The autotuner object is shared by dense and routed GEMM, so the knob must
    not leak: a dense tune keeps the historical 200 ms / 200-call protocol even
    when the environment asks for a shorter cold-start sweep.
    """
    import torch

    from DLKernel.autotuner import _DEFAULT_BENCH_TIMED_CALLS, _DEFAULT_BENCH_WARMUP_MS, Autotuner

    tuner = Autotuner(lambda *a, **k: None, [], [AutotuneConfig(tile_m=128)])
    monkeypatch.setenv("DLKERNEL_TUNE_WARMUP_MS", "40")
    monkeypatch.setenv("DLKERNEL_TUNE_TIMED_CALLS", "32")

    # dense: hook declines -> historical protocol, environment ignored
    tuner._budget_override = False
    assert tuner._bench_budget() == (_DEFAULT_BENCH_WARMUP_MS, _DEFAULT_BENCH_TIMED_CALLS)

    # routed: hook returns a key -> the override applies, clamped into range
    tuner._budget_override = True
    assert tuner._bench_budget() == (40.0, 32)

    monkeypatch.setenv("DLKERNEL_TUNE_TIMED_CALLS", "4")
    assert tuner._bench_budget() == (40.0, 16)  # floor keeps the sweep meaningful


def test_call_sets_the_budget_flag_from_the_policy(monkeypatch):
    """``__call__`` derives the flag from the policy, not from stale state.

    Both directions are pinned with a warm cache, so no kernel is ever benched.
    """
    import torch

    from DLKernel.autotuner import AutotuneConfig, Autotuner, TunePolicy

    configs = [AutotuneConfig(x=1), AutotuneConfig(x=2)]
    calls = []

    class DecliningPolicy(TunePolicy):
        # Default key, no budget opt-in: the plain-operator specialization.
        def make_key(self, args, kwargs, default_key):
            calls.append(1)
            return default_key()

    tuner = Autotuner(lambda *a, **k: None, [], configs, policy=DecliningPolicy())
    tensor = torch.empty(4, 4)
    default_key = ((4, 4), (2, 1), tensor.dtype)  # what __call__ builds for a bare tensor
    tuner.cache[default_key] = configs[0]
    tuner._budget_override = True  # stale value from an earlier routed call
    tuner(tensor)
    assert calls == [1]
    assert tuner._budget_override is False

    class RoutedPolicy(TunePolicy):
        # Produces its own bucket-style key and opts into the routed budget.
        def make_key(self, args, kwargs, default_key):
            return ("routed",)

        def budget_override(self, key):
            return True

    routed = Autotuner(lambda *a, **k: None, [], configs, policy=RoutedPolicy())
    routed.cache[("routed",)] = configs[0]
    routed(tensor)
    assert routed._budget_override is True


def test_bench_budget_rejects_invalid_environment_values(monkeypatch):
    from DLKernel.autotuner import AutotuneConfig, Autotuner

    tuner = Autotuner(lambda x, **kwargs: None, [], [AutotuneConfig(x=1)])
    tuner._budget_override = True
    monkeypatch.setenv("DLKERNEL_TUNE_WARMUP_MS", "nan")
    with pytest.raises(ValueError, match="DLKERNEL_TUNE_WARMUP_MS"):
        tuner._bench_budget()
    monkeypatch.setenv("DLKERNEL_TUNE_WARMUP_MS", "40")
    monkeypatch.setenv("DLKERNEL_TUNE_TIMED_CALLS", "oops")
    with pytest.raises(ValueError, match="DLKERNEL_TUNE_TIMED_CALLS"):
        tuner._bench_budget()


def test_varlen_selector_accepts_canonical_sm_arch_override(monkeypatch):
    import torch

    from DLKernel.autotuner import AutotuneConfig
    import DLKernel.varlen_selector as selector_mod
    from DLKernel.varlen_selector import VarlenGemmTunePolicy

    seen = []

    def selector(configs, **kwargs):
        seen.append(kwargs["device_capability"])
        return configs

    monkeypatch.setattr(selector_mod, "select_varlen_gemm_candidates", selector)
    policy = VarlenGemmTunePolicy()
    policy.bind_signature(("A", "B", "cu_seqlens_m"), (None,) * 3)
    monkeypatch.setenv("DLKERNEL_ARCH", "sm_90")
    A = torch.empty((64, 16), dtype=torch.bfloat16)
    B = torch.empty((2, 16, 32), dtype=torch.bfloat16)
    cu = torch.empty((3,), dtype=torch.int32)
    configs = [AutotuneConfig(config="a")]
    assert policy.shortlist(configs, {"A": A, "B": B}, {"cu_seqlens_m": cu}) == configs
    assert seen == [9]


# ---------------------------------------------------------------------------
# Autotuner policy/cache contracts (CPU-only)
# ---------------------------------------------------------------------------


@pytest.fixture
def _cpu_tuning(monkeypatch, tmp_path):
    """Disable GPU-only setup while exercising the host-side tuner contract."""
    from contextlib import nullcontext

    import DLKernel.autotuner as at
    from DLKernel.cache import async_compile

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("DLKERNEL_FORCE_CACHE_UPDATE", raising=False)
    monkeypatch.setattr(at, "_gpu_warmup", lambda: None)
    monkeypatch.setattr(async_compile, "pool_scope", lambda: nullcontext(None))


def _bench_once(fn, **kwargs):
    fn()
    return [1.0, 1.0, 1.0]


@pytest.mark.parametrize("direction", ["cu_seqlens_m", "cu_seqlens_k"])
@pytest.mark.parametrize("positional", [False, True])
def test_autotuner_required_cache_rejects_varlen_miss(
    _cpu_tuning, monkeypatch, direction, positional
):
    import torch
    import DLKernel.autotuner as at

    monkeypatch.setenv("DLKERNEL_REQUIRE_TUNE_CACHE", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("required-cache miss entered warmup or benchmarking")

    monkeypatch.setattr(at, "_gpu_warmup", forbidden)

    def kernel(x, cu_seqlens_m=None, cu_seqlens_k=None, block=1):
        return x * block

    tuner = at.Autotuner(
        kernel,
        configs=[at.AutotuneConfig(block=b) for b in (1, 2)],
        key=[],
        do_bench=forbidden,
        cache_results=True,
    )
    x = torch.arange(4.0)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="Missing autotune cache"):
        if positional:
            tuner(x, cu if direction == "cu_seqlens_m" else None,
                  cu if direction == "cu_seqlens_k" else None)
        else:
            tuner(x, **{direction: cu})


def test_autotuner_cache_miss_tunes_when_required_cache_is_unset(_cpu_tuning, monkeypatch):
    import torch
    import DLKernel.autotuner as at

    monkeypatch.delenv("DLKERNEL_REQUIRE_TUNE_CACHE", raising=False)

    def kernel(x, block=1):
        return x * block

    tuner = at.Autotuner(
        kernel,
        configs=[at.AutotuneConfig(block=b) for b in (1, 2)],
        key=[],
        do_bench=_bench_once,
        cache_results=True,
    )
    x = torch.arange(4.0)
    torch.testing.assert_close(tuner(x), x)


def test_autotuner_does_not_cache_an_all_failed_sweep(_cpu_tuning):
    import torch
    import DLKernel.autotuner as at

    def failed_bench(fn, **kwargs):
        fn()
        return [float("inf")] * 3

    def kernel(x, block=1):
        return x * block

    tuner = at.Autotuner(
        kernel,
        configs=[at.AutotuneConfig(block=b) for b in (1, 2)],
        key=[],
        do_bench=failed_bench,
        cache_results=True,
    )
    with pytest.raises(RuntimeError, match="all .* candidates failed"):
        tuner(torch.arange(4.0))
    assert tuner.cache == {}


class _RecordingPolicy(TunePolicy):
    """Small policy test double that records the public hook order."""

    def __init__(self, events):
        self.events = events

    def make_key(self, args, kwargs, default_key):
        self.events.append("key")
        return ("policy", default_key())

    def prepare_candidates(self, configs, named_args, kwargs):
        self.events.append("prepare")
        return list(configs)

    def shortlist(self, configs, named_args, kwargs):
        self.events.append("shortlist")
        return list(configs[:1])


def test_autotuner_policy_hooks_run_in_order():
    import torch
    from DLKernel.autotuner import AutotuneConfig, Autotuner

    events = []

    def kernel(x, scale=1, block=1):
        return x * scale * block

    tuner = Autotuner(
        kernel,
        configs=[AutotuneConfig(block=1), AutotuneConfig(block=2)],
        key=["scale"],
        policy=_RecordingPolicy(events),
        do_bench=_bench_once,
    )
    x = torch.arange(4.0)
    torch.testing.assert_close(tuner(x, scale=3), x * 3)
    assert events == ["key", "prepare", "shortlist"]


def test_autotuner_policy_rejects_empty_or_duplicate_candidates():
    import torch
    from DLKernel.autotuner import AutotuneConfig, Autotuner, TunePolicy

    configs = [AutotuneConfig(x=1), AutotuneConfig(x=2)]

    class EmptyPolicy(TunePolicy):
        def prepare_candidates(self, configs, named_args, kwargs):
            return []

    empty = Autotuner(
        lambda x, config=None: None,
        [],
        configs,
        policy=EmptyPolicy(),
        do_bench=lambda *args, **kwargs: pytest.fail("benchmark must not run"),
    )
    with pytest.raises(RuntimeError, match="empty candidate"):
        empty(torch.empty(1))

    class DuplicatePolicy(TunePolicy):
        def shortlist(self, configs, named_args, kwargs):
            return [configs[0], configs[0]]

    duplicate = Autotuner(
        lambda x, config=None: None,
        [],
        configs,
        policy=DuplicatePolicy(),
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        duplicate(torch.empty(1))


def test_gemm_policy_keeps_split_k_legality_separate_from_shortlist(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    import torch

    from DLKernel.autotuner import AutotuneConfig
    from DLKernel.gemm_config import GemmConfig
    from DLKernel.gemm_tune_policy import GemmTunePolicy

    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=132),
    )
    policy = GemmTunePolicy()
    config = GemmConfig(device_capacity=9)
    base = [AutotuneConfig(config=config)]
    named = {"A": torch.empty((4096, 4096)), "B": torch.empty((4096, 4096))}

    prepared = policy.prepare_candidates(base, named, {"split_k": None})
    assert prepared == [
        *base,
        *(AutotuneConfig(config=replace(config, split_k=s)) for s in (2, 4, 8, 16)),
    ]
    # Occupancy is a shortlist concern; this shape is not occupancy-starved.
    assert policy.shortlist(prepared, named, {"split_k": None}) == base
