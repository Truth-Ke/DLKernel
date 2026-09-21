# Adapted from https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py
# Copyright (C) 2025, Tri Dao.
from __future__ import annotations

import builtins
import math
import os
import sys
import time
import inspect
import base64
import hashlib
import json
import fcntl
from pathlib import Path
from functools import cached_property, partial
from typing import Dict, Tuple, List, Optional, Any
from DLKernel.bench.bench_utils import (
    _bench_cuda_graph_l2_rotate,
    _clone_l2_rotate_inputs,
    _pick_l2_rotate_count,
)

import torch
from torch import Tensor

import triton

from . import __version__


PACKAGE_NAME = "dlkernel"
VERSION = __version__

#: Historical L2-cold bench protocol, read off the helper so the two can never
#: drift: dense calls always pass exactly these values.
_BENCH_SIGNATURE = inspect.signature(_bench_cuda_graph_l2_rotate).parameters
_DEFAULT_BENCH_WARMUP_MS = float(_BENCH_SIGNATURE["warmup_target_ms"].default)
_DEFAULT_BENCH_TIMED_CALLS = int(_BENCH_SIGNATURE["n_timed_calls"].default)


def _canonical_bench_budget() -> Tuple[float, int]:
    """Parse and clamp the optional routed benchmark budget once per miss.

    Invalid values fail closed with an actionable error instead of surfacing as
    an unrelated ``ValueError`` from deep inside the benchmark loop.
    """
    warmup_name = f"{PACKAGE_NAME.upper()}_TUNE_WARMUP_MS"
    calls_name = f"{PACKAGE_NAME.upper()}_TUNE_TIMED_CALLS"
    raw_warmup = os.getenv(warmup_name)
    raw_calls = os.getenv(calls_name)
    if raw_warmup is None:
        warmup = _DEFAULT_BENCH_WARMUP_MS
    else:
        try:
            warmup = float(raw_warmup)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{warmup_name} must be a finite number in [10, 2000] ms") from exc
        if not math.isfinite(warmup) or warmup < 0:
            raise ValueError(f"{warmup_name} must be a finite number in [10, 2000] ms")
    if raw_calls is None:
        calls = _DEFAULT_BENCH_TIMED_CALLS
    else:
        try:
            calls = int(raw_calls)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{calls_name} must be an integer in [16, 2000]") from exc
        if calls < 0:
            raise ValueError(f"{calls_name} must be an integer in [16, 2000]")
    return min(max(warmup, 10.0), 2000.0), min(max(calls, 16), 2000)


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment switch without treating ``"0"`` as true."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _tune_trace(message: str) -> None:
    """Emit one line of opt-in tuning diagnostics (``DLKERNEL_DEBUG_TUNE=1``).

    Cold-start tuning happens once per key and is otherwise silent, so a trace
    hook is the only way to tell "tuned once" from "retuned on every call" or
    "read the disk cache".  It is one ``getenv`` on the bench path, which runs
    once per candidate, not per call.
    """
    if os.environ.get(f"{PACKAGE_NAME.upper()}_DEBUG_TUNE") == "1":
        print(message, flush=True)


def _key_digest(value: object) -> str:
    """Stable short identifier for a tuning key, for cross-process diagnostics."""
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:16]


def get_home_dir():
    return os.getenv(f"{PACKAGE_NAME.upper()}_HOME", Path.home())


def default_cache_dir():
    return os.path.join(get_home_dir(), f".{PACKAGE_NAME}", "cache")


class FileCacheManager(triton.runtime.cache.FileCacheManager):
    def __init__(self, key):
        super().__init__(key)
        self.cache_dir = (
            os.getenv(f"{PACKAGE_NAME.upper()}_CACHE_DIR", "").strip() or default_cache_dir()
        )
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            raise RuntimeError("Could not create or locate cache dir")


def _base32(key):
    # Assume key is a hex string.
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")


#: How long a deferred config may wait on its pool compile before the bench
#: loop stops trusting the pool and benches it with the pool suppressed
#: (in-process compile). Guards against a wedged worker / a foreign flock
#: holder that never produces the .o; without it a permanently-"pending"
#: sha would rotate forever. Tests override this.
_POOL_WEDGE_TIMEOUT_S = 300.0


def _gpu_warmup(duration_ms=200):
    """Saturate the GPU to reach thermal steady-state before benchmarking.

    Without this, the first autotuning config gets artificially good numbers
    because the GPU hasn't been power-throttled yet.
    """
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    target = duration_ms / 1000
    t0 = time.time()
    while time.time() - t0 < target:
        for _ in range(100):
            a = a @ a
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Candidate-config compilation
#
# There is no separate precompile phase: the bench loop in ``benchmark()``
# (inside ``Autotuner.__call__``) runs under ``pool_scope()`` from
# DLKernel.cache.async_compile. A config whose kernel misses the .o cache
# raises ``CompilePending`` from jit_cache after shipping the pickled
# ``_compile_*`` key to a CPU worker; the loop rotates that config to the
# back and benches whichever config is ready. Total wall stays
# max(parallel_compile, serial_bench), key discovery uses the real tensors
# in-process, and workers never launch kernels (they call the tensor-free
# ``_compile_*`` functions directly).
# ---------------------------------------------------------------------------


class TunePolicy:
    """Variation points for autotune semantics.

    The tuner owns the execution lifecycle; a policy only describes cache
    identity and candidate selection.  Methods intentionally receive plain
    arguments so the cache-hit path does not need to allocate a rich context.

    Onboarding a new operator family: subclass this, override only the hooks
    that genuinely differ, and leave the rest inherited.

    * ``make_key``       -- winner equivalence class / dynamic-shape bucketing.
    * ``prepare_candidates`` -- deterministic candidate generation and
      provably-safe legality pruning (may ADD configs, e.g. split-K variants).
    * ``shortlist``      -- heuristic reduction allowed to lose performance,
      never correctness; must return a non-empty subset of its input.
    ``make_key``, ``prepare_candidates`` and ``shortlist`` are the three
    candidate-policy hooks.  ``budget_override`` is deliberately only an
    internal benchmark-protocol coordination point; it does not participate in
    candidate correctness.

    ``make_key`` must never synchronize (no ``.item()``/``.tolist()``/
    ``.cpu()`` on device data): it runs on every cache-hit.  Policies do not
    retain per-call state; the tuner hands the decorated signature to ``bind_signature``
    once at construction.  See ``DLKernel/gemm_tune_policy.py`` (GEMM,
    epilogue GEMM) and ``DLKernel/rmsnorm.py`` (RMSNorm) for the three
    existing specialization levels.
    """

    def make_key(self, args, kwargs, default_key):
        return default_key()

    def bind_signature(self, arg_names, arg_defaults):
        """Receive the decorated function signature when the tuner is built.

        Policies are commonly constructed next to a decorator, before the
        wrapped function is available.  This hook keeps that API ergonomic
        while allowing policies that need canonical argument binding (for
        example GEMM's varlen key) to receive the signature once.
        """
        signature = (tuple(arg_names), tuple(arg_defaults))
        bound = getattr(self, "_bound_signature", None)
        if bound is not None and bound != signature:
            raise ValueError("TunePolicy signature is already bound to a different function")
        self._bound_signature = signature

    def budget_override(self, key):
        """Whether this key opts into the routed/varlen bench budget."""
        del key
        return False

    def prepare_candidates(self, configs, named_args, kwargs):
        return configs

    def shortlist(self, configs, named_args, kwargs):
        return configs


class Autotuner:
    def __init__(
        self,
        fn,
        key,
        configs,
        restore_value=None,
        do_bench=None,
        cache_results=False,
        policy=None,
    ):
        """
        :param policy: optional :class:`TunePolicy` describing this operator
            family's tune semantics (winner key, deterministic candidate
            preparation, heuristic shortlist).  ``None`` uses the base policy:
            the default exact key and the full config list.  Operators with
            varlen bucketing, legality pruning, or measured shortlists
            subclass :class:`TunePolicy` and pass the instance here.
        """
        if not configs:
            self.configs = [AutotuneConfig()]
        else:
            self.configs = configs
        signature = inspect.signature(fn)
        self.keys = key
        self.cache: Dict[Tuple, AutotuneConfig] = {}
        self.arg_names = list(signature.parameters.keys())
        self._arg_positions = {name: index for index, name in enumerate(self.arg_names)}
        self.arg_defaults = tuple(
            parameter.default for parameter in signature.parameters.values()
        )
        self._arg_name_set = frozenset(self.arg_names)
        self.cache_results = (
            cache_results or os.getenv(f"{PACKAGE_NAME.upper()}_CACHE_AUTOTUNING", None) == "1"
        )

        self.restore_value = []
        if restore_value is not None:
            self.restore_value = list(restore_value)

        if len(self.restore_value) > 0:

            def _pre_hook(kwargs):
                self.restore_copies = {name: kwargs[name].clone() for name in self.restore_value}

            self.pre_hook = _pre_hook
        else:
            self.pre_hook = None

        if len(self.restore_value) > 0:

            def _post_hook(kwargs, exception):
                for name in self.restore_value:
                    kwargs[name].copy_(self.restore_copies[name])
                self.restore_copies = {}

            self.post_hook = _post_hook
        else:
            self.post_hook = None

        self.fn = fn
        self._do_bench = do_bench
        self.policy = policy or TunePolicy()
        self.policy.bind_signature(tuple(self.arg_names), self.arg_defaults)
        # Cache-hit hot path: the exact base policy contributes nothing (its
        # make_key IS the default key and it never overrides the budget), so
        # plain operators keep the historical inline path with zero extra
        # indirection; custom policies pay one call per hook they use.
        self._base_policy = type(self.policy) is TunePolicy
        self._policy_make_key = self.policy.make_key
        self._policy_budget_override = self.policy.budget_override
        #: Set per call from ``policy.budget_override(key)``: True only when the
        #: policy tuned this call under its own routed/varlen key.  The
        #: environment bench-budget overrides are documented varlen-only and
        #: hang off exactly this signal.
        self._budget_override = False

    def _bench_budget(self) -> Tuple[float, int]:
        """``(warmup_ms, timed_calls)`` for the L2-cold protocol.

        Dense calls always get the historical protocol.  Only a call whose
        policy opted in via ``budget_override`` (a routed/varlen GEMM here)
        may read a different budget, and only when the operator asked for one
        through the environment; the values are clamped so a typo cannot turn tuning into a
        no-op or an hour.  The varlen key includes the budget environment
        settings, so changing the protocol does not reuse its old winner.
        """
        if not self._budget_override:
            return _DEFAULT_BENCH_WARMUP_MS, _DEFAULT_BENCH_TIMED_CALLS
        return _canonical_bench_budget()

    @cached_property
    def do_bench(self):
        if self._do_bench is None:
            return partial(triton.testing.do_bench, warmup=5, rep=25)
        return self._do_bench

    def _bench(self, *args, config, **meta):
        verbose = os.environ.get(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
        if verbose:
            print(f"Autotuning kernel {self.fn.__name__} with config {config}")

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        # Default path: L2-cold CUDA-graph round-robin bench. ``__call__``
        # sets ``self._l2_cold_arg_sets`` / ``self._l2_cold_kwarg_sets`` to
        # pre-cloned (args, kwargs) sets once per shape (reused across all
        # configs). Round-robin over fresh sets keeps the kernel measured
        # under the cache-cold conditions that match production access
        # patterns, so the autotuner picks configs that win at the same
        # workload the user actually runs.
        l2_cold_arg_sets = getattr(self, "_l2_cold_arg_sets", None)
        l2_cold_kwarg_sets = getattr(self, "_l2_cold_kwarg_sets", None)
        has_hooks = self.pre_hook is not None or self.post_hook is not None
        use_l2_cold = (
            self._do_bench is None
            and l2_cold_arg_sets is not None
            and l2_cold_kwarg_sets is not None
            and not has_hooks
        )

        if use_l2_cold:
            try:
                # Cold-start tuning can be bounded independently of the
                # steady-state benchmark protocol.  A shorter budget is only
                # ever used when explicitly requested through the environment,
                # so the historical 200 ms / 200-call protocol stays the
                # default for every caller.  The varlen key separates budget
                # settings: a 200-call and a 64-call sweep are not the same
                # measurement.
                warmup_target_ms, n_timed_calls = self._bench_budget()
                timings = _bench_cuda_graph_l2_rotate(
                    self.fn,
                    l2_cold_arg_sets,
                    l2_cold_kwarg_sets,
                    extra_kwargs=config.all_kwargs(),
                    warmup_target_ms=warmup_target_ms,
                    n_timed_calls=n_timed_calls,
                    quantiles=(0.5, 0.2, 0.8),
                )
                return timings
            except (RuntimeError, MemoryError) as e:
                # Narrow catch: only swallow GPU-side failures (smem
                # overflow, kernel launch errors, OOM). Programming errors
                # (TypeError, AssertionError, ValueError from conflict check
                # above) propagate so the user sees them.
                if verbose:
                    print(f"Autotuning failed with {type(e).__name__}: {e}")
                return [float("inf"), float("inf"), float("inf")]

        # Legacy path: triton.testing.do_bench or user-supplied do_bench.
        # Used when (a) a custom do_bench was passed via the decorator's
        # ``do_bench=`` arg, or (b) pre/post hooks are configured (the
        # clone/restore inside hooks doesn't work under CUDA graph capture).
        def kernel_call():
            if self.pre_hook is not None:
                self.pre_hook(full_nargs)
            try:
                self.fn.__call__(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    if self.post_hook is not None:
                        self.post_hook(full_nargs, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            if self.post_hook is not None:
                self.post_hook(full_nargs, exception=None)

        try:
            timings = self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
            return timings
        except Exception as e:
            if verbose:
                print(f"Autotuning failed with {e}")
            return [float("inf"), float("inf"), float("inf")]

    @torch.compiler.disable
    def check_disk_cache(self, tuning_key, configs, bench_fn):
        if not tuning_key:
            bench_fn()
            return

        fn = self.fn
        config_str_list = [str(c) for c in configs]
        assert len(config_str_list) == len(set(config_str_list)), "Config strings must be unique"
        cache_key = [VERSION, str(tuning_key)] + config_str_list
        cache_key = hashlib.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
        cache = FileCacheManager(_base32(cache_key))
        file_name = f"{fn.__name__[:150]}.autotune.json"
        path = cache.get_file(file_name)

        def load_cached(path):
            str2config = {s: c for s, c in zip(config_str_list, configs)}
            with open(path, "r") as cached_configs:
                timings = json.load(cached_configs)["configs_timings"]
                timings = {str2config[config]: timing for config, timing in timings}
                self.cache[tuning_key] = builtins.min(timings, key=timings.get)
                self.configs_timings = timings
                self.bench_time = 0

        # There's an environment variable to force cache update
        force_update = _env_flag(f"{PACKAGE_NAME.upper()}_FORCE_CACHE_UPDATE")
        if path and not force_update:
            _tune_trace(
                f"DLKERNEL_TUNE_DISK_HIT fn={fn.__name__} "
                f"key={_key_digest(tuning_key)} path={path}"
            )
            load_cached(path)
            return

        # ``DLKERNEL_REQUIRE_TUNE_CACHE=1`` turns a cold key into an error
        # instead of a bench: it is how a caller asserts "this shape was warmed
        # before the run started" and refuses to pay tuning inside a step.
        if os.environ.get(f"{PACKAGE_NAME.upper()}_REQUIRE_TUNE_CACHE") == "1":
            raise RuntimeError(
                f"Missing autotune cache for {fn.__name__}; prewarm this shape "
                f"before the run starts (key={_key_digest(tuning_key)})"
            )

        # Multiple ranks commonly discover the same cold key concurrently.
        # Serialize the expensive benchmark and re-check after acquiring the
        # lock so only one rank searches the candidate set.
        lock_path = os.path.join(cache.cache_dir, f"{file_name}.lock")
        _tune_trace(
            f"DLKERNEL_TUNE_DISK_MISS fn={fn.__name__} key={_key_digest(tuning_key)}"
        )
        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            path = cache.get_file(file_name)
            if path and not force_update:
                load_cached(path)
                return
            bench_fn()
            cache.put(
                json.dumps(
                    {
                        "key": str(tuning_key),
                        "configs_timings": [
                            (str(config), timings) for config, timings in self.configs_timings.items()
                        ],
                    }
                ),
                file_name,
                binary=False,
                )

    @staticmethod
    def _validate_candidate_pipeline(prepared, shortlisted):
        """Validate the policy contract before any compilation or benchmark."""
        prepared = list(prepared)
        shortlisted = list(shortlisted)
        if not prepared:
            raise RuntimeError("policy produced an empty candidate set during preparation")
        if not shortlisted:
            raise RuntimeError("policy produced an empty candidate set")
        for index, candidate in enumerate(shortlisted):
            if candidate not in prepared:
                raise RuntimeError(
                    f"policy shortlist candidate at index {index} is not in prepared candidates"
                )
            if candidate in shortlisted[:index]:
                raise RuntimeError("policy shortlist contains duplicate candidates")
        return prepared, shortlisted

    def __call__(self, *args, **kwargs):
        used_cached_result = True
        if len(self.configs) > 1:
            if self._base_policy:
                key = None
                self._budget_override = False
            else:
                key = self._policy_make_key(
                    args,
                    kwargs,
                    lambda: self._default_key(args, kwargs),
                )
                self._budget_override = self._policy_budget_override(key)
            if key is None:
                # Cache-hit fast path: build the key straight from args/kwargs.
                # This runs on every tuned call, so avoid dict merges and str()
                # formatting. The merged named-args view is materialized only
                # on a miss, for pruning.
                key = self._default_key(args, kwargs)
            named_args = dict(zip(self.arg_names, args))
            if key not in self.cache:
                self.nargs = named_args
                used_cached_result = False
                prepared_configs = self.policy.prepare_candidates(self.configs, self.nargs, kwargs)
                prepared_input = list(prepared_configs)
                prepared_snapshot = list(prepared_input)
                shortlisted = self.policy.shortlist(prepared_input, self.nargs, kwargs)
                if prepared_input != prepared_snapshot:
                    raise RuntimeError("policy shortlist mutated the prepared candidate list")
                prepared_configs, pruned_configs = self._validate_candidate_pipeline(
                    prepared_input,
                    shortlisted,
                )

                @torch.compiler.disable  # Don't want any tracing here
                def benchmark():
                    # Compile/bench overlap via the async compile pool
                    # (DLKernel.cache.async_compile): the bench loop runs inside
                    # pool_scope(). A config whose kernel isn't compiled yet
                    # raises CompilePending from jit_cache (after shipping the
                    # key to a CPU worker); the loop rotates it to the back
                    # and benches whichever config is ready, retrying once its
                    # .o lands. Discovery happens in-process with the real
                    # tensors (no fake-tensor reconstruction), and the pool
                    # workers replay the pickled _compile_* key directly --
                    # which never launches, by construction.
                    #
                    # CompilePending can only fire OUTSIDE CUDA graph capture:
                    # the L2-cold bench does priming launches before capture,
                    # and the legacy do_bench path warms up first, so a cold
                    # key raises at the first plain launch.
                    from collections import deque

                    from DLKernel.cache.async_compile import (
                        CompilePending,
                        pool_scope,
                        suppress_pool,
                    )

                    bench_start = time.time()
                    _tune_trace(
                        f"DLKERNEL_TUNE_START fn={self.fn.__name__} "
                        f"key={_key_digest(key)} configs={len(pruned_configs)}"
                    )
                    verbose = os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
                    has_hooks = self.pre_hook is not None or self.post_hook is not None
                    timings = {}
                    _MAX_ATTEMPTS = 20
                    try:
                        _gpu_warmup()
                        # Pre-allocate cloned (args, kwargs) sets once per
                        # shape; the same sets are reused across all configs
                        # to avoid ~400x re-cloning. Skipped when hooks are
                        # present or a custom do_bench was supplied (legacy
                        # fallback in _bench).
                        if self._do_bench is None and not has_hooks:
                            try:
                                n_buffers = _pick_l2_rotate_count(args, kwargs)
                                arg_sets, kwarg_sets = _clone_l2_rotate_inputs(
                                    args, kwargs, n_buffers
                                )
                                self._l2_cold_arg_sets = arg_sets
                                self._l2_cold_kwarg_sets = kwarg_sets
                            except (RuntimeError, MemoryError):
                                # Cloning failed (likely OOM at extreme N);
                                # legacy do_bench path will be used by _bench.
                                # The two protocols rank configs differently,
                                # so say so instead of degrading silently.
                                self._l2_cold_arg_sets = None
                                self._l2_cold_kwarg_sets = None
                                _tune_trace(
                                    "DLKERNEL_TUNE_L2_CLONE_UNAVAILABLE "
                                    f"fn={self.fn.__name__} falling back to legacy do_bench"
                                )
                        else:
                            self._l2_cold_arg_sets = None
                            self._l2_cold_kwarg_sets = None

                        with pool_scope() as pool:
                            queue = deque(pruned_configs)
                            awaiting = {}  # id(config) -> sha
                            attempts = {}  # id(config) -> int
                            deadline = {}  # id(config) -> wedge deadline
                            spins = 0
                            while queue:
                                config = queue.popleft()
                                sha = awaiting.get(id(config))
                                wedged = sha is not None and time.monotonic() > deadline[id(config)]
                                if sha is not None and not wedged:
                                    state, _ = pool.poll(sha)
                                    if state == "pending":
                                        queue.append(config)
                                        spins += 1
                                        if spins >= len(queue):
                                            time.sleep(0.05)
                                            spins = 0
                                        continue
                                spins = 0
                                n = attempts.get(id(config), 0) + 1
                                attempts[id(config)] = n
                                try:
                                    if wedged or n > _MAX_ATTEMPTS:
                                        # Wedged pool: compile in-process so
                                        # the sweep always terminates.
                                        with suppress_pool():
                                            timings[config] = self._bench(
                                                *args, config=config, **kwargs
                                            )
                                    else:
                                        timings[config] = self._bench(
                                            *args, config=config, **kwargs
                                        )
                                except CompilePending as e:
                                    awaiting[id(config)] = e.sha
                                    deadline.setdefault(
                                        id(config),
                                        time.monotonic() + _POOL_WEDGE_TIMEOUT_S,
                                    )
                                    queue.append(config)
                    finally:
                        # Free L2-cold sets before persisting the cache so the
                        # user's subsequent .fn(...) call has full HBM.
                        self._l2_cold_arg_sets = None
                        self._l2_cold_kwarg_sets = None
                    bench_end = time.time()
                    if verbose:
                        for config, time_ in timings.items():
                            print(f"[{config}] -> {time_[0]:.3f}ms")
                    # Surface bench failures (configs returning inf timings)
                    # so smem-overflow / launch errors aren't silently masked.
                    n_failed = sum(1 for t in timings.values() if not math.isfinite(float(t[0])))
                    if n_failed:
                        print(
                            f"DLKernel autotune: {n_failed}/{len(timings)} configs "
                            f"failed for {self.fn.__name__}{key}; "
                            f"set {PACKAGE_NAME.upper()}_PRINT_AUTOTUNING=1 for details",
                            file=sys.stderr,
                        )
                    self.bench_time = bench_end - bench_start
                    _tune_trace(
                        f"DLKERNEL_TUNE_END fn={self.fn.__name__} "
                        f"key={_key_digest(key)} seconds={self.bench_time:.3f}"
                    )
                    self.configs_timings = timings
                    finite_timings = {
                        config: timing
                        for config, timing in timings.items()
                        if math.isfinite(float(timing[0]))
                    }
                    if not finite_timings:
                        raise RuntimeError(
                            f"all {len(timings)} candidates failed for {self.fn.__name__}{key}"
                        )
                    self.cache[key] = builtins.min(finite_timings, key=finite_timings.get)

                if self.cache_results:
                    self.check_disk_cache(key, pruned_configs, benchmark)
                else:
                    benchmark()

            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        # Cheap flag first: the getenv (plus its f-string) is measurable on the
        # per-call cache-hit path.
        if not used_cached_result and (
            os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
        ):
            print(
                f"{PACKAGE_NAME} autotuning for function {self.fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )
        ret = self.fn(*args, **kwargs, **config.all_kwargs())
        self.nargs = None
        return ret

    def _default_key(self, args, kwargs):
        """Build the legacy exact metadata key with canonical named scalars."""
        key = []
        for name in self.keys:
            if name in kwargs:
                key.append(kwargs[name])
                continue
            position = self._arg_positions.get(name)
            if position is not None and position < len(args):
                key.append(args[position])
                continue
            default = self.arg_defaults[position] if position is not None else inspect.Parameter.empty
            if default is not inspect.Parameter.empty:
                key.append(default)
        for arg in args:
            if isinstance(arg, Tensor):
                key.append(tuple(arg.shape))
                key.append(tuple([s if s < 2 else 2 for s in arg.stride()]))
                key.append(arg.dtype)
        for name, arg in kwargs.items():
            if isinstance(arg, Tensor) and name in self._arg_name_set:
                key.append(tuple(arg.shape))
                key.append(tuple([s if s < 2 else 2 for s in arg.stride()]))
                key.append(arg.dtype)
        return tuple(key)


class AutotuneConfig:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.kwargs = state.get("kwargs", {})

    def all_kwargs(self):
        return self.kwargs

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        return ", ".join(res)

    def __hash__(self):
        return hash(tuple(self.all_kwargs().items()))

    def __eq__(self, other):
        self_tuple = tuple(self.all_kwargs().items())
        other_tuple = tuple(other.all_kwargs().items())
        return self_tuple == other_tuple


def autotune(
    configs, key=None, restore_value=None, do_bench=None, cache_results=True, policy=None,
):
    f"""
    Decorator for auto-tuning a function function.

    .. highlight:: python

    If the environment variable :code:`{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING` is set to
    :code:`"1"`, we will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`AutotuneConfig` objects
    :type configs: list[AutotuneConfig]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param policy: optional TunePolicy for this operator family, see :class:`Autotuner`.
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    :param cache_results: whether to cache autotune timings to disk.  Defaults to False.
    "type cache_results: bool
    """

    if key is None:
        key = []

    def decorator(fn):
        return Autotuner(
            fn,
            key,
            configs,
            restore_value=restore_value,
            do_bench=do_bench,
            cache_results=cache_results,
            policy=policy,
        )

    return decorator
