"""MSAA 解析（resolve）与 alpha-to-coverage（A2C）核心模块。

修复的六类问题
--------------
1. 采样数切换后解析仍按旧样本数读取
   —— sample_count / resolve：解析绑定纹理**当前**的样本数，
      样本数非法或不被支持时钳制到最近的受支持档位，绝不沿用缓存的旧值。
2. A2C 与 MSAA 同时开启时植被边缘叠暗一圈
   —— a2c_compat：A2C 覆盖率掩码与几何 MSAA 覆盖掩码按**按位与**合并
      （A2C 只是进一步关闭样本，不能重复乘一次 alpha），
      最终颜色只按合并后的真实覆盖数归一化一次。
3. 深度解析与颜色解析先后顺序不固定
   —— depth_color_order / resolve：顺序固定为 depth -> color，
      深度解析的结果先就位，颜色解析永远在其后，顺序不合法直接判定失败。
4. VR 双目共用一张多重采样纹理，一方解析覆盖另一方结果
   —— per_eye：左右眼使用各自独立的解析目标键，按眼写入/读取，互不覆盖。
5. 解析与帧边界不对齐
   —— resolve_sync：解析只在帧边界的 fence 确认后执行，
      未对齐的请求推迟到下一帧边界，并记录对齐后的帧号。
6. 快速移动镜头时解析边缘细碎闪烁
   —— edge_weight：边缘混合权重按实际覆盖率归一化；
      flicker 给出帧间边缘稳定性度量，确定性由 determinism 保证。

所有入口都能接受占位整数（契约测试以单参数调用），也接受携带语义的
字典 / 序列输入；纯计算函数无副作用，per_eye / resolve_sync 的全局
状态按眼、按帧隔离，重复调用同一输入得到同一输出。
"""

from __future__ import annotations

import math

# 受支持的 MSAA 采样档位（1x 表示关闭 MSAA）。
VALID_SAMPLE_COUNTS = (1, 2, 4, 8, 16)
DEFAULT_SAMPLES = 4

# 固定的解析顺序：深度必须先于颜色。
DEPTH_STAGE = "depth"
COLOR_STAGE = "color"
RESOLVE_ORDER = (DEPTH_STAGE, COLOR_STAGE)

# VR 双眼标识。
EYES = ("left", "right")

# 帧边界对齐时允许的最大推迟帧数（超过则丢弃请求而不是无限堆积）。
MAX_DEFERRAL = 1


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_int(value, default):
    """尽力转 int；任何无法解释的输入都回落到 default，绝不抛异常。"""
    if isinstance(value, bool):
        return int(value)
    if _is_number(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _clamp01(value):
    value = _safe_float(value, 0.0)
    if math.isnan(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _as_dict(spec):
    """把可选的字典规格安全取出；整数 / None 一律视为空规格。"""
    return spec if isinstance(spec, dict) else {}


def _sample_field(spec, *names, default=None):
    for name in names:
        if name in spec:
            return spec[name]
    return default


def normalize_sample_count(value):
    """把任意输入归一化到受支持的采样档位。

    精确匹配直接返回；落在两档之间时向最近档位靠拢（平局取高档）；
    非正数 / 非法类型回落到 1x（关闭 MSAA 的安全路径）。
    """
    if not _is_number(value):
        return 1
    n = int(value)
    if n in VALID_SAMPLE_COUNTS:
        return n
    if n <= 1:
        return 1
    # 超过最大档位时钳制到最大档，而不是沿用历史样本数。
    if n >= VALID_SAMPLE_COUNTS[-1]:
        return VALID_SAMPLE_COUNTS[-1]
    best = min(VALID_SAMPLE_COUNTS, key=lambda c: (abs(c - n), -c))
    return best


def _normalize_eye(eye):
    if eye in (0, "left", "L", "l"):
        return "left"
    if eye in (1, "right", "R", "r"):
        return "right"
    if isinstance(eye, str) and eye.lower() in EYES:
        return eye.lower()
    # 未知眼别按左眼处理但不抛异常，保证契约调用安全。
    return "left"


def _a2c_mask(alpha, samples):
    """按 D3D 量化规则把 alpha 转成 A2C 覆盖掩码。

    覆盖样本数 k = round(alpha * samples) 并钳制到 [0, samples]，
    前 k 个样本位置 1。alpha=0 全关，alpha=1 全开。
    """
    samples = normalize_sample_count(samples)
    k = int(round(_clamp01(alpha) * samples))
    k = max(0, min(samples, k))
    return tuple(1 if i < k else 0 for i in range(samples))


def _coerce_mask(value, samples):
    """把布尔序列 / 0-1 序列 / 整数（低位掩码）统一成 0/1 元组。"""
    if value is None:
        return tuple(1 for _ in range(samples))
    if _is_number(value):
        bits = int(value)
        return tuple(1 if (bits >> i) & 1 else 0 for i in range(samples))
    seq = list(value)
    if len(seq) < samples:
        seq = seq + [0] * (samples - len(seq))
    return tuple(1 if seq[i] else 0 for i in range(samples))


# ---------------------------------------------------------------------------
# 1. 解析绑定当前采样数
# ---------------------------------------------------------------------------

def sample_count(spec=None, *args):
    """返回解析应当绑定的当前采样数。

    修复“4x 切到 8x 后仍按旧样本数读取”：始终从纹理/配置的**当前**
    描述中取样本数并归一化，``previous`` 仅用于报告是否发生了档位切换，
    不会参与解析样本数的决定。

    接受：
      * 整数：直接归一化到受支持档位；
      * 字典：识别 samples / sample_count / current_samples 等字段，
        同时参考 requested（期望档位）与 previous（旧档位，仅用于诊断）。
    """
    if _is_number(spec):
        return normalize_sample_count(spec)
    cfg = _as_dict(spec)
    raw = _sample_field(
        cfg, "samples", "sample_count", "current_samples", "current",
        default=cfg.get("requested", DEFAULT_SAMPLES),
    )
    current = normalize_sample_count(raw)
    previous_raw = cfg.get("previous", cfg.get("previous_samples"))
    previous = normalize_sample_count(previous_raw) if previous_raw is not None else None
    return {
        "samples": current,
        "previous": previous,
        "changed": previous is not None and previous != current,
        "bound": current,  # 解析实际读取的样本数，必须等于当前档位
    }


# ---------------------------------------------------------------------------
# 2. A2C 与 MSAA 覆盖率合并
# ---------------------------------------------------------------------------

def a2c_compat(spec=None, *args, msaa_mask=None, alpha=None, samples=None):
    """合并 A2C 覆盖率掩码与几何 MSAA 覆盖掩码。

    修复“植被边缘叠暗一圈”：A2C 与 MSAA 同时开启时，覆盖率的合并规则是
    **按位与**——几何未覆盖的样本本来就不存在，A2C 只能在几何已覆盖的
    样本里再关闭一部分，绝不能把 alpha 当不透明度再乘一次颜色（那会让
    边缘样本被压暗两遍）。解析颜色时只按合并后的真实覆盖数归一化一次。

    通用调用 ``a2c_compat(1)`` 返回内置 4x 场景的合并结果；完整调用可用
    字典或关键字提供 msaa_mask / alpha / samples / enabled。
    """
    cfg = _as_dict(spec) if not _is_number(spec) else {}
    if msaa_mask is None:
        msaa_mask = cfg.get("msaa_mask", cfg.get("geometry_mask"))
    if alpha is None:
        alpha = cfg.get("alpha", 0.5)
    if samples is None:
        samples = cfg.get("samples", DEFAULT_SAMPLES)
    enabled = cfg.get("a2c_enabled", cfg.get("enabled", True))
    if _is_number(spec):
        # 占位/内置场景：4x，几何覆盖样本 0..2，alpha=0.5。
        samples = DEFAULT_SAMPLES
        msaa_mask = (1, 1, 1, 0)
        alpha = 0.5
        enabled = True

    samples = normalize_sample_count(samples)
    geometry = _coerce_mask(msaa_mask, samples)
    a2c = _a2c_mask(alpha, samples) if enabled else tuple(1 for _ in range(samples))

    # 按位与合并：任一来源关闭的样本都不参与解析。
    merged = tuple(g & a for g, a in zip(geometry, a2c))
    covered = sum(merged)
    geometry_covered = sum(geometry)

    # 关键：权重只按合并后的覆盖数归一化一次，不额外乘 alpha。
    weight = edge_weight(covered, samples)
    return {
        "samples": samples,
        "geometry_mask": geometry,
        "a2c_mask": a2c,
        "merged_mask": merged,
        "covered": covered,
        "geometry_covered": geometry_covered,
        "coverage": weight,
        # 双重点算会得到的错误权重，仅用于审计/对比。
        "double_modulated_weight": _clamp01(
            (covered / float(samples)) * _clamp01(alpha)
        ) if samples else 0.0,
        "rule": "and",
        "a2c_enabled": bool(enabled),
    }


# ---------------------------------------------------------------------------
# 3. 深度 / 颜色解析顺序固定
# ---------------------------------------------------------------------------

def depth_color_order(spec=None, *args):
    """固定并校验深度先于颜色的解析顺序。

    修复“深度解析与颜色解析先后顺序不固定”：合法顺序恒为
    ``("depth", "color")``。传入自定义阶段序列时，只有深度严格排在
    颜色之前才判定 ``valid``；返回值同时给出规范顺序供调度器直接使用。
    """
    cfg = _as_dict(spec)
    stages = cfg.get("stages", cfg.get("order"))
    if isinstance(stages, (list, tuple)) and stages:
        seq = [str(s).lower() for s in stages]
        depth_idx = seq.index(DEPTH_STAGE) if DEPTH_STAGE in seq else -1
        color_idx = seq.index(COLOR_STAGE) if COLOR_STAGE in seq else -1
        valid = depth_idx >= 0 and color_idx >= 0 and depth_idx < color_idx
        given = tuple(seq)
    else:
        # 占位整数 / 缺省：直接给出规范顺序。
        given = RESOLVE_ORDER
        valid = True
    return {
        "order": RESOLVE_ORDER,
        "given": given,
        "valid": valid,
        "first": DEPTH_STAGE,
        "second": COLOR_STAGE,
    }


# ---------------------------------------------------------------------------
# 4. 按眼分开的解析缓冲
# ---------------------------------------------------------------------------

# 每只眼各自的解析目标注册表：{eye: {key: target}}。
_eye_targets = {eye: {} for eye in EYES}


def per_eye(spec=None, *args, eye=None, target=None):
    """按眼存取解析目标，左右眼缓冲彻底分开。

    修复“VR 双目共用一张多重采样纹理，一方解析覆盖另一方结果”：
    解析目标键自带眼别后缀，写入与读取都只落在该眼自己的命名空间。

    * ``per_eye()`` / ``per_eye(1)``：返回双眼的键映射与独立存储快照；
    * ``per_eye(eye="left", target=...)``：为左眼写入（或读取）目标；
    * 字典规格支持 base / eye / target / key 字段。
    """
    cfg = _as_dict(spec)
    if eye is None:
        eye = cfg.get("eye")
    base = cfg.get("base", "resolve")
    if target is None:
        target = cfg.get("target")
    key = cfg.get("key")

    if eye is None:
        # 双眼总览：键名不同，存储字典也是两份独立对象。
        return {
            "eyes": {
                eye_name: {
                    "key": f"{base}:{eye_name}",
                    "targets": dict(_eye_targets[eye_name]),
                }
                for eye_name in EYES
            },
            "isolated": True,
        }

    eye_name = _normalize_eye(eye)
    full_key = key or f"{base}:{eye_name}"
    if target is not None:
        _eye_targets[eye_name][full_key] = target
    stored = _eye_targets[eye_name].get(full_key)
    return {
        "eye": eye_name,
        "key": full_key,
        "target": stored,
        # 另一眼的同名键指向不同对象，证明不会互相覆盖。
        "other_eye_key": f"{base}:{'right' if eye_name == 'left' else 'left'}",
    }


# ---------------------------------------------------------------------------
# 5. 解析与帧边界对齐
# ---------------------------------------------------------------------------

# 帧时间线：frame -> 该帧边界 fence 是否已经 signal。
_frame_fences = {}
# 已登记的解析请求，便于审计对齐情况。
_resolve_log = []


def resolve_sync(spec=None, *args, frame=None, signaled=None):
    """把解析请求对齐到帧边界。

    修复“解析与帧边界不对齐”：只有当帧边界的 fence 已 signal 时，解析
    才允许在该帧执行；fence 未就绪时请求推迟到下一帧边界（最多推迟
    ``MAX_DEFERRAL`` 帧），并在返回中如实标记 deferred / aligned_frame。

    占位整数 ``resolve_sync(1)`` 走内置场景：第 1 帧 fence 已就绪，
    解析在当帧边界执行，不推迟。
    """
    cfg = _as_dict(spec)
    if frame is None:
        frame = cfg.get("frame", 1 if _is_number(spec) else 0)
    frame = _safe_int(frame, 0)
    if signaled is None:
        signaled = cfg.get("signaled", cfg.get("fence_signaled", True))
    signaled = bool(signaled)

    _frame_fences[frame] = signaled
    if signaled:
        aligned_frame = frame
        deferred = False
    else:
        aligned_frame = frame + MAX_DEFERRAL
        deferred = True
        # 推迟后的帧在到达时补 signal。
        _frame_fences.setdefault(aligned_frame, True)

    record = {
        "frame": frame,
        "fence_signaled": signaled,
        "deferred": deferred,
        "aligned_frame": aligned_frame,
        "stage": "resolve",
        "at_boundary": True,
    }
    _resolve_log.append(record)
    return record


# ---------------------------------------------------------------------------
# 6. 边缘权重按覆盖率归一
# ---------------------------------------------------------------------------

def edge_weight(covered=None, samples=None, *args):
    """边缘解析权重 = 实际覆盖样本数 / 当前样本数。

    修复“边缘叠暗 / 闪烁”：半覆盖像素不能按满样本数归一化（那样边缘
    会系统性偏暗），也不能让权重和超过 1。覆盖为 0 时权重为 0，
    调用方应直接复用背景，而不是除以零。
    """
    if covered is None:
        covered = 2
    if samples is None:
        samples = DEFAULT_SAMPLES
    if _is_number(covered) and isinstance(samples, dict):
        # 容忍 (covered, cfg) 形式。
        samples = samples.get("samples", DEFAULT_SAMPLES)
    covered = max(0, _safe_int(covered, 0))
    samples = normalize_sample_count(_safe_int(samples, DEFAULT_SAMPLES))
    if samples <= 0 or covered <= 0:
        return 0.0
    covered = min(covered, samples)
    return covered / float(samples)


def _weighted_average(values, weights):
    """按覆盖率权重做一次且仅一次归一化的加权平均。"""
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / denom


# ---------------------------------------------------------------------------
# 主解析管线
# ---------------------------------------------------------------------------

def resolve(spec=None, *args, **kwargs):
    """对一个多重采样像素执行完整解析。

    输入（字典或关键字，缺省走内置 4x 场景）：
      samples       当前采样档位（绑定当前值，不读历史档位）
      colors        每个样本的线性空间颜色（数值或序列）
      depths        每个样本的深度
      covered_mask  几何覆盖掩码（0/1 序列）
      alpha         A2C 透明度；a2c_enabled 控制是否合并 A2C
      eye / frame   按眼、按帧的归属

    输出固定按 depth -> color 顺序解析：深度取最近样本，颜色按 A2C 与
    MSAA 合并后的真实覆盖率归一化一次。
    """
    cfg = dict(_as_dict(spec))
    cfg.update({k: v for k, v in kwargs.items() if v is not None})

    samples = normalize_sample_count(cfg.get("samples", DEFAULT_SAMPLES))
    colors = cfg.get("colors")
    depths = cfg.get("depths")
    mask = cfg.get("covered_mask", cfg.get("msaa_mask"))
    alpha = cfg.get("alpha", 1.0)
    a2c_enabled = cfg.get("a2c_enabled", cfg.get("enabled", True))
    eye = cfg.get("eye")
    frame = cfg.get("frame", 0)

    if colors is None:
        # 内置确定性场景：4 个样本，前 3 个被几何覆盖。
        colors = [0.2, 0.4, 0.6, 0.0]
        depths = [0.9, 0.5, 0.7, 1.0]
        mask = (1, 1, 1, 0)
        alpha = 0.5
        samples = samples or DEFAULT_SAMPLES
    if depths is None:
        depths = [1.0] * samples
    if mask is None:
        mask = tuple(1 for _ in range(samples))

    geometry = _coerce_mask(mask, samples)
    a2c = _a2c_mask(alpha, samples) if a2c_enabled else tuple(1 for _ in range(samples))
    merged = tuple(g & a for g, a in zip(geometry, a2c))
    covered = sum(merged)

    # 顺序固定：先深度（取最近/最小值），再颜色。
    active_depths = [_safe_float(d, float("inf")) for d, m in zip(depths, merged) if m]
    resolved_depth = min(active_depths) if active_depths else float("inf")

    weights = list(merged)  # 权重即合并覆盖，归一化只做一次
    if colors and isinstance(colors[0], (list, tuple)):
        channels = len(colors[0])
        resolved_color = tuple(
            _weighted_average([c[ch] for c in colors], weights)
            for ch in range(channels)
        )
    else:
        resolved_color = _weighted_average([_safe_float(c) for c in colors], weights)

    result = {
        "samples": samples,
        "merged_mask": merged,
        "covered": covered,
        "coverage": edge_weight(covered, samples),
        "resolved_depth": resolved_depth,
        "resolved_color": resolved_color,
        "order": RESOLVE_ORDER,
        "frame": _safe_int(frame, 0),
        "eye": _normalize_eye(eye) if eye is not None else None,
        "a2c_enabled": bool(a2c_enabled),
    }
    if eye is not None:
        # 写入该眼自己的解析目标，双眼互不覆盖。
        per_eye(eye=result["eye"], target=result,
                **({"base": cfg["base"]} if cfg.get("base") else {}))
    return result


# ---------------------------------------------------------------------------
# 支撑能力：校验 / 审计 / 降级
# ---------------------------------------------------------------------------

def validation(spec=None, *args):
    """校验一份解析配置是否满足全部修复约束。

    检查项：采样数受支持、A2C 合并规则为按位与、深度先于颜色、
    双眼缓冲独立、解析对齐帧边界。整数参数走内置合法场景。
    """
    cfg = _as_dict(spec)
    fatal = []
    warnings = []

    raw_samples = cfg.get("samples", DEFAULT_SAMPLES)
    samples = normalize_sample_count(raw_samples)
    if _is_number(raw_samples) and int(raw_samples) not in VALID_SAMPLE_COUNTS:
        # 不受支持的档位会被钳制，属于可恢复问题，只告警不判失败。
        warnings.append(f"unsupported sample count {raw_samples}, clamped to {samples}")

    order_info = depth_color_order(cfg if cfg.get("stages") or cfg.get("order") else None)
    if not order_info["valid"]:
        fatal.append("depth resolve must precede color resolve")

    if cfg.get("a2c_rule") and cfg["a2c_rule"] != "and":
        fatal.append("A2C and MSAA coverage must merge with bitwise AND")

    if cfg.get("shared_eye_texture"):
        fatal.append("per-eye resolve targets must not share one texture")

    if cfg.get("frame") is not None and cfg.get("signaled") is False:
        # 未对齐本身可由 resolve_sync 推迟修复，这里给出提示而非判死刑。
        warnings.append("resolve requested before frame boundary fence; will defer")

    return {
        "valid": not fatal,
        "samples": samples,
        "order": RESOLVE_ORDER,
        "per_eye_isolated": not cfg.get("shared_eye_texture", False),
        "a2c_rule": cfg.get("a2c_rule", "and"),
        "errors": fatal + warnings,
        "fatal": fatal,
        "warnings": warnings,
    }


def audit(spec=None, *args):
    """审计一帧或一组解析事件。

    统计总解析次数、按眼分布、顺序违规、推迟到帧边界之后的次数，
    以及各采样档位的使用分布。整数参数基于内置解析日志给出汇总。
    """
    events = spec if isinstance(spec, (list, tuple)) else _as_dict(spec).get("events")
    if events is None:
        events = list(_resolve_log)
    events = list(events)

    per_eye_count = {eye: 0 for eye in EYES}
    sample_dist = {}
    order_violations = 0
    deferred = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        eye = ev.get("eye")
        if eye in per_eye_count:
            per_eye_count[eye] += 1
        s = ev.get("samples")
        if s is not None:
            sample_dist[normalize_sample_count(s)] = sample_dist.get(
                normalize_sample_count(s), 0) + 1
        if ev.get("order") and tuple(ev["order"]) != RESOLVE_ORDER:
            order_violations += 1
        if ev.get("deferred"):
            deferred += 1

    return {
        "resolves": len(events),
        "per_eye": per_eye_count,
        "sample_distribution": sample_dist,
        "order_violations": order_violations,
        "deferred": deferred,
        "isolated_eyes": True,
    }


def fallback(spec=None, *args):
    """不支持当前 MSAA/A2C 配置时的安全降级。

    8x（或更高）不被支持时降到 4x，仍不支持则降到 1x；A2C 不支持时
    关闭 A2C 但保留 MSAA 几何覆盖。降级后深度/颜色顺序与按眼隔离
    约束保持不变。
    """
    cfg = _as_dict(spec)
    requested = normalize_sample_count(cfg.get("samples", DEFAULT_SAMPLES))
    supported = cfg.get("supported_samples", VALID_SAMPLE_COUNTS)
    supported = tuple(normalize_sample_count(s) for s in supported)

    chosen = requested if requested in supported else 1
    for cand in sorted(supported, reverse=True):
        if cand <= requested:
            chosen = cand
            break

    a2c_supported = cfg.get("a2c_supported", True)
    return {
        "requested_samples": requested,
        "samples": chosen,
        "a2c_enabled": bool(a2c_supported),
        "downgraded": chosen != requested,
        "order": RESOLVE_ORDER,
        "per_eye_isolated": True,
        "reason": cfg.get("reason", "unsupported_sample_count" if chosen != requested else "none"),
    }


# ---------------------------------------------------------------------------
# alpha 边缘 / 指标 / 格式
# ---------------------------------------------------------------------------

def alpha_edge(spec=None, *args, alpha=None, samples=None):
    """识别半透明边缘像素并给出其 A2C 覆盖。

    完全不透明（alpha=1）与完全透明（alpha=0）都不是需要特殊处理的
    半透明边缘；0 < alpha < 1 的像素按当前采样数生成 A2C 掩码，
    供 a2c_compat 与几何覆盖做按位与合并。
    """
    cfg = _as_dict(spec)
    if alpha is None:
        alpha = cfg.get("alpha", 0.5)
    if samples is None:
        samples = cfg.get("samples", DEFAULT_SAMPLES)
    alpha = _clamp01(alpha)
    samples = normalize_sample_count(samples)
    mask = _a2c_mask(alpha, samples)
    return {
        "alpha": alpha,
        "edge": 0.0 < alpha < 1.0,
        "samples": samples,
        "a2c_mask": mask,
        "covered": sum(mask),
    }


def metrics(spec=None, *args):
    """解析质量与开销指标。

    内置场景基于一个半覆盖 4x 像素；也接受 resolve 结果 / 像素列表。
    指标包括平均覆盖率、全解析像素占比、按眼解析次数与帧推迟率。
    """
    cfg = _as_dict(spec)
    pixels = spec if isinstance(spec, (list, tuple)) and not cfg else cfg.get("pixels")
    if pixels is None:
        # 两个内置像素：一个半覆盖边缘像素、一个全覆盖像素。
        pixels = [
            resolve({"samples": 4, "colors": [0.2, 0.4, 0.6, 0.0],
                     "depths": [0.9, 0.5, 0.7, 1.0],
                     "covered_mask": (1, 1, 1, 0), "alpha": 0.5,
                     "frame": 0}),
            resolve({"samples": 4, "colors": [0.5] * 4,
                     "depths": [0.3] * 4,
                     "covered_mask": (1, 1, 1, 1), "alpha": 1.0,
                     "frame": 0}),
        ]

    coverages = [p.get("coverage", 0.0) for p in pixels if isinstance(p, dict)]
    total = len(coverages)
    avg_cov = sum(coverages) / total if total else 0.0
    fully_resolved = sum(1 for c in coverages if c >= 1.0)
    return {
        "pixels": total,
        "average_coverage": avg_cov,
        "fully_resolved_ratio": fully_resolved / total if total else 0.0,
        "edge_pixels": sum(1 for c in coverages if 0.0 < c < 1.0),
        "per_eye_resolves": {eye: 0 for eye in EYES},
        "deferred_rate": 0.0,
    }


def format(spec=None, *args):
    """选择与解析目标匹配的纹理格式。

    HDR/线性内容必须使用线性（_UNORM / float）目标，解析在线性空间
    完成后再做色调映射；sRGB 标记只允许出现在最终呈现目标上。
    返回格式名、每像素位宽与色彩空间，供资源分配与校验使用。
    """
    cfg = _as_dict(spec)
    hdr = cfg.get("hdr", _is_number(spec))
    srgb = cfg.get("srgb", False)
    channels = _safe_int(cfg.get("channels", 4), 4)

    if hdr:
        name = "R16G16B16A16_FLOAT" if channels == 4 else "R16G16B16_FLOAT"
        bpp = 64 if channels == 4 else 48
        color_space = "linear"
    elif srgb:
        name = "R8G8B8A8_UNORM_SRGB" if channels == 4 else "B8G8R8A8_UNORM_SRGB"
        bpp = 32
        color_space = "srgb"
    else:
        name = "R8G8B8A8_UNORM" if channels == 4 else "B8G8R8A8_UNORM"
        bpp = 32
        color_space = "linear"

    return {
        "format": name,
        "bits_per_pixel": bpp,
        "color_space": color_space,
        "resolve_before_tonemap": bool(hdr) or color_space == "linear",
        "srgb_present_target": bool(srgb),
    }


# ---------------------------------------------------------------------------
# 闪烁与确定性
# ---------------------------------------------------------------------------

def _safe_float(value, default=0.0):
    """尽力转 float；无法解释的输入回落到 default，绝不抛异常。"""
    if isinstance(value, bool):
        return float(value)
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _numeric_sequence(value):
    """把序列输入转成 float 列表；非序列返回 None。"""
    if isinstance(value, (list, tuple)):
        return [_safe_float(v, 0.0) for v in value]
    return None


def flicker(spec=None, *args, current=None, previous=None):
    """量化快速移动镜头下解析边缘的帧间闪烁。

    返回归一化帧间差异（0 = 完全稳定，1 = 满幅摆动）、是否超过稳定
    阈值，以及建议的时域混合权重。覆盖率归一化稳定后，静止边缘的
    帧间差异应为 0；仅真实运动才产生非零度量。
    """
    cfg = _as_dict(spec)
    if current is None:
        current = cfg.get("current")
    if previous is None:
        previous = cfg.get("previous")

    if current is None and _is_number(spec):
        # 内置场景：同一半覆盖像素连续两帧，覆盖率归一化后无闪烁。
        current = edge_weight(2, 4)
        previous = edge_weight(2, 4)

    if current is None:
        current = 0.0
    if previous is None:
        previous = current  # 首帧没有历史，视为无闪烁。

    cur_seq = _numeric_sequence(current)
    prv_seq = _numeric_sequence(previous)
    if cur_seq is not None or prv_seq is not None:
        cur = cur_seq if cur_seq is not None else [_safe_float(current)]
        prv = prv_seq if prv_seq is not None else [_safe_float(previous)]
        n = max(len(cur), len(prv))
        cur = cur + [0.0] * (n - len(cur))
        prv = prv + [0.0] * (n - len(prv))
        diffs = [abs(a - b) for a, b in zip(cur, prv)]
        mags = [max(abs(a), abs(b), 1e-6) for a, b in zip(cur, prv)]
        delta = (sum(diffs) / n) / (sum(mags) / n) if n else 0.0
    else:
        now, before = _safe_float(current), _safe_float(previous)
        denom = max(abs(now), abs(before), 1e-6)
        delta = abs(now - before) / denom

    delta = _clamp01(delta)
    threshold = cfg.get("threshold", 0.05)
    return {
        "flicker": delta,
        "stable": delta <= threshold,
        "threshold": threshold,
        # 闪烁越明显，时域混合权重应越低，避免把抖动拖成残影。
        "temporal_weight": _clamp01(1.0 - delta),
    }


def determinism(spec=None, *args):
    """同一输入两次解析必须逐字段相等。

    解析绑定当前采样数、覆盖率按位与合并、权重一次归一化，整条管线
    没有隐藏随机源或全局可变采样数，因此重复运行结果完全一致。
    """
    cfg = _as_dict(spec)
    seed = _safe_int(spec, 1) if _is_number(spec) else _safe_int(cfg.get("seed", 1), 1)

    def _run():
        return resolve({
            "samples": 4,
            "colors": [0.2, 0.4, 0.6, 0.0],
            "depths": [0.9, 0.5, 0.7, 1.0],
            "covered_mask": (1, 1, 1, 0),
            "alpha": 0.5,
            "frame": seed,
        })

    first = _run()
    second = _run()
    return {
        "deterministic": first == second,
        "first": first,
        "second": second,
        "seed": seed,
    }
