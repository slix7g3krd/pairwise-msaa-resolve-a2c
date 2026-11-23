# pairwise-msaa-resolve-a2c

## 领域

游戏、图形与多媒体

## 主题

MSAA 解析与 A2C

## 说明

样本数绑定、路径兼容与按眼解析。

初始状态下 `graphics/msaa_resolve_a2c.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
