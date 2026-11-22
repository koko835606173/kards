# KARDS 副官（kards_adjutant）

[N.E.K.O](https://github.com/Project-N-E-K-O/N.E.K.O) 的 KARDS 游戏副官插件：
回合内一键斩杀计算。截取当前屏幕 → 视觉模型**只转录**局面 → **纯代码穷举**本回合
全部出牌线 → 结论交给角色用自己的语气转述。模型只负责读数和说话，游戏规则全部
落在可单测的确定性代码里。

## 功能

- **斩杀计算**：能否斩杀／最大总部伤害／最优出牌路线／建议先清哪些单位
- 规则引擎覆盖：前线占领、守护相邻、长程单位、伏击先手、冲击、重甲、烟幕、
  奋战、操作费预算、坦克移动+攻击、轰炸机拦截、手牌直伤指令……每条规则的
  出处写在 [kards_adjutant/lethal_solver.py](kards_adjutant/lethal_solver.py) 注释里
- **牌库增强**（可选）：导入 [kardscm](https://github.com/AABur/kardscm) 导出的
  全量卡表后，按名称补全读不清的基础数值（类型/操作费/关键词），画面实况
  （血量/位置/增益）永远以截图为准 —— 见 [kards_adjutant/docs/CARDS.md](kards_adjutant/docs/CARDS.md)
- 读不清的数字标把握度、没有的数值不猜、未知关键词明示，结论附免责清单

## 依赖

- 运行于 N.E.K.O 插件宿主（SDK v2）
- 一个绑定的**视觉模型**（`lethal_vision` 槽，需 image_input 能力）
- 卡片数据库（可选）与卡图版权归 1939 Games 所有，本仓库**不携带任何卡牌
  数据**，由使用者本地生成、本地使用（个人非商用）

## 安装（在 N.E.K.O 仓库内）

把 `kards_adjutant/` 放到 `plugin/plugins/` 下，`tests/` 对应
`plugin/tests/unit/plugins/`，然后在插件管理里启用并给「斩杀计算视觉模型」槽
绑定模型。

## 使用

对角色说「算一下斩杀」或点快捷按钮。插件会自动截屏并推送结论。

## 测试

测试需要在 N.E.K.O 仓库的路径结构下运行（依赖 `plugin.sdk`）：

```
uv run python -m pytest plugin/tests/unit/plugins/test_lethal_solver.py plugin/tests/unit/plugins/test_kards_adjutant.py
```

## License

Apache-2.0（与 N.E.K.O 主仓库一致）。KARDS 及其卡牌名称、美术、商标归
1939 Games ehf 所有；本项目为非官方粉丝工具，与 1939 Games 无关联。
