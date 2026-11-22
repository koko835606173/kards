# 牌库（cards.json）生成与使用

KARDS 副官的视觉读数在**没有牌库时也能工作**（完全凭画面），但有牌库后可以：

- 把「类型/操作费/关键词/重甲」读不清的单位**救回来**（按名称匹配补全基础数值）
- 补全手牌直伤指令的费用
- 不覆盖画面实况：血量、位置、增益后的攻击力永远以画面为准

## 为什么不随仓库分发卡牌数据

KARDS 卡牌数据归 1939 Games 所有；kardscm 项目自己也坚持"仓库不带卡牌数据"
（个人非商用用途）。因此本插件**不提交** cards.json，由主人本地生成、本地使用。

## 生成步骤（一次性，约 2 分钟）

```powershell
git clone https://github.com/AABur/kardscm.git <任意目录>\kardscm
cd <任意目录>\kardscm
uv sync

# api.kards.com 在部分地区会被阻断；失败时先设置代理再重试：
#   $env:HTTPS_PROXY = "http://127.0.0.1:你的代理端口"
uv run kardscm sync --yes

uv run kardscm export -f json -o cards.json
```

导出的 JSON 是 kardscm 的「raw API-shape dump」：`title`/`text` 为多语言对象
（含中文），字段含 `type` / `kredits`（部署费）/ `operationCost`（操作费）/
`attack` / `defense` / `attributes`（关键词）等，与本插件的
[card_db.py](../card_db.py) 直接对接。

## 放置位置

把 `cards.json` 放进本插件的**数据目录**：

```
<NEKO 插件数据根>/kards_adjutant/data/cards.json
```

（即插件 SDK `data_path("cards.json")` 解析到的路径。）

放好后无需重启：下一次「斩杀计算」会自动懒加载；结论末尾会出现
「牌库：已启用（N 张卡）」一行。文件缺失 = 维持纯视觉模式；文件存在但损坏
会在结论末尾提示加载失败原因，不阻断计算。

## 已知边界

- kardscm 已知会丢弃部分官方 API 未暴露、仅客户端可见的属性
  （如 BecomesVeteran），并以手工维护的 extra_abilities 部分弥补；
  未知关键词在结论里以「未知关键词按无效果计」提示。
- 卡牌更新（新扩展包）后重跑一次 `sync + export` 并覆盖同名文件即可。
