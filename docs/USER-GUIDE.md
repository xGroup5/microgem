# microGEM 使用说明书

**版本 1.0.0** ｜ 微生物组源代谢酶打分系统 ｜ 中文版

> English: [USER-GUIDE.en.md](USER-GUIDE.en.md)

---

## 一、这是什么

microGEM 回答一个问题：**给定一个没人表征过的蛋白序列，它是否属于这 17 类酶化学之一？**

它不是数据库检索工具。UniProt、BRENDA、MetaCyc、KEGG 回答"这个已表征的酶是什么"；
microGEM 回答"这个未表征的蛋白**可能是**什么，值不值得做实验"。宏基因组里 60–80% 的预测
ORF 没有可靠注释，这些数据库对它们完全沉默——microGEM 就是为这一块设计的。

**模型极简**：冻结的 ESM-2 650M 蛋白语言模型 → 1280 维向量 → 标准化 → 逻辑回归 → sigmoid。
17 个面板共用同一份 embedding，各自一个线性探针。全部权重 244 KB，推理只需要 NumPy。

**评估极严**：所有性能数字都来自**簇感知交叉验证**——正负样本一起聚类（bitscore 200），
整簇分到同一折，杜绝同家族跨训练/测试集泄漏。比多数酶挖掘论文报告的随机切分严格得多，
这也是一些数字看起来偏低的原因。

---

## 二、安装

### 2.1 最短路径

```bash
git clone <repo> && cd microgem-skill
./install.sh
```

安装脚本做两件互相独立的事：

1. **装命令**：把 `microgem` 装进 Python 环境（默认 `pip install --user`）
2. **装技能**：把 `skill/` 复制到 AI 助手的技能目录，让助手知道什么时候、怎么调用

### 2.2 安装选项

| 选项 | 作用 |
|---|---|
| `--cli-only` | 只装命令，不装技能 |
| `--skill-only` | 只装技能，不装命令 |
| `--venv PATH` | 建一个独立虚拟环境再装（推荐，避免污染系统 Python） |
| `--agent NAME` | 指定技能目录：`dsh` \| `claude` \| `agents` \| 任意路径 |
| `--download-model` | 顺手下载 2.6 GB 的 ESM-2 权重 |
| `--with-torch` | 顺手装 PyTorch + fair-esm（GPU 快 5–15 倍） |
| `--dry-run` | 只打印要做什么，不落盘 |

### 2.3 手动安装（不用脚本）

```bash
pip install .                       # 只装命令，网络好的话 30 秒
# 或者指定虚拟环境
python3 -m venv ~/.venvs/microgem
~/.venvs/microgem/bin/pip install .
```

技能安装就是把目录复制过去：

```bash
# DSH
cp -R skill ~/.dsh/skills/microgem
# Claude Code
cp -R skill ~/.claude/skills/microgem
# 通用 agents 目录
cp -R skill ~/.agents/skills/microgem
```

### 2.4 模型权重（必须，一次性）

模型 2.6 GB，**不随包分发**。三种方式任选：

```bash
# ① 自动下载（需要先 pip install torch fair-esm）
microgem download

# ② 已经有 .pt 文件，指过去就行
export MICROGEM_ESM2=/path/to/esm2_t33_650M_UR50D.pt

# ③ 只想要 torch-free 路径：转换一次
microgem download --convert     # 之后推理不再需要 torch
```

**没有 GPU 也能用**。两条推理路径：

| 路径 | 依赖 | 速度 | 说明 |
|---|---|---|---|
| `torch` | torch + fair-esm + .pt | GPU 0.2 秒/条，CPU 约 1.5 秒/条 | 快，推荐 |
| `numpy` | 只要 numpy | CPU 约 3 秒/条 | 慢但轻，笔记本可跑 |

两条路径给出**同一个向量**（实测最大偏差 9.5e-7）。`microgem convert` 生成 NumPy 副本后，
可以卸载 torch。

---

## 三、五分钟上手

### 3.1 确认装好了

```bash
microgem info
```

```
microGEM 1.0.0
  embedding backend : torch
  model cache       : /home/<user>/.cache/microgem
  ESM-2 checkpoint  : /home/<user>/models/esm2/esm2_t33_650M_UR50D.pt
  NumPy weights     : .../esm2_t33_650M_UR50D_numpy (not converted)
  panels            : 17
```

`backend: none` 表示还缺模型，回第二章第 4 节。

### 3.2 自检（约 15 秒 GPU / 2.5 分钟 CPU）

```bash
microgem selftest
```

```
worst deviation: 4.95e-07 (AATE-lyase|Q13YI7/AATE-lyase)   tolerance 1e-03
cross-panel false positives: 27/752 (3.6%)
PASS — this installation reproduces the exported model
```

这一步验证**你这个安装算出来的分数，和导出模型的机器上算出来的一致**。47 条参考蛋白覆盖
17 个面板，每条都有一个"该面板阳性对照"或"未饱和的漂移探针"。改了 embedding、换了错权重、
下载被截断——都会在这里失败。

### 3.3 打分

```bash
microgem score proteins.faa
```

```
microGEM 1.0.0 | 17 sequences | 17 panels | per-panel calibrated cut-offs

> FATE-L-modifying|Q96GR2  (724 aa)
    CALL FATE-L-modifying  1.000 [####################] reliable AUPRC 0.894
    CALL BA                0.959 [###################.] reliable AUPRC 0.959
    CALL FATE-S            0.702 [##############......] reliable AUPRC 0.891
         ARO               0.063 [#...................] reliable AUPRC 0.890
         ... 13 further panels, all <= 0.015 (--verbose for all)
    -> assigned: FATE-L-modifying  (Fatty Acyl Transformation Enzyme — Long chain)
    -> also above cut-off: FATE-S, BA
```

每一行读法：**是否判定为阳性 ｜ 面板名 ｜ 概率 ｜ 条形图 ｜ 可靠度 ｜ 该面板 AUPRC**。

### 3.4 三种输出格式

```bash
microgem score proteins.faa                    # 人读
microgem score proteins.faa --format tsv       # 表格，Excel/R 直接读
microgem score proteins.faa --format json      # 下游程序用（推荐给 AI 助手）
```

JSON 每个序列包含：`id`、`length`、`sha1`、`called`（过了切点的面板）、`best`、
`best_probability`、`margin`、`scores`（全部 17 个概率）。

### 3.5 Python 调用

```python
from microgem import score, PANELS

result = score("proteins.faa")                    # 全部 17 个面板
result = score(["MKVL...", "MSQE..."])            # 序列字符串也行
result = score({"my_protein": "MKVL..."})         # 自己命名

for item in result:
    best = item.best
    print(f"{item.identifier}: {best.panel} {best.probability:.3f} "
          f"(tier {best.tier}, margin {item.margin:.3f})")
    print("  全部判定:", item.called)

result.to_json()      # 完整报告
```

---

## 四、17 个面板

### 4.1 面板一览

| 面板 | 家族 | 判的是什么 | AUPRC | 可靠度 |
|---|---|---|---|---|
| `BA` | 胆汁酸 | 胆汁酸为底物 | 0.959 | 可靠 |
| `AATE-ligase` | 氨基酸 | ATP 依赖成键（氨酰-tRNA 合成酶等） | 0.925 | 可靠 |
| `PLY` | 多胺 | 腐胺/亚精胺/精胺代谢 | 0.919 | 可靠 |
| `RDX` | 氧化应激 | 谷胱甘肽/硫氧还蛋白/过氧化物酶 | 0.909 | 可靠 |
| `FATE-L-releasing` | 长链脂肪酸 | 脂质酯水解、释放长链脂肪酸 | 0.899 | 可靠 |
| `FATE-L-modifying` | 长链脂肪酸 | 游离长链脂肪酸或其 CoA 被修饰 | 0.894 | 可靠 |
| `FATE-S` | 短链脂肪酸 | C2–C6 脂肪酸/酰基-CoA | 0.891 | 可靠 |
| `ARO` | 芳香族 | 芳香环裂解与修饰 | 0.890 | 可靠 |
| `AATE-hydrolysis` | 氨基酸 | 水解型（脱氨酶、酰胺酶） | 0.871 | 可靠 |
| `GLY` | 糖 | 糖苷键水解（对应 CAZy 的 GH 类） | 0.863 | 可靠 |
| `SATE` | SAM | SAM 转移甲基（产 SAH） | 0.861 | 可靠 |
| `VIT` | 维生素 | 维生素/辅因子从头合成 | 0.819 | 中等 |
| `TRP` | 色氨酸 | 色氨酸→吲哚/AhR 配体分支 | 0.783 | 中等 |
| `AATE-redox` | 氨基酸 | 氧化还原型（脱氢酶、氧化酶） | 0.751 | 中等 |
| `AATE-lyase` | 氨基酸 | 裂解/异构型（脱羧酶、转氨酶） | 0.715 | 中等 |
| `SALE` | SAM | SAM 被消耗但不转移甲基（自由基 SAM） | 0.628 | **弱** |
| `AATE-CoA` | 氨基酸 | 酰基-CoA 依赖的 N-酰化 | 0.503 | **弱** |

可靠度按 AUPRC 分档：≥0.85 可靠 / 0.65–0.85 中等 / <0.65 弱。

### 4.2 面板怎么选的

每个面板由**酶作用于什么**定义，不按 EC 号。后果有两个：同一条 EC 下的酶可能分到不同面板
（化学不同），一个面板也可能横跨多个 EC。

几个关键裁决：

- **FATE-L 拆成两个**：底物侧是游离脂肪酸（modifying）还是脂质酯、脂肪酸是产物
  （releasing）——这是两种相反的化学。拆之前 AUPRC 0.607，拆后 0.894 / 0.899。
- **SATE 拆出 SALE**：结合 SAM 但不转移甲基的（自由基 SAM）是另一个结构家族。不拆时
  40.8% 的正样本得分低于 0.5。
- **FATE-S 剔除醛脱氢酶**：乙醛脱氢酶的反应从**醛**开始，不是脂肪酸。正则误匹配
  `acetaldehyde` 把 111 条全判错，去掉后 AUPRC 0.752 → 0.901。
- **AATE 按反应化学拆五类**：348 个反应里 131 个没有 EC 号，按 EC 拆会丢 40%。

完整定义和裁决理由见 `skill/references/panels.md`。

### 4.3 面板之间的优先级

一个反应同时符合多个面板时，按下面的顺序抢占，被抢走的面板就看不到它：

```
BA > GLY > SALE > SATE > PLY > TRP > ARO > VIT > RDX
   > FATE-L-modifying > FATE-L-releasing > FATE-S
   > AATE-CoA > AATE-ligase > AATE-lyase > AATE-redox > AATE-hydrolysis
```

45 组两两交叉，反应重叠为 0。这是多标签训练成立的前提。

---

## 五、怎么读分数（最重要的一章）

### 5.1 概率不是"后验概率"

面板用 `class_weight="balanced"` 在约 10% 阳性的训练集上拟合，输出更像**似然比**而不是
概率。直接后果：**同一个 0.5，在不同数据上对应完全不同的精确率。**

| 面板 | 切点 | 交叉验证精确率 | 假定阳性率 0.1% 时的精确率 |
|---|---|---|---|
| SATE | 0.30 | 0.900 | 0.075 |
| AATE-ligase | 0.60 | 0.882 | 0.064 |
| TRP | 0.40 | 0.723 | 0.020 |
| GLY | 0.20 | 0.833 | 0.045 |

第三列才是"把整个宏基因组丢进去"时你会看到的数字。第二列不是骗人，它描述的是交叉验证集
（约 10% 阳性）上的表现。

想要按你的数据算：

```bash
microgem score proteins.faa --prevalence 0.001 --show-operating-points
```

### 5.2 面板之间不能直接比大小

`SALE` 上的 0.85 比 `GLY` 上的 0.65 **更弱**。两个探针训练时用的负样本集不同、类别比例不同，
输出不在同一把尺子上。能比的是**可靠度档位**，以及同一序列内的排名。

排序时按：**① 可靠度档 > ② 概率 > ③ margin**。

### 5.3 margin 小于 0.1 时不要下结论

margin = 最高分 − 第二高分。小于 0.1 意味着两个探针同时被激活，模型分不开。三种常见情况：

- `FATE-L-modifying` + `FATE-L-releasing` 都高：蛋白卡在"酯/游离酸"边界，底物广的酯酶就是这样
- 相邻两个 `AATE-*` 面板：反应落在化学拆分的边界上
- `SATE` + `SALE`：结合 SAM，但产物化学在表示里不明确

**两个都报**，说明模型分不开。不要偷偷挑一个。

### 5.4 弱面板要提前说明

- **`AATE-CoA`**：57.6% 的正样本来自同一个反应（EC 2.3.1.1），succinyl-CoA 亚类每一折都判错，
  30 种子 AUPRC 0.62 ± 0.068。**不能用来筛候选。**
- **`SALE`**：标签化学上自洽，但正样本横跨多个不相关折叠。高分只意味着"SAM 被消耗但不转移甲基"。

用户专门问这两个方向时，先说明面板弱，再给数字。

### 5.5 RDX 必须先看物种来源

`RDX`（谷胱甘肽、硫氧还蛋白、过氧化物酶、过氧化氢酶）与宿主酶高度同源。人的过氧化物酶
和微生物的得分一样高。肠道样本做 RDX，先用分类学信息确认候选确实是细菌的。

### 5.6 BA 样本量小

只有 54 个正样本（277 个簇），AUPRC 的置信区间很宽，且正样本里混了真核酶和细菌酶。
**胆汁酸方向推荐用 BEAUT**（Ding et al., *Cell* 2025），它有远多的数据和实验验证。

### 5.7 引用概率时必须带 AUPRC

> ❌ "这个蛋白是糖苷水解酶，置信度 0.91。"

> ✅ "这个蛋白在 `GLY` 上得 0.91（AUPRC 0.863，可靠档），没有其他面板超过切点。"

没有 AUPRC 的概率是一个没有刻度的数字。

---

## 六、常用命令

```bash
# 单个方向
microgem score proteins.faa --panel GLY

# 一个家族（自动展开成 2 个或 5 个子面板）
microgem score proteins.faa --panel FATE-L      # FATE-L-modifying + FATE-L-releasing
microgem score proteins.faa --panel AATE        # 5 个子面板
microgem score proteins.faa --panel SAM         # SATE + SALE

# 多个面板（任意组合，逗号分隔，大小写不敏感）
microgem score proteins.faa --panel GLY,SATE,PLY

# 自己定切点（注意：这会让面板脱离校准）
microgem score proteins.faa --threshold 0.5

# 只跑前 100 条（试水，避免浪费几小时）
microgem score big.faa --limit 100

# 显示未判定序列的说明
microgem score proteins.faa --verbose

# 查面板详情
microgem panels
microgem panels --format json

# 查安装状态
microgem info --format json
```

### 6.1 参数速查

| 参数 | 说明 |
|---|---|
| `--panel` | 面板名 / 家族名 / `all`（默认） |
| `--threshold` | 强制统一切点，默认用各面板校准值 |
| `--format` | `text` \| `tsv` \| `json` |
| `--prevalence` | 你的输入里"这一类酶"占多大比例，用于算现实精确率 |
| `--show-operating-points` | 打印切点处的精确率/召回率 |
| `--limit N` | 只跑前 N 条 |
| `--model` | 指定 .pt 文件 |
| `--device` | `cuda` \| `cpu` \| `mps` |
| `--backend` | `auto` \| `torch` \| `numpy` |
| `--batch-size` | 批大小（默认 16，显存小就调低） |
| `--verbose` | 显示全部 17 行 + 未判定说明 |
| `--quiet` | 关闭进度和提示 |

---

## 七、在 AI 助手里用

### 7.1 用 `/microgem` 调用（推荐）

在对话框里输入斜杠命令，技能正文会被直接注入当前这一步：

```
/microgem
```

不带参数时，助手会先跑 `info` 和 `selftest` 两项检查，然后问你一句话：

> microGEM 就绪：17 个面板，backend torch，selftest PASS。
> 要打分什么？给我 FASTA 路径或直接贴序列，并说明是跑全部 17 个方向还是某个方向（如 GLY、FATE-L、AATE）。

带任务时可以直接把需求写在后面：

```
/microgem 帮我看看 candidates.faa 里哪些是糖苷水解酶
/microgem 这批肠道宏基因组蛋白里有值得做实验的色氨酸代谢酶吗
/microgem 这个蛋白判成 SALE，能不能信
```

`/microgem` 这个手势出现在句子中间也有效，比如"用 `/microgem` 跑一下这个文件"。
但要注意：`/microgem/refs` 这种带第二个斜杠的写法**不会**触发（那是路径不是命令）。

### 7.2 也可以直接说人话

不写斜杠命令时，只要问题落在技能描述覆盖的范围内（酶功能预测、宏基因组筛酶、
microGEM 分数怎么读），助手也会自动加载它——两条入口行为一致。

助手按 SKILL.md 里的流程走：`info` 确认能跑 → `selftest` 确认结果可信 → 打分 →
按"面板 + 分数 + 可靠度 + margin"四件套汇报 → 建议交叉验证。

### 7.3 技能里写了什么

| 文件 | 内容 |
|---|---|
| `skill/SKILL.md` | 调用方式、主流程、六种典型错误、汇报模板、硬性禁令 |
| `skill/references/what-it-is.md` | 性能表、三个标签修正的量化效果、验证状态 |
| `skill/references/panels.md` | 17 个面板的完整定义与裁决理由 |
| `skill/references/interpreting.md` | 六种误读、切点选择、汇报模板、什么时候该停手 |

### 7.4 硬性禁令（写在技能里）

- 不许说 microGEM "比 UniProt/KEGG/CAZy 更好"——它在它们**之前**（先产候选，再查已知）
- 不许说候选"已验证"——microGEM 零湿实验验证
- 不许在用户要求的方向里悄悄删掉弱面板
- 用户问的酶不在这 17 个方向里时，要说出来并指向 eggNOG/KEGG/dbCAN
- 引用概率必须同时给 AUPRC

---

## 八、从分数到实验

### 8.1 推荐流程

```
宏基因组 ORF ─► microGEM 打分 ─► 候选列表
                                   ├─► dbCAN/CAZy（GLY 方向，独立真值）
                                   ├─► UniProt/BRENDA（查已表征的同源物）
                                   ├─► MetaCyc（可能落在哪条通路）
                                   ├─► 分类学（微生物还是宿主？RDX 必查）
                                   └─► 湿实验
```

### 8.2 交叉验证建议

| 面板 | 建议的独立验证 |
|---|---|
| `GLY` | **dbCAN3 HMM**。microGEM 的 GLY 已用 dbCAN3 V15 验证：67/67 阳性命中 CAZy HMM，66/67（98.5%）是糖苷水解酶；630 条阴性只有 5.7% 有命中。**不一致的地方最有信息量。** |
| `RDX` | 分类学注释（同源宿主酶是主要假阳性来源） |
| `BA` | 直接换 BEAUT |
| 其他 | UniProt/BLAST 查最近同源物的注释；AlphaFold/ESMFold 看口袋 |

### 8.3 挑多少候选

取决于你的通量。给两个参考：

- 96 孔板验证：按 `--prevalence` 后的精确率排序，取前 88 个（留 8 个对照）
- 发现型筛选：可以放宽切点，靠第二轮计算过滤（同源性、分类学、结构域）

---

## 九、性能与成本

### 9.1 计算成本

| 场景 | GPU | CPU |
|---|---|---|
| 单条（约 400 aa） | 0.2 秒 | 1.5 秒（torch）/ 3 秒（numpy） |
| 1,000 条 | 3 分钟 | 25 分钟 / 50 分钟 |
| 100,000 条 | 5.5 小时 | 42 小时 / 83 小时 |

打分本身（17 个线性探针）可以忽略：1280×17 的矩阵乘法，100 万条也只占 CPU 级时间。
**瓶颈永远是 embedding。**

### 9.2 模型体积

| 文件 | 大小 |
|---|---|
| 17 个面板权重合计 | 244 KB |
| ESM-2 检查点（.pt） | 2.6 GB |
| NumPy 副本（.npy × 570） | 2.5 GB |
| 参考集 + 元数据 | 76 KB |

面板权重极小的好处：可以随包分发、可以进 git、可以贴进论文附录。

### 9.3 缓存位置

```bash
microgem info | grep cache
```

默认 `~/.cache/microgem/`，可用 `MICROGEM_HOME` 改。

---

## 十、限制（请务必读）

1. **零湿实验验证。** microGEM 的所有数字都来自计算方法。它的方法学前身 BEAUT 从 614,616 个
   候选里抽了 102 个做实验，47 个有活性；microGEM 一个都没有。
2. **只覆盖 Rhea 的 8.2%。** 17 个面板 1,734 个反应 / 共 19,005 个 master 反应。这 17 个方向
   之外，microGEM 无话可说。
3. **不预测底物特异性。** `FATE-L-modifying` 只说"作用于长链酰基"，不说具体是哪种脂肪酸。
   CAZy 的家族级分类在这条轴上更细。
4. **不输出 EC 号。** 一个面板按设计横跨多个 EC。
5. **不处理大分子底物。** 组蛋白/DNA/RNA 相关的酶不在范围内。
6. **输出是假设，不是结论。** 它的定位在数据库**之前**（产生候选），不是**之后**（检索已知）。

### 什么时候不该用 microGEM

| 情况 | 该用什么 |
|---|---|
| 酶化学不在这 17 个方向 | eggNOG-mapper、KEGG、dbCAN、MetaCyc |
| 蛋白已被表征 | UniProt / BRENDA（且有实验证据） |
| 想知道"作用于哪个具体底物" | CAZy 家族（糖类）/ 结构对接 |
| 需要已验证的酶 | microGEM 帮不上，直接说 |

---

## 十一、排错

### `backend: none`

缺模型或缺 PyTorch。`microgem info` 会准确告诉你缺哪个：

```bash
pip install torch fair-esm && microgem download
```

或者在有 torch 的机器上转一次，之后只用 NumPy：

```bash
microgem download --convert
```

### `microgem selftest` 失败

| 报错 | 原因 |
|---|---|
| `DRIFT`（偏差 > 1e-3） | embedding 变了：权重文件不对、下载被截断、tokenization 被改 |
| `positive control failed` | 探针或权重文件损坏 |
| `cross-panel false-positive rate too high` | 同上，先看是不是换了模型文件 |

先 `md5sum` 对比检查点，再 `microgem convert` 重转 NumPy 副本。

### 内存不够

NumPy 后端是内存映射的（`.npy` 一个张量一个文件），常驻内存只有几百 MB；
torch 后端加载整个 2.6 GB。内存吃紧就 `--backend numpy`。

### 显存不够

```bash
microgem score big.faa --batch-size 4
```

### 跑得太慢

1. 确认在用 GPU：`microgem info` 的 `backend` 是 `torch`，再加 `--device cuda` 显式指定
2. 先用 `--limit 100` 试水，估一下总时间再决定
3. 只要跑少数面板就 `--panel GLY`，但**注意 embedding 还是要算全部**——17 个探针共用一份
   向量，省不了这一步

### 分数全都是 0

几乎一定是序列被清成了 `X`（比如输入是核苷酸序列、或者格式有问题）。

```bash
head -2 proteins.faa
```

确认是氨基酸序列（大写 ACDEFGHIKLMNPQRSTVWY）。microGEM 会把不认识的字符变成 `X`，
而不是删掉——因为在序列中间删字符会悄悄改变池化向量。

---

## 十二、复现

| 内容 | 怎么做 |
|---|---|
| 已发表的 AUPRC | `pipeline/bin/07_baseline_linear_probe.py` |
| 本包用的权重 | `scripts/build_weights.py`（先重放 CV，确认 AUPRC 一致后在全量上重拟合） |
| 操作特性表 | `scripts/build_operating_points.py`（从 OOF 预测算） |
| 参考集 | `scripts/build_reference.py` |
| 当前安装 | `microgem selftest` |

`build_weights.py` 有一道硬检查：它重放 CV 循环，如果算出的 AUPRC 与已发表值差超过 0.0005
就拒绝导出。所以包里的权重和论文里的数字是同一套。

---

## 十三、引用

如果你用了 microGEM，请引用：

> Jun Xu, microGEM: sequence-based scoring of 17 microbial enzyme chemistries with an explicitly calibrated, distribution-ready classifier, https://github.com/xGroup5/microgem (MIT licence).

BibTeX：

```bibtex
@software{xu2026microgem,
  author  = {Xu, Jun},
  title   = {microGEM: sequence-based scoring of 17 microbial enzyme chemistries
             with an explicitly calibrated, distribution-ready classifier},
  year    = {2026},
  url     = {https://github.com/xGroup5/microgem},
  license = {MIT}
}
```

microGEM 建立在 BEAUT 流水线与 Rhea 标签之上，相关场合请一并引用。

方法论来源：

> Ding, Y., Luo, X., Guo, J., et al. (2025). Identification of gut microbial bile acid
> metabolic enzymes via an AI-assisted pipeline. *Cell*, 188(21), 6012–6027.e20.
> https://doi.org/10.1016/j.cell.2025.07.017

标签来源：

> Morgat, A., Lombardot, T., Coudert, E., et al. (2020). Enzyme annotation in UniProtKB
> using Rhea. *Bioinformatics*, 36(6), 1896–1901. https://doi.org/10.1093/bioinformatics/btz817

蛋白语言模型：

> Lin, Z., Akin, H., Rao, R., et al. (2023). Evolutionary-scale prediction of atomic-level
> protein structure with a language model. *Science*, 379(6637), 1123–1130.
> https://doi.org/10.1126/science.ade2574

---

## 十四、文件结构

```
microgem-skill/
├── install.sh                    # 一键安装（命令 + 技能）
├── pyproject.toml                # pip 安装定义
├── microgem/                     # Python 包
│   ├── cli.py                    # 命令行
│   ├── score.py                  # 打分核心
│   ├── embed.py                  # ESM-2 embedding（torch / numpy 双后端）
│   ├── numpy_model.py            # torch-free 前向传播
│   ├── panels.py                 # 17 个面板的注册表
│   ├── selftest.py               # 安装自检
│   └── assets/                   # 权重（244 KB）+ 元数据 + 参考集
├── skill/                        # AI 技能包（复制到技能目录）
│   ├── SKILL.md
│   └── references/
├── scripts/                      # 构建脚本（从研究流水线导出）
├── tests/                        # 32 个测试
├── docs/
│   ├── USER-GUIDE.md             # 本文件
│   └── USER-GUIDE.en.md          # 英文版
└── README.md
```

## 十五、许可

MIT。ESM-2 权重遵循 fair-esm 的 MIT 许可；未随本仓库分发。
