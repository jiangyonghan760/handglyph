# handglyph

**用你自己的手写字，生成手写风格的图片。**

> 🤖 **本文档由 AI 生成**（非人工逐字编写，可能滞后于代码，**以源码与 `能力说明.md` 为准**；发现不符请提 Issue）｜ ⚠️ **使用前请读文末 [免责声明](#免责声明--disclaimer)** ｜ 非官方个人项目，按"现状"提供
> AI-generated README · unofficial personal project · provided "AS IS" · please read the [Disclaimer](#免责声明--disclaimer) at the end first.


---

## 这是什么

给它一张你手写的字版照片，它把里面的字逐个切出来，建成"你的字形库"；
之后你给一段文字，它就用**你自己的笔画**把这些字重新排到纸面上，输出一张图片。

不是字体渲染 —— 每一笔都来自你本人写的字。全程本地计算，不联网、
**不需要 AI 模型**、不产生水印。

> 📌 注意区分：**本程序本身不使用 AI**（笔画全部来自你的手写照片，靠几何算法
> 切分与排版）；但**这份 README 文档是 AI 写的**。两者不是一回事。

**English.** Give it a photo of your own handwriting, and it slices each character out
into a personal glyph library. Then hand it any text, and it re-lays those characters
onto a paper background using **your actual strokes** — output as a single image.
No font rendering, no AI model, no network, no watermark. It runs fully offline;
your handwriting never leaves your machine.

---

## 快速开始（源码方式）

```bash
git clone https://github.com/jiangyonghan760/handglyph.git
cd handglyph
pip install pillow numpy scipy

python handglyph.py doctor        # 环境自检
python handglyph.py selftest      # 跑内置回归测试（69 条断言）
```

想要**开箱即用**的话，去 [Releases](https://github.com/jiangyonghan760/handglyph/releases) 下载 `handglyph_v0.7.zip`：
解压后在 Windows 上双击 `handglyph.cmd`、在 macOS/Linux 上跑 `./handglyph.sh` 即可，
不必自己 clone。发行包里含完整文档与 5 张内置纸面。

---

## 它和"手写字体"有什么不同

手写字体是别人写一遍、矢量化、再复用，一千个人用同一套字，而且**不管什么
内容都是同一个字模**。handglyph 直接从你的字版照片里抠出位图笔迹，所以你写
什么样，出来的就是什么样 —— 包括你那种偏扁的"口"、右下角拖长的撇。

同一个字你多写几遍存进库，排版时会轮着用，所以一页里同一个"的"不会长得一模一样。

---

## 三步跑起来

**1. 装 Python**（3.9+）

Windows 安装时勾选 "Add Python to PATH"。

**2. 装依赖**

```bash
pip install pillow numpy scipy
```

就这三个是必需的。想用 OCR 从图片自动提文字可以另装（可选，不装不影响使用）：

```bash
pip install handglyph[ocr]      # pytesseract 后端，需另装 tesseract 程序本体+中文语言包
pip install handglyph[ocr-pp]   # PaddleOCR 后端，中文识别更准，但 paddlepaddle 有数百 MB
```

**3. 自检**

```bash
handglyph.cmd doctor            # Windows
./handglyph.sh doctor           # macOS / Linux
```

逐项检查 Python 版本、依赖、中文字体、背景图、字库状态，全绿就说明环境没问题。

---

## 正式使用

### ① 写一张字版，拍照

在纸上按顺序写下你打算用到的字，拍一张清楚的照片。正对纸面拍、光线均匀、对焦准确、发原图不压缩。

### ② 建字库

```bash
handglyph.cmd build 你的照片.jpg --expect "照片上从左到右的全部字符" -o library
```

`--expect` 必须和照片上的字符顺序完全一致（含标点），程序靠它给每个字命名。
跑完会生成 `library/` 目录和 `atlas.png`（切分图鉴，用来核对有没有切错）。

### ③ 检查缺字 / 生成补字单

```bash
handglyph.cmd coverage --lib library -t "你打算写的文字"
handglyph.cmd form "缺的字" -o 补字单.png --copies 3
```

打印补字单，手写填好，再入库合并：

```bash
handglyph.cmd build 补字单照片.jpg --form-sheet --expect "与门输入" \
             --merge --replace -o library
```

`--copies 3` = 每个字连着写 3 格。**建议就是 3 遍**：同一个字多几个实例，
排版时轮着用，字迹就不会满篇一副样子。抄到第 4、5 遍人会开始敷衍，那几个
实例质量反而更差 —— 写 5 遍的话你不如自己写了。

⚠️ 补字单**必须**带 `--form-sheet`。补字单上的说明文字、参考字、格线都是印刷
内容；不加 `--form-sheet` 会把它们当成你的笔迹吃进字库。带 `--form-sheet` 时
程序按格子取字：格位顺序就是字符顺序，格外的一律丢弃。

### ④ 出图

```bash
handglyph.cmd render examples/page3.txt --lib library --paper lined -o out.png
```

`--paper` 可选 `white` / `lined` / `grid` / `redgrid` / `cream`，
也可以直接给一张**整幅是纸**的图片路径。

会同时生成 `out.audit.txt`，是这一页的质检报告（逐字检查合格与否，并写明
"应渲染几字 / 实际渲染几字"的字数对账）。

### ⑤ 自动排版

一页写不满、尾巴上一大片空白时：

```bash
handglyph.cmd render examples/page3.txt --lib library --paper lined --flow -o out.png
```

把整篇内容按"填满一页再进下一页"重新分页，内容多了自动分成多页。

---

## 字迹像不像你，靠什么

按重要性排序：

1. **换笔迹** —— 用**你自己的**字库。这是决定性的，别的都是修饰。
2. **字库容量** —— 补字单抄 3 遍，同一个字多几个实例轮着用。
3. **极轻的旋转与形变** —— 默认只有 0.35 度，肉眼基本看不出来。
4. **笔画粗细的轻微浓淡** —— 很小很小，只负责"这一次落笔重了一点点"。

⛔ 不要指望靠调大 ④ 来"制造差异"。任何你**一眼就能看出这个字更粗**的幅度都
属于过大。真想让字迹更丰富，去多补几个字的实例（走 ②），而不是把粗细拉大。

---

## 关于"粗细不齐"

补字是分次写的，不同次写的整体轻重不一样。**程序不管这件事。**

v0.7 曾试过用程序改笔画宽度，试了三版都失败，已于该版整块删除。不是没写完，
是**这件事在数据上做不到**：

- **细笔画根本不可修**：1~1.5px 的笔画做一次"变细"操作就整条消失。实测 1px
  竖条要求变细，输出墨量 40 → 40 点，一点没动 —— 它已经只剩 1px，退无可退。
- **粗一点的笔画只能整格跳**：加粗一次就是一整圈像素。实测条宽 6/10/16 加粗时
  墨量涨幅全是"恰好 1px 宽度"的量，做不到"细一点点"。
- **连量都量不准**：2px 条与 3px 条量出来都是 1.0。所以"看起来没变"里头有很大
  一部分是量尺看不出差别，不是在真比较。

所以现在**粗细一律不动**，只归"墨色浓淡"（改深浅不改几何）。真要粗细齐，
只有两条路，都在写的时候：照补字单上的要求写，同一篇里笔画轻重保持一致；
或者同一个字多写几遍，选最顺眼的那个。

字库会按**批次**记录"哪些字是哪一次存的"：

```bash
handglyph.cmd batch --lib library
```

偏差超过 ±20% 就判"偏离过大"并给出重写清单。程序**只提示、不自动改** ——
自动改是把笔画缩放，会把字弄得不像你写的。

---

## 已知边界

用之前先看一眼，这些都不是 bug，是**已知边界**：

| 边界 | 说明 |
|---|---|
| **自由字版的切分仍不稳** | 直接拿"手写文章"的照片去 build，遇横线纸、打印抬头就会错名。横线纸照片实测未通过。**需要精确入库一律走补字单路线。** |
| **补字单 `--expect` 长度要是 `--cols` 的整数倍** | 不是整数倍时，多出来的格子会被编造成 `u0164` 这样的假字符名写进字库。实测长度 164 配 12 列就会多出 4 格。 |
| **补字单格位漏读会让后面的字整体错名** | 程序按"读到几个有墨的格子"分配字名，中间漏一格，后面全部前移。入库后**务必看 `library/atlas.png`** 核对编号。 |
| **`quality` 报告数字口径不统一** | 同一份报告可能同时出现"需重写 4"和"43 个字不合格"。差在按字符算还是按字形实例算。**以最后那条为准**，它是 `form` 该抄的清单。 |
| **纸面不做检测** | 程序不会替你判断导入的图是不是纸面。误判代价太大，所以得你自己看一眼。 |
| **粗细不齐程序不管** | 见上一节。只能靠写的时候保持一致。 |
| **一页容量有限** | 放不下会以退出码 5 结束并在报告首行写明"内容被截断"，不会静默少字。多页输出尚未实现，先用 `--flow`。 |
| **符号需预先入库** | `⊕`、上下标、`→` 等要么在字版里写过，要么程序合成。目前仅 `F`、`V` 为程序合成。 |

---

## 命令一览

| 命令 | 作用 |
|---|---|
| `doctor` | 环境自检（依赖/字体/背景/字库） |
| `selftest` | 跑内置回归测试（改代码后先跑它） |
| `build` | 字版照片 → 字形库（补字单加 `--form-sheet`） |
| `atlas` | 切分图鉴；`--fix` 按"序号=字"改名字（改名逃生门） |
| `coverage` | 检查缺字 |
| `form` | 生成补字单（打印出来手写填）；`--copies 3` 让每字抄 3 遍 |
| `quality` | 字库质检：哪些字该重写 |
| `shape` | 形状校验：排查被压扁、墨量过低的字形 |
| `batch` | 字库分批：列各批次、比对批间粗细 |
| `calibrate` | 校准质检阈值 |
| `prune` | 淘汰字库里的差实例 |
| `clean` | 对既有字库统一去格线、裁边 |
| `extract` | 从图片提取文本（需 OCR，可选） |
| `render` | 排版并合成图片；`--flow` 自动排版 |

每个命令都可以加 `-h` 看详细参数。

**退出码**（脚本化调用时按这个判成败）：

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 自检失败或发现缺字 |
| 2 | 用法或环境错误（含主动拒绝） |
| 3 | 未预料的运行期错误 |
| 4 | 跑完了但零产出 |
| 5 | 跑完了但有内容没落地（render 超页被截断） |
| 130 | Ctrl-C 中断 |

---

## 几个"会拦住你"的设计（别当成 bug）

- **目标目录已有字库时，直接 build 会被拒绝** —— 那会整体覆盖旧的字形映射。
  要加字就加 `--merge`；只重写照片上这几个字加 `--merge --replace`；确实要
  推倒重建才用 `--rebuild`。
- **补字单模式必须带 `--expect`**。字符名只能来自它，不给就只能编造 `u00xx`，
  所以程序直接退出（退出码 2）而不是硬着头皮入库。
- **补字单上一个字都没读到**时，build 以**退出码 4**结束（不是 0）。跑完了但零
  产出不该算成功。
- **有字形的操作之后会自动删除 `library/metrics.json`**。那是全库质检基准，
  不失效的话新并入的字会被旧基准判成离群，`prune` 会按旧分布删掉好字。
- **`form` 与 `build --form-sheet` 的 `--cols` 必须同值**（默认都是 12），
  否则格位对不上。`form` 会按纸面尺寸告诉你一页最多能放几格。

---

## 从源码运行 / 参与开发

本仓库就是一个可运行的源码树，所有命令都可以用 `python handglyph.py <子命令>` 代替。

代码是**单文件** `handglyph.py`（约 6000 行），顶层分类整齐，没有自建包结构。

改完代码按这个顺序自检：

```bash
handglyph.cmd selftest      # 内置回归测试，70 处断言，全过才继续
python pack.py              # 打包门禁：静态检查 + 编译检查 + lint + selftest
```

`pack.py` 会做**静态检查**（例如"模块常量必须先于引用它的函数默认值定义"），
任一环节不过就**不出 zip**。要跳过门禁强制出包用 `--no-gate`，但发版别这么干。

版本号只有一个真源：`pyproject.toml` 的 `version`。包的目录名、zip 名都从它派生；
`pack.py` 会核对它与 `handglyph.py` 里的 `__version__` 是否一致。

---

## 隐私

- **全程本地计算**，不联网、不上传任何数据。
- **本仓库不包含任何人的手写字形库**。字形库是个人的笔迹数据，每个人需要用
  自己的字版建一份自己的库。`library/`、`*.audit.txt`、`atlas.png` 都在
  `.gitignore` 里，不会进版本库。
- `backgrounds/` 里的纸面是程序生成的纯纸纹理，可以放心用。

---

## 许可

[MIT](LICENSE)。

本程序生成的图片，其笔迹来自使用者自己提供的字版照片。使用者需自行确保对
所提供的字版照片拥有相应权利。

---

更详细的参数默认值、内部机制与历轮修复记录，见 [能力说明.md](能力说明.md)。

---
---

# 免责声明 · Disclaimer

**请在使用前完整阅读本节。**

## 进入条款

**您下载、克隆、安装、复制、运行、修改或以任何方式使用本项目（以下称"本软件"）
的全部或部分，即视为您已完整阅读、充分理解并不可撤销地无条件接受本免责声明的
全部条款。** 若您不同意本声明的任何内容，请立即停止使用并删除您持有的全部副本。
您对本软件的任何使用行为，均构成对本声明的持续接受。

## 一、非官方声明

1. **本软件为个人独立开发、业余时间完成的非商业性作品**，系开发者个人学习与
   技术实践的产物，**不隶属于任何公司、组织、机构、团体或项目**，亦不代表
   任何实体的立场或观点。
2. **本软件未获得任何第三方（包括但不限于任何公司、品牌、平台、服务商、
   开源基金会、教育机构）的授权、认可、赞助、合作、认证或背书**，亦与之
   **不存在任何形式的关联、代理、合伙、雇佣或隶属关系**。
3. 本软件中出现的任何名称、标识、示例文本、演示内容，均为**描述性原创命名
   或为说明功能而作的虚构示例，不指向、不影射任何现实主体**。若与现实中的
   任何主体存在名称重合或内容雷同，**纯属巧合且非开发者本意**。
4. 开发者在撰写本文档与编写本软件时**未有意引用、复制或抄袭**任何第三方的
   专有代码、文档、图形、商标或其他受保护内容。若权利人认为本软件或本文档
   存在侵权内容，请通过本仓库的 **Issues** 渠道联系，开发者将在核实后
   **立即删除相关内容**并配合处理。

## 二、技术与内容免责

1. **本软件按"现状"（AS IS）与"现有"（AS AVAILABLE）提供**，不附带任何形式的
   明示或默示担保，包括但不限于：**适销性担保、特定用途适用性担保、不侵权
   担保、准确性担保、完整性担保、无错误担保、不中断担保、无有害成分担保、
   以及因交易习惯或行业惯例而产生的任何担保**。上述担保在适用法律允许的最大
   范围内**全部予以排除**。
2. **开发者不保证本软件能够满足您的任何特定需求或期望**，不保证其运行不会
   中断、不会出错、不会失败、不会产生非预期结果，也不保证其中的任何缺陷
   （已知或未知）会被发现、报告或修复。
3. **开发者不保证本软件与您的硬件、操作系统、Python 版本、第三方依赖库
   或其他软件环境兼容**，亦不保证跨平台运行结果一致。
4. ⚠️ **本软件不是任何形式的专业建议来源。** 本软件及其输出内容**不构成**
   法律、医疗、金融、投资、税务、心理、安全或其他任何专业领域的意见或建议。
   本软件输出的任何内容**均不得作为任何决策的依据**。如有相关需求，**应当
   咨询具备相应资质的专业人士**。
5. ⚠️ **本文档由人工智能生成。** 本文档（含本免责声明）为 AI 辅助撰写，
   其对功能、参数、性能、限制条件的描述**可能存在偏差、过时、遗漏或错误**。
   **一切以源码实际行为为准**；文档与代码不一致时，**以代码为准**。

## 三、网络、第三方组件与接口免责

1. **本软件的核心功能全程在本地计算，不联网、不上传任何数据、不调用任何
   云端模型或在线接口**，亦**不主动收集、不上传、不存储、不分析**任何形式的
   个人信息、使用数据或内容数据。
2. 本软件的可选功能（如 OCR 文本提取）**依赖第三方开源组件**。这些组件
   **非本项目所有、运营或控制**，其可用性、稳定性、功能范围、许可条款与
   收费标准**可能随时变更、中止或终止**，开发者对此**不承担任何责任**，亦
   **不作任何明示或默示的担保**。您**应当自行查阅并遵守**这些第三方组件的
   许可条款与使用政策。
3. 本软件的某些可选功能可能需要网络连接。**您应当自行确保网络环境可用、
   合规、安全**。因网络不可达、被拦截、被限制、被中断而产生的任何后果
   （包括但不限于功能不可用、数据不完整、任务失败），**开发者不承担任何责任**。
4. ⚠️ **您不得通过本软件处理任何敏感信息、个人隐私信息、商业机密信息或
   依法受保护的数据**，除非您已自行确保处理行为完全符合适用法律法规的要求。
   开发者**不建议**将本软件用于上述场景，并对此**不承担任何责任**。
5. 本软件处理的是**您本人提供的字版照片与文本内容**。**您应当自行承担**对
   这些材料的保管、备份与保密义务。因您自身的保管不当、设备故障、误操作、
   磁盘损坏等导致的数据丢失或泄露，**开发者不承担任何责任**。

## 四、使用限制

1. **您应当自行确保**对本软件的使用行为完全符合您所在国家或地区的全部适用
   法律法规、监管要求及公序良俗，并**自行取得**为使用本软件所必需的一切
   授权、许可或同意。**因您的使用行为违反前述要求而产生的一切后果与法律责任，
   由您自行承担。**
2. **禁止将本软件用于任何违法违规用途**，包括但不限于：
   - 制作、复制、传播违反法律法规的信息或内容；
   - 侵犯他人的著作权、商标权、专利权、名誉权、隐私权、肖像权或其他合法权益；
   - 实施骚扰、欺诈、诽谤、恐吓、跟踪等侵害他人权益的行为；
   - 伪造他人签名、笔迹、文件、票据、凭证或身份信息；
   - 规避、破坏或干扰任何安全机制、访问控制或技术保护措施。
3. ⚠️ **禁止将本软件或其输出内容用于商业营销、品牌代言、官方客服、自动化
   应答、身份冒用，以及任何可能使公众误认为您与任何主体存在关联或获得其
   授权的场景**，除非您已取得相应主体的书面授权并自行承担全部法律后果。
4. ⚠️ **禁止将本软件生成的任何内容用于伪造、冒充或暗示他人身份或意思表示。**
   手写风格图像的生成具有一定逼真度，**您应当自行确保**最终产出不被用于任何
   可能引起误解、争议或法律纠纷的用途。**因您的使用行为引发的一切后果，
   由您自行承担。**
5. **禁止移除、篡改、隐藏或以任何方式淡化**本软件中的著作权声明、许可声明
   及本免责声明。
6. **禁止将本软件或其修改版本用于任何违反本声明的用途。** 违反者**责任独立
   自负**，并应当赔偿由此给开发者造成的全部损失。

## 五、责任限制

1. **在适用法律允许的最大范围内，开发者对本软件及其使用或无法使用所产生的
   任何损害均不承担任何责任**，包括但不限于：**直接损害、间接损害、附带损害、
   特殊损害、惩罚性损害、示范性损害、后果性损害**。
2. 前述损害**包括但不限于**：数据丢失或损坏、设备损坏或不可用、系统故障、
   业务中断、利润损失、收入损失、商誉损失、机会损失、替代服务成本、第三方
   索赔，以及因本软件生成内容而引发的任何争议、纠纷、索赔或诉讼。
3. ⚠️ **无论基于何种责任理论（包括但不限于合同责任、侵权责任、过失责任、
   严格责任或其他任何理论），无论开发者是否已被告知该等损害发生的可能性，
   前述责任限制均同样适用。**
4. **本声明的各项条款具有可分割性。** 若其中任何条款被有管辖权的机关认定为
   无效、非法或不可执行，该条款应当在最小必要范围内被限缩或删除，**其余条款
   的效力不受影响，继续完全有效**。
5. **使用本软件的全部风险与后果，由您自行承担。** 您应当自行评估本软件是否
   适合您的使用场景，并自行采取必要的预防措施（包括但不限于数据备份、
   结果校验、合规审查）。

## 六、第三方组件

1. 本软件依赖的第三方开源组件**各自适用其自身的许可协议**，与本项目的许可
   **相互独立、互不影响**。您的使用行为**应当同时遵守**这些第三方许可协议。
2. **您应当自行查阅并遵守**所依赖的第三方组件的许可条款。**因您违反第三方
   许可条款而产生的一切责任，由您自行承担**，开发者不承担任何责任。
3. 本项目**不分发任何专有代码、AI 模型权重、商业字体或受版权保护的素材**。
   本项目内置的背景纸面均为程序生成的纯纸纹理。若您自行向本项目中引入任何
   第三方素材，**由此产生的全部责任由您自行承担**。

## 七、许可范围

1. 本项目的源代码依据 **MIT 许可协议**发布，具体条款**以本仓库的
   [`LICENSE`](LICENSE) 文件为准**。
2. **开源许可协议仅约束源代码的使用、复制、修改与分发行为。本免责声明是
   额外的风险提示与责任约定，不削减、不限制开源许可协议所授予的任何权利。**
3. ⚠️ **若本免责声明与开源许可协议之间存在不一致：就责任限制、担保排除与
   使用限制事项，以本免责声明为准；就授权范围与权利授予事项，以开源许可
   协议为准。**
4. 本声明**不改变**您依据适用法律的强制性规定所享有的、不可被合同排除的
   法定权利。

## 八、条款变更与解释

1. **开发者保留随时修改、更新、补充、暂停或终止本免责声明及本项目的权利，
   无需事先通知，亦无需征得您的同意。** 变更后的条款自发布之日起生效。
   **您在任何变更后继续使用本软件，即视为您已接受变更后的全部条款。**
   建议您定期查阅本声明的最新版本。
2. 本声明各节的**标题仅为阅读便利而设，不影响、不限制、不扩张任何条款的
   含义与解释**。
3. ⚠️ **本声明以中文版本为准。** 文末所附英文摘要**仅供参考**，若中英文表述
   存在任何歧义或冲突，**一律以中文版本为准**。
4. 本声明的解释与适用，应当遵循其字面含义与订立目的。对条款含义存在争议时，
   应当按照"限制开发者责任、由使用者自担风险"的目的进行解释。

---

## English Summary (for reference only — the Chinese text above prevails)

**By downloading, cloning, installing, copying, or running this software, you
acknowledge that you have read, understood, and unconditionally accepted this
disclaimer in its entirety.** If you do not agree, stop using the software
immediately and delete all copies.

This project is a **personal, non-commercial hobby work** released for learning
and technical practice purposes. It is **not affiliated with, authorized by,
endorsed by, sponsored by, or in any way associated with any company, brand,
platform, organization, or service provider.** Any resemblance to existing
names or content is purely coincidental.

This software is provided **"AS IS" and "AS AVAILABLE", without warranty of any
kind**, express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, title, accuracy,
completeness, non-infringement, and any warranties arising out of course of
dealing or usage of trade. **All such warranties are disclaimed to the maximum
extent permitted by applicable law.**

**This software does not constitute professional advice of any kind.** Its
output must not be relied upon as a basis for any decision.

**This README document is AI-generated** and may contain inaccuracies.
**The source code prevails.**

You are solely responsible for ensuring that your use of this software complies
with all applicable laws, regulations, and third-party rights in your
jurisdiction. You are prohibited from using this software to forge or misappropriate
another person's identity, signature, or handwriting, and from using it in any
manner that could cause the public to mistakenly believe you are associated with
or authorized by any entity.

**In no event shall the author be liable for any claim, damages, or other
liability, whether in an action of contract, tort, negligence, strict liability,
or otherwise, arising from, out of, or in connection with the software or the
use or other dealings in the software**, including but not limited to direct,
indirect, incidental, special, punitive, exemplary, or consequential damages,
or damages for loss of data, loss of profits, business interruption, loss of
goodwill, or third-party claims, **even if the author has been advised of the
possibility of such damages. All risk and responsibility for use of this project
rests solely with you.**

Third-party components are governed by their own licenses. The MIT License
governs the source code only and is not diminished by this disclaimer. Where
this disclaimer and the MIT License conflict, this disclaimer controls with
respect to liability limitations, warranty disclaimers, and usage
restrictions, while the MIT License controls with respect to the scope of
rights granted.

The author may modify, update, suspend, or terminate this disclaimer and this
project at any time without prior notice. Continued use constitutes acceptance.

**This disclaimer is governed by the Chinese text above.**
