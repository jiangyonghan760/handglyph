"""handglyph - 用你自己的手写样本生成看起来像手写的图片。

流程：字版照片 -> 字形库 -> 覆盖率检查/补字单 -> 按纸面排版合成。

版本号在这里；`pyproject.toml` 里的 version 必须与它一致（打包脚本会核对，
不一致时警告而不中止 —— 毕竟版本号写错不该拦住出包）。
"""

#: 程序版本。改动行为时递增，并与 pyproject.toml 同步。
__version__ = "0.7.0"

import argparse
import glob
import json
import os
import sys
import time


def _bail_on_missing_deps():
    """在 import 第三方库之前先自检。

    为什么必须放在最前面：如果直接 `import numpy`，缺库时 Python 会在执行到 main()
    之前就抛 ModuleNotFoundError，用户看到的是一屏 traceback，不知道该干什么。
    这里抢先检查，缺什么就直说装什么。
    """
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("PIL", "pillow"), ("scipy", "scipy")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("缺少运行所需的组件：%s" % ", ".join(missing))
        print("")
        print("在命令行里执行下面这行装一下就好：")
        print("")
        print("    pip install %s" % " ".join(missing))
        print("")
        print("如果提示 pip 不是命令，说明电脑上还没装 Python。")
        print("去 https://www.python.org/downloads/ 下载安装，")
        print("安装时务必勾选 “Add Python to PATH”，装完重开命令行再试。")
        print("")
        print("装好后可以先跑一次自检：")
        print("    handglyph doctor")
        sys.exit(2)


_bail_on_missing_deps()

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy import ndimage

INK = (32, 32, 36)

# 格子边长（像素）。**唯一真源**：make_form 画格、build --form-sheet 兜底、
# 图鉴排版、残片闸门、自检造样张全都引用它。
# 原来这个 108 在十几处各写各的，其中 `_write_atlas` 写的是 104 —— 两处不一致，
# 导致图鉴格子比补字单窄 4px（不影响正确性，但说明"同一个几何量散在多处"必然漂移）。
#
# ⚠️ 必须定义在**所有用到它的函数默认值之前**（本文件第一个这样的默认值在
# build 的签名里）。Python 的默认参数在 def 执行时求值，放晚了就是 NameError。
CELL = 108

# 字形 PNG 里 RGB 通道存的墨色。**必须与 INK 同源** —— 这两处原来各写各的：
# 渲染用的 INK 是 (32,32,36)，入库写死 38，clean 又写回 32。
# 结果同一个字形经 render 和经 clean 出来的颜色差 6 级，肉眼在深色纸上能看出来。
# 现在一律由 INK 派生，只此一处。
def ink_rgb(v=None):
    """字形 PNG 的 RGB 通道值：默认取 INK 的亮度代表（三通道取均值后取整）。"""
    return int(round(sum(INK) / 3.0)) if v is None else int(v)


PAD = 4
LATIN = 0.95
WEAK = set()
# 扶正角度的缓存：键是 (tol, cap, gain, 形状, 内容和, 平方和)，值是 lean_report 的结论。
# 在同一页里同一个字会出现在几十个位置，vary() 每次都会问一遍"这字歪不歪" ——
# 而那是这支程序里最贵的单字操作（两级搜索几十次旋转）。缓存掉纯重复计算。
_UPRIGHT_CACHE = {}
# 命中/未命中计数。存在的意义是**可观测**：没有它，"缓存其实没生效"没人能发现。
# 用 [int] 而非模块级 int，省掉 global；selftest 直接读它做 >80% 的断言。
CACHE_HIT = [0]
CACHE_MISS = [0]
# 整字微旋转上限（度）。**这是"极轻"档。**
# 用户 2026-09-17 的原话是"略微……调整一下旋转角度" —— 略微是关键词。
# 手写时字的倾角分布集中在 ±1.5° 内，能看出 3° 就已经是"字东倒西歪"了。
# 原来 0.8 在放大图里已经能靠肉眼逐字分辨出"这个字转了" —— 太显眼。
# 0.35 的效果：整篇看上去基线平整，只有并排比对同一字的两个实例才能看出差别。
ROTATE = 0.35
STRETCH = 0.14

# 收笔出锋的长度抖动区间（相对字高的比例）。
#
# ⛔ 原来这里是 `uniform(0.6, 1.0) * STRETCH` —— 即**长度永远是正的**，
# 每写一个字都出一次锋，而且幅度在 0.6~1.0 倍之间摆。这是"粗细/墨量"观感里
# 最大的一处来源：实测同一字形走两个随机种子，墨量差 12.1%，其中 7.6 个点
# 全来自这一步（restroke 才 7.3、旋转只有 0.1）。
# 用户 2026-09-17 说「粗细变化幅度太大了」——指的就是这个。
# 现在改成 0.55~0.95 倍，且**不是每字都做**（见 vary 里的概率），
# 让出锋变成"偶尔有一笔带出个小尾巴"，而不是"每个字都长出尖来"。
ELONGATE_LO = 0.55
ELONGATE_HI = 0.95
# 有多少比例的字做出锋。原来是 0.75，现在降到 0.45 —— 一半以上的字不做出锋，
# 墨量分布就窄了，整页的"胖瘦不均"也就没了。
ELONGATE_P = 0.45
# 出锋允许带来的墨量增幅上限。出锋本身是"把字搬出去一截"，方向不同增幅差到
# 14 个百分点（水平 +18%、对角 +32%），所以算完必须归一化回来，
# 只留这 12% 当作"这一笔收得重了点"。见 elongate 的注释。
ELONGATE_INK = 0.12

# ---- 粗细（笔画墨量）波动的强度。**这是"微妙"档，不要再往上调。**
#
# 观感的分工（2026-09-17 用户明确）：
#   字迹像不像本人，主要靠①换笔迹（用本人字库）+②字库容量（多几个实例轮着用）
#   +③极轻的旋转，**不是**靠笔画粗细。
#   粗细只负责"同一支笔、同一只手，这一次落笔比上一次重了一点点"这一层。
#
# 上一版把粗细做成了主角（边缘 ±0.34 开合、整体墨量再乘 1.16），放大看像描粗的
# 艺术字。这一版把开合动作**降级为一层连续的轻微增浓/减淡**，不再做 0/1 硬切换：
#   · 不再往字形外面"糊"一圈边缘像素（那是变胖，不是变重）—— 之前 STROKE_HI=0.80
#     会把扩张边缘整圈点亮，实测两遍墨量差 12.3%，肉眼一眼能看出，被断言拦下；
#   · 改成按噪声场给墨迹乘一个 1±δ 的系数，δ 上限只有 6%；
#   · 再叠一点点"顺着噪声明暗"的墨量重分配，让笔画边缘的深浅有起伏，
#     模拟墨水量不同的那种不均匀，而不是整字变胖变瘦。
# 三者相乘后单字墨量差异实测约 ±4%（见 selftest 的"粗细扰动实测"那条断言），
# 这个量级只有把同一个字并排摆两个才看得出。要更弱就把 --stroke 往 0 调。
STROKE_BIAS = 0.06
STROKE_LO = 0.88
STROKE_HI = 0.90
STROKE_GATE = 0.30
# 上面四个的**基准值**（只读）。`--stroke` 必须每次从基准派生，不能从
# "上一次已经算过的值"派生 —— 后者在同进程里调两次 main() 时会累积成
# k²、k³…（单次 CLI 调用看不出来，但库用法/CLI 化的自检会踩到）。
_STROKE_BIAS0, _STROKE_LO0, _STROKE_HI0, _STROKE_GATE0 = (
    STROKE_BIAS, STROKE_LO, STROKE_HI, STROKE_GATE)
# 描边场（restroke 内部那层低频噪声）的平滑尺度。越小越"碎"、越像退化的笔画；
# 越大越像整体换了一支笔。1.6 是原值，保留。
STROKE_SIGMA = 1.6

# 行基线漂移幅度（像素）。真实书写时，手腕沿一条缓慢起伏的基线走，
# 不是每个字各自独立上下跳。原来只有"每字 ±2px 独立抖动"，
# 放大看会觉得字都钉在一条数学直线上。改成整行叠一条低频正弦（周期约 2.6 字宽）。
BASE_DRIFT = 3.0
# 长行是否折行到下一行（默认折）。关掉则恢复"整行等比缩小"的老行为。
NO_WRAP = False

# 渲染时由程序自己处理的字符，不需要字库里的字形：
#   DOT_CHARS —— 渲染成程序合成的顿点（中文逗号、顿号、间隔号、英文句点、中文句号）
#   SUP_CHARS —— 上下标语法标记，不是要写的字
#   SKIP_CHARS —— 空白加上以上全部
# coverage 与 render 必须共用同一组集合。此前 coverage 只跳空白，而 render 额外
# 跳过了标点，导致 coverage 把「，」报成缺字、让用户去写标点补字单，
# 而 render 根本不需要它；反过来「。」两边都没覆盖，写句子时会被静默跳过。
DOT_CHARS = "，、·.。"
SUP_CHARS = "^~"
SKIP_CHARS = " \t" + DOT_CHARS + SUP_CHARS

# 竖画扶正：容差（度）。字形竖画偏离竖直超过该角度才动手；0 关闭。
# 默认 1.2° —— 正常手写竖画本身有 1 度左右的自然抖动，低于它不该被判成"歪"。
LEAN_TOL = 1.2
# 单次扶正幅度上限（度）。宁可少扶，不能把横画带斜。
LEAN_CAP = 3.5
# 置信门限：候选角的长竖画得分须比"整条曲线的中位分"高该比例，才认为真找到了竖画。
# 标定依据：用合成长竖线（已知倾斜 3~12 度）测试，gain 取到 1.0 仍能 100% 召回，
# 而真实字库里会被改动的字数从 28 降到 15 —— 同等召回率下误判减半，故取 1.0。
LEAN_GAIN = 1.0
# 角度搜索范围（度）。手写竖画很少歪过 18 度，超过就不是扶正能救的了。
LEAN_SPAN = 18.0


# ---------------------------------------------------------------- 跨平台字体查找
# 程序里需要一个"正楷/黑体"字体来：画补字单的提示文字、画 atlas 序号、以及
# 形状校验时渲染参照字形。这些都不参与成品笔迹生成（成品笔画全部来自用户手写），
# 只是界面与参照用途，所以找不到中文字体也不影响核心功能。
#
# 各平台的常见中文字体路径。按优先级排列，取第一个存在的。
_FONT_CANDIDATES = {
    "win32": [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑粗体
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
        "C:/Windows/Fonts/simkai.ttf",    # 楷体
    ],
    "darwin": [
        "/System/Library/Fonts/PingFang.ttc",              # 苹方
        "/System/Library/Fonts/STHeiti Medium.ttc",        # 华文黑体
        "/System/Library/Fonts/Supplemental/Songti.ttc",   # 宋体
        "/Library/Fonts/Arial Unicode.ttf",
    ],
    "linux": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ],
}
# Linux 发行版字体文件名差异极大，找不到时再按名字模糊兜一遍。
_FONT_GLOBS = [
    "/usr/share/fonts/**/NotoSansCJK*.ttc",
    "/usr/share/fonts/**/NotoSansSC*.otf",
    "/usr/share/fonts/**/wqy-*.ttc",
    "/usr/share/fonts/**/*CJK*.tt[cf]",
    "/usr/local/share/fonts/**/*CJK*.tt[cf]",
]
_font_cache = {}


def _platform_key():
    if sys.platform.startswith("win"):
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def cjk_font_path():
    """找一个可用的中文字体文件，返回路径；一个都没有则返回 None。

    优先看环境变量 HANDGLYPH_FONT（用户可用它指定任意字体，覆盖自动探测）。
    """
    env = os.environ.get("HANDGLYPH_FONT")
    if env and os.path.isfile(env):
        return env
    for p in _FONT_CANDIDATES.get(_platform_key(), []):
        if os.path.isfile(p):
            return p
    for pat in _FONT_GLOBS:
        try:
            hits = sorted(glob.glob(pat, recursive=True))
        except Exception:
            hits = []
        if hits:
            return hits[0]
    return None


def cjk_font(size, bold=False):
    """按字号取字体对象。找不到中文字体时降级为 PIL 内置位图字体（仍可出图，只是汉字显示为方块）。"""
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    path = cjk_font_path()
    f = None
    if path:
        try:
            f = ImageFont.truetype(path, size)
        except Exception:
            f = None
    if f is None:
        try:
            f = ImageFont.load_default(size)
        except TypeError:      # 老版本 Pillow 的 load_default 不收 size
            f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def font_warning():
    """字体缺失时给一次性提示，写清楚怎么解决（用户是电脑就能自己修）。"""
    if FONT_WARNED[0]:
        return []
    FONT_WARNED[0] = True
    if cjk_font_path():
        return []
    return [
        "提示：本机没找到中文字体，补字单与图集里的说明文字会显示不出来（不影响成品笔迹）。",
        "  解决：装任一中文字体（Windows 自带微软雅黑；macOS 自带苹方；Linux 装 fonts-noto-cjk），",
        "  或设置环境变量 HANDGLYPH_FONT 指向一个字体文件，例如：",
        "    set HANDGLYPH_FONT=D:\\fonts\\my.ttf      (Windows)",
        "    export HANDGLYPH_FONT=/path/to/my.ttf    (macOS / Linux)",
    ]


FONT_WARNED = [False]


# ---------------------------------------------------------------- 读图与读文本

def open_rgb(path):
    """读图为 RGB，并把 EXIF 的旋转标记真正应用到像素上。

    为什么必须做：手机竖拍时，像素往往仍按"横向"存储，只在 EXIF 里写一条
    Orientation=6/8 表示"显示时要转 90 度"。PIL 的 Image.open 只读像素、
    不读这条标记，于是竖着拍的字版进到程序里是横躺的 —— 行分组、去格线、
    投影检格这些"按行按列"的假设全部失效，表现是切出来的字形乱七八糟，
    而用户完全不知道问题出在哪（照片在手机里看是正的）。

    ImageOps.exif_transpose 会把标记转成真实像素并删掉标记，
    对没有标记的图是空操作，可以无脑调用。
    """
    im = Image.open(path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass                      # EXIF 损坏不该让程序挂掉，按原样继续
    return im.convert("RGB")


def read_text(path):
    """读文本，兼容 Windows 记事本存的 BOM。

    Windows 记事本保存 UTF-8 时会写一个 BOM（\\ufeff）在文件开头。
    用 encoding="utf-8" 读，BOM 会变成正文第一个看不见的字符：
    首行 `* 标题` 的判定会失效（因为拿到的是 "\\ufeff*"），
    换行语义也连带错位。用 utf-8-sig 读会自动吃掉 BOM，
    对本来就没有 BOM 的文件完全等价（不产生任何副作用）。
    """
    with open(path, encoding="utf-8-sig") as f:
        return f.read()


def read_json(path, default=None):
    """读 JSON 文件，顺手把句柄关掉，读不了就返回 default。

    为什么单独封装：原来全篇是 `json.load(open(p, encoding=...))`。
    CPython 的引用计数会让这种写法"看起来"没问题，但它是靠 GC 兜底的 ——
    文件句柄在 json.load 返回后才被回收。在 Windows 上，句柄没关时
    对这个文件做 os.remove / os.replace 会直接抛 PermissionError。
    本程序正好大量"读完就改写同一个 manifest.json"，这个坑迟早会踩到。
    """
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data, indent=1):
    """写 JSON 文件（同样保证句柄关闭）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)



# ---------------------------------------------------------------- 字形库

def edge_ratio(m):
    """锐利度：笔画边缘像素占笔画面积之比。笔画越粗/越糊，这个值越低（3px 笔画≈0.6，8px 糊团≈0.25）。"""
    if m.sum() < 8:
        return 0.0
    edge = m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3)))
    return float(edge.sum()) / float(m.sum())


def score_alpha(a):
    """字形清晰度打分：越大、越实、锐利度越高越好。"""
    ink = a > 0.55
    if ink.sum() < 8:
        return 0.0
    h = a.shape[0]
    fill = float(ink.sum()) / float(a.size)
    solid = float(a[ink].mean())
    sharp = min(1.0, edge_ratio(ink) / 0.65)
    return 0.45 * min(1.0, h / 34.0) + 0.25 * min(1.0, solid) + 0.15 * min(1.0, fill / 0.16) + 0.15 * sharp


def man_has_glyphs(man):
    """manifest 里有没有"真的存着字形"。

    判据是 chars 里存在非空列表，不是"文件存在"也不是"chars 非空 dict"：
    空字库（chars={} 或 {"甲": []}）当作没有，允许直接 build 进去。
    """
    if not isinstance(man, dict):
        return False
    for _ch, rels in (man.get("chars") or {}).items():
        if rels:
            return True
    return False


def safe_rel(lib_path, rel):
    """把 manifest 里的 rel 变成一个"确认落在字库目录内"的绝对路径；不安全返回 None。

    为什么必须有：manifest.json 是纯文本，用户可以手改，也可能从别人的字库拷来。
    里面写一条 "../../important.png"，prune / clean 就会照删不误 —— 那是字库外的文件。
    读库的每个入口都过一遍这道闸，比在删除处单独判断可靠（删除点有五六个）。

    判据用 realpath 前缀比较（能识破符号链接绕行），不是简单的 ".." 字符串检查。
    """
    if not rel or os.path.isabs(rel):
        return None
    root = os.path.realpath(lib_path)
    p = os.path.realpath(os.path.join(lib_path, rel))
    if p != root and not p.startswith(root + os.sep):
        return None
    return p


def invalidate_metrics(lib_path):
    """删掉质检缓存 —— 任何改动字形的操作之后都必须调它。

    为什么集中成一个函数：原来 atlas_fix / clean / prune 各自手写一遍
    "os.remove(metrics.json)"，而 _store_glyphs（build 的落库口，**最常改库的那条路**）
    漏了。后果不是"缓存过期"这么温和 —— metrics.json 存的是**全库分组的
    中位数与 MAD**，新并入的一批字若拍得偏软，旧基准会把这批新字全判成离群，
    quality 让你白重写一遍；更糟的是 prune 用旧基准算相对闸门，
    按旧分布"正常"的分值去删新字，删掉的是好字。
    """
    mp = os.path.join(lib_path, "metrics.json")
    try:
        if os.path.exists(mp):
            os.remove(mp)
            return True
    except OSError:
        pass
    return False


def load_manifest(lib_path, need=True):
    """安全读字库 manifest。

    need=True 时字库不存在就退出并说清怎么办；need=False 返回空骨架（用于建库场景）。
    """
    mf = os.path.join(lib_path, "manifest.json")
    if not os.path.isfile(mf):
        if need:
            print("找不到字库：%s" % mf)
            print("  如果还没建库，先跑：")
            print("    handglyph build 你的字版照片.jpg --expect \"照片上的全部字符\" -o library")
            print("  如果已有库，检查 --lib 是否指对了目录。")
            sys.exit(2)
        return {"chars": {}, "version": 1}
    try:
        with open(mf, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        print("字库文件读不了：%s" % mf)
        print("  %s: %s" % (type(e).__name__, str(e)[:120]))
        print("  文件可能损坏。可以重新建库，或从备份恢复。")
        sys.exit(2)


def load_alpha(p):
    """读一张字形 PNG 的 alpha 通道并归一化到 0~1。

    字形文件统一存成 RGBA：RGB 通道恒为墨色，笔画形状全部在 alpha 里。
    原来这行在源码里重复了 7 遍。
    """
    im = Image.open(p).convert("RGBA")
    return np.asarray(im.split()[-1]).astype(np.float32) / 255.0


def load_library(path):
    """读字库。字库不存在/不完整时给出可执行的提示，而不是抛 traceback。"""
    mf = os.path.join(path, "manifest.json")
    if not os.path.isfile(mf):
        if not os.path.isdir(path):
            print("找不到字库目录：%s" % path)
            print("")
            print("如果这是第一次使用，需要先用你的字版照片建一个字库：")
            print("  handglyph build 你的字版照片.jpg --expect \"照片上的全部字符\" -o library")
            print("")
            print("然后把 --lib 指到它（默认就是 library）：")
            print("  handglyph render 页面描述.txt --lib library -o out.png")
        else:
            print("字库目录存在但不完整：%s" % path)
            print("  缺少 manifest.json，说明它不是一个完整的字库。")
            print("  请用 build 重新建库，或把 --lib 指向正确的字库目录。")
        sys.exit(2)
    try:
        with open(mf, encoding="utf-8-sig") as f:
            man = json.load(f)
    except Exception as e:
        print("字库文件读不了：%s" % mf)
        print("  %s: %s" % (type(e).__name__, str(e)[:120]))
        print("  文件可能损坏了。可以用 build 重新建库。")
        sys.exit(2)
    lib = {}
    for ch, items in man.get("chars", {}).items():
        alphas = []
        for rel in items:
            p = safe_rel(path, rel)
            if p is None:
                continue
            if not os.path.exists(p):
                continue
            a = load_alpha(p)
            # 读取侧只拦**残片**，不再做形状否决。
            # 理由：形状问题已在入库时（_store_glyphs）判过一轮，这里再否决一次
            # 等于把当时的误判固化进"读不出来"——用户明明在库里的字，render 时说没有。
            # 更糟的是这条误判对单笔画字（一丨-）恒真，那几个字永远渲染不出来。
            if plaus(a, ch=ch) == "残片":
                continue
            alphas.append((score_alpha(a), a))
        if alphas:
            alphas.sort(key=lambda t: -t[0])
            if alphas[0][0] < 0.62:
                WEAK.add(ch)
            lib[ch] = [a for _s, a in alphas[:2]]
    if not lib:
        print("字库里没有任何可用的字形：%s" % path)
        print("  可能字形文件都丢了。请用 build 重新建库。")
        sys.exit(2)
    return lib


def coverage(lib, text):
    """查缺字。跳过集合必须与 render 一致，否则会报出 render 根本不需要的缺字。"""
    need = {}
    for ch in text:
        if ch in "\n\r" or ch in SKIP_CHARS:
            continue
        need[ch] = need.get(ch, 0) + 1
    missing = {c: n for c, n in need.items() if c not in lib}
    return need, missing


# ---------------------------------------------------------------- 笔画处理

def warp(a, rng, amp=2.0):
    h, w = a.shape
    if h < 12 or w < 12:
        return a

    def fld(cells=(2, 2), sigma=1.8):
        g = rng.uniform(-1.0, 1.0, (cells[0] + 1, cells[1] + 1)).astype(np.float32)
        im = Image.fromarray(((g + 1) * 127.5).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        return ndimage.gaussian_filter(np.asarray(im).astype(np.float32) / 127.5 - 1.0, sigma)

    dx, dy = fld() * amp, fld() * amp
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = np.clip(xx + dx, 0, w - 1.001)
    sy = np.clip(yy + dy, 0, h - 1.001)
    x0, y0 = sx.astype(np.int32), sy.astype(np.int32)
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    fx, fy = sx - x0, sy - y0
    return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x1] * fx * (1 - fy) +
            a[y1, x0] * (1 - fx) * fy + a[y1, x1] * fx * fy)


def restroke(a, rng):
    """局部粗细：同一支笔这一次下笔重了点、那一次轻了点。

    ⛔ **这里改过一次做法，别再改回去**（2026-09-17）：
    老做法是"把扩张出来的边缘像素整圈点亮"（`out[md & ~m] = 0.80`）。
    那不是在调粗细，是在**把字喂胖一圈** —— 一个字周围多出一层实心边，
    两遍跑下来墨量差 12.3%，放大一眼就能看出"这个字明显比那个粗"，
    用户原话是「你这个参数太明显了……粗细变化幅度太大了」。
    现在改成两步，都在墨迹**内部**做文章：
      ① 按低频噪声场给整字乘一个 1 + δ·max(f,0) 的增益，δ 只有 6%；
      ② 顺着同一个噪声场的正负，把墨迹里较淡的像素轻微加重、较浓的轻微减淡，
         幅度取 STROKE_LO/HI —— 看上去是"这一笔蘸墨多、那一笔少"，
         而不是"整字胖瘦不同"。
    这两步都不向外扩张轮廓，所以字的外形分毫不动，只有墨色深浅在动。
    """
    m = a > 0.45
    if m.sum() < 12:
        return a
    h, w = a.shape
    g = rng.uniform(-1.0, 1.0, (3, 3)).astype(np.float32)
    f = ndimage.gaussian_filter(np.asarray(Image.fromarray(((g + 1) * 127.5).astype(np.uint8)).resize((w, h), Image.BICUBIC)).astype(np.float32) / 127.5 - 1.0, STROKE_SIGMA)
    # ⚠️ 必须去均值。噪声明明是零均值的，但只有 3×3 个随机点、又过了一遍高斯，
    # 落到具体一个 35×34 的字形上时均值可以偏到 +0.1 左右 ——
    # 于是 `1 + BIAS * max(f, 0)` 就变成**整字一起变浓**，实测 −6.8%/+7% 的
    # 单边偏移，两遍下来差 7.3%。去均值之后才真的是"有浓有淡"。
    f = f - float(f.mean())
    out = a.copy()
    # ① 整体增浓（只有正的噪声才加，避免整体变浅 —— 变浅会让笔画发灰像铅笔）
    out = out * (1.0 + STROKE_BIAS * np.maximum(f, 0.0))
    # ② 墨量重分配：只动已有墨迹，不动外形。
    #    "较淡的像素"= 有墨但没到实心（0.05~0.9 之间），这部分是笔画的过渡边。
    soft = (a > 0.05) & (a < 0.90)
    heavy = a >= 0.90
    out[soft & (f > STROKE_GATE)] *= STROKE_HI / max(0.35, STROKE_HI)
    out[heavy & (f < -STROKE_GATE)] *= STROKE_LO
    return np.clip(out, 0, 1)


def shift(a, dy, dx):
    out = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
    if dy > 0:
        out[:dy, :] = 0
    elif dy < 0:
        out[dy:, :] = 0
    if dx > 0:
        out[:, :dx] = 0
    elif dx < 0:
        out[:, dx:] = 0
    return out


def elongate(a, rng, amount=None):
    """沿一个主方向把笔画拉长一点，模拟收笔出锋（长横、撇捺带尾）。

    ⛔ **拉长之后必须把墨量拉回原量，这一步不能省**（2026-09-17 实测）：
    出锋是"把字形沿某方向搬运再叠上去"，所以方向不同、墨量增幅差得很远 ——
    实测同一个字：水平方向 +18.3%、对角方向 +32.3%，**光方向选择就造成 14 个点
    的墨量差**。这比粗细扰动本身大了四五倍，用户看到的"一片字有粗有细"
    绝大部分来自这里，而不是来自 restroke。
    做法：算完出锋后按 `目标墨量 / 实际墨量` 整体缩一次，目标墨量取
    "原字形墨量 × (1 + 一点点)"，那一点点才是这次出锋允许带来的增浓。
    缩的是**整体不透明度**，字的外形分毫不改 —— 出锋的尾巴还在，只是淡下去，
    正好就是真实书写里"收笔那一带墨越来越干"的样子。
    """
    amount = STRETCH if amount is None else amount
    h, w = a.shape
    if h < 12 or amount <= 0:
        return a
    dirs = [(0, 1), (0, 1), (1, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, 1)]
    dy, dx = dirs[int(rng.integers(0, len(dirs)))]
    n = max(1, int(h * amount * float(rng.uniform(ELONGATE_LO, ELONGATE_HI))))
    out = a.copy()
    for k in range(1, n + 1):
        decay = (1.0 - k / float(n + 1)) * 0.85
        out = np.maximum(out, shift(a, dy * k, dx * k) * decay)
    out = np.clip(out, 0, 1)
    # 墨量归一：出锋允许把墨量抬 ELONGATE_INK 倍，多出来的部分整体压回去。
    base = float(a.sum())
    cur = float(out.sum())
    if base > 1e-6 and cur > 1e-6:
        want = base * (1.0 + ELONGATE_INK)
        out = np.clip(out * min(1.0, want / cur), 0, 1)
    return out


def vscore(a, thr):
    """当前角度下"长竖画"的总长度。

    只累加长度 >= thr 的竖直连续段，因此：
      · 有长竖画的字：竖画越竖直，得分越高 —— 曲线有内部极大值，峰位即倾角；
      · 没有长竖画的字（如 一、二、· 等）：得分恒为 0，不会被误转。

    thr 必须由调用方按**原字高**算好并全程固定，不能随旋转后的画布变化 ——
    旋转用 expand=True 会放大画布，若 thr 跟着涨，大角度下落墨游程会短于阈值、
    得分直接塌成 0，判据在最该起作用的 8~15 度区间变成瞎子。
    """
    # 向量化：先逐行递推"每列自顶向下连续落墨的长度"，再取每个游程的终点值。
    # 原来是 h×w 次 Python 循环（一格一次），现在是 h 次整行 numpy 运算，结果完全一致。
    b = a > 0.5
    if not b.any():
        return 0.0
    h, w = b.shape
    run = np.zeros((h, w), dtype=np.int32)
    acc = np.zeros(w, dtype=np.int32)
    for i in range(h):
        acc = (acc + 1) * b[i]
        run[i] = acc
    nxt = np.empty((h, w), dtype=bool)
    nxt[:-1] = b[1:]
    nxt[-1] = False
    ends = run[b & ~nxt]
    if ends.size:
        return float(ends[ends >= thr].sum())
    return 0.0


def lean_report(a, span=None, step=1.0, up=4, thr_frac=0.35):
    """返回 (校正角, 置信度)：竖画倾角，以及该结论有多可信。

    置信度 = 峰值得分 / 该曲线自身的中位得分 - 1，衡量"峰从背景里凸出来多少"。

    为什么分母不能用"0 度时的得分"：一个已经歪了 8 度的竖画，在 0 度下得分就是 0，
    用它做分母会让置信度恒为 0，于是最该被扶正的字反而被跳过。
    用整条曲线的中位作背景，才能同时覆盖"本来很正"和"明显歪了"两种情况。

    ⛔ **这里做过"两级搜索"提速，被证伪后回退了**（2026-09-17）：
    想法是先用 step=3° 粗搜找峰、再在峰旁 ±2° 内以 0.25° 细化，把 37 次旋转压到 30 次。
    实测结果否掉了它 ——
      真值 +3° → 逐度扫描 0°（误差 3.0，恰好压线过关） vs 两级搜索 -0.25°（误差 3.25，不过）
      真值 -5° → 逐度扫描 +2°（误差 3.0，过关）        vs 两级搜索 +0.5°（误差 4.5，不过）
    selftest 的倾角断言因此由通过转为失败。原因：vscore 的峰很窄（约 ±2°），
    粗搜一旦跨过真峰就会落到旁边的噪声肩上，细化只能在这个错峰附近打磨，
    救不回来。这条曲线**不值得为省 7 次旋转付出精度**，故保持逐度扫描。
    （原注释里"实测结论与逐度扫描一致"是错的，已删。）
    """
    span = LEAN_SPAN if span is None else span
    if a.max() <= 0.02 or a.shape[0] < 14:
        return 0.0, 0.0
    # 阈值按原字高固定，全程不变（见 vscore 注释）
    thr = max(4, int(thr_frac * a.shape[0] * up))
    src = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    src = src.resize((src.width * up, src.height * up), Image.BILINEAR)
    ths, vals = [], []
    th = -span
    while th <= span + 1e-6:
        arr = np.asarray(src.rotate(float(th), resample=Image.BILINEAR,
                                    fillcolor=0, expand=True), dtype=np.float32) / 255.0
        ths.append(float(th))
        vals.append(vscore(arr, thr))
        th += step
    vals = np.asarray(vals, dtype=np.float32)
    peak = float(vals.max())
    if peak <= 0:
        return 0.0, 0.0
    bg = float(np.median(vals))
    # 背景为 0（尖峰孤立）时置信度记为满，避免除零
    conf = peak / bg - 1.0 if bg > 0.05 * peak else 1.0
    return float(ths[int(np.argmax(vals))]), float(conf)


def upright(a, tol=None, cap=None, gain=None, gid=None):
    """扶正竖画：只有在**确认存在长竖画且确实歪了**时才旋转，幅度受限。

    三重闸门（任一不过就不动）：
      1. 置信度 gate —— 峰值得分须比原图高 gain 以上，证明"确实找到了长竖画"；
      2. 容差 gate   —— 倾角须超过 tol（正常手写的 1° 左右抖动不算歪）；
      3. 幅度 cap    —— 单次最多转 cap 度。
    为什么要设上限：全局旋转无法同时满足竖画与横画。实测把竖画扶正 8°，
    横画得分会从 320 掉到 0 —— 横画被带斜。所以只能"少量、只对真歪的字做"。

    gid 是**源字形身份**（如 (字, 实例下标)），由调用方（vary ← _render_line ← Picker）
    给出。给了它，同一实例在一页里出现 50 次只算 1 次 lean_report；
    不给则退回"按传入数组的内容指纹缓存"——那种键在随机形变之后每次都不同，
    命中率≈0，等于没缓存（而结果完全正确，从输出上根本看不出来）。

    ⚠ 语义差异要明确接受：有 gid 时，缓存的是该实例**第一次**（某一随机扰动下）
    算出的倾角，之后同一实例的所有位置复用它；没有 gid 时每次都是各自扰动后的
    倾角。前者更接近"这个字本来就歪"的物理含义，且是全页一致。
    """
    tol = LEAN_TOL if tol is None else tol
    cap = LEAN_CAP if cap is None else cap
    gain = LEAN_GAIN if gain is None else gain
    if tol <= 0 or cap <= 0 or a.max() <= 0.02 or a.shape[0] < 14:
        return a, 0.0
    if gid is not None:
        key = (gid, round(float(tol), 4), round(float(cap), 4), round(float(gain), 4))
    else:
        key = (a.shape, round(float(a.sum()), 2), round(float((a * a).sum()), 2),
               round(float(tol), 4), round(float(cap), 4), round(float(gain), 4))
    cached = _UPRIGHT_CACHE.get(key)
    if cached is None:
        CACHE_MISS[0] += 1
        cached = lean_report(a)
        if len(_UPRIGHT_CACHE) > 4096:
            _UPRIGHT_CACHE.clear()
        _UPRIGHT_CACHE[key] = cached
    else:
        CACHE_HIT[0] += 1
    fix, conf = cached
    if conf < gain or abs(fix) <= tol:
        return a, 0.0
    applied = float(np.clip(fix, -cap, cap))
    if abs(applied) < 0.4:
        return a, 0.0
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    a = np.asarray(im.rotate(applied, expand=True,
                             resample=Image.BILINEAR)).astype(np.float32) / 255.0
    return a, applied


def vary(a, rng, gid=None):
    """把一个字形实例化成"这一次写出来的那个样子"。

    扰动的**分量次序有意如此**，别随手调换：
      ① restroke 粗细 —— 最轻，只动边缘
      ② warp 形变     —— 手写的笔画不是几何精确的
      ③ elongate 出锋 —— 长横撇捺的收笔
      ④ 整字微旋转    —— 手腕抖动
      ⑤ upright 扶正  —— 先抖，再把真歪的字拉回来（顺序反了会把抖出来的歪角当成真歪）
      ⑥ 整体缩放      —— 最后一步，所以不会破坏前面各步的像素尺度假设

    gid 是源字形身份，透传给 upright() 的倾角缓存 —— 必须由调用方给：
    到这一步 a 已经过随机形变，从 a 本身再也反推不出"这是哪个实例"，
    而按内容指纹做键的缓存命中率恒为 0（见 upright 的说明）。
    """
    if a.max() <= 0.02:
        return a * 0
    a = restroke(warp(a, rng, amp=1.6), rng)
    if rng.random() < ELONGATE_P:
        a = elongate(a, rng)
    if ROTATE > 0:
        # 整字微旋转：模拟落笔时手腕的自然抖动。expand 必须为 False 保持画布尺寸，
        # 否则字宽会按对角线膨胀，直接破坏排版。放在扶正之前 —— 先抖，再扶正。
        ang = float(rng.uniform(-ROTATE, ROTATE))
        if abs(ang) > 1e-3:
            im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
            a = np.asarray(im.rotate(ang, resample=Image.BILINEAR,
                                     fillcolor=0)).astype(np.float32) / 255.0
    a, _ap = upright(a, tol=LEAN_TOL, gid=gid)
    h, w = a.shape
    # 整体缩放的抖动幅度。原来 ±(4~5)%，现在收到 ±1.5%。
    # 理由同 ROTATE：这一层只管"手写字不会每个都一样大"，不管"观感差异"。
    # 缩放是**等比**的，墨量跟着面积走，所以 1.5% 的边长差 = 3% 的墨量差，
    # 再往上就会跟粗细扰动叠在一起、把整页的字弄得胖瘦不齐。
    sx, sy = float(rng.uniform(0.985, 1.015)), float(rng.uniform(0.985, 1.015))
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((max(2, int(w * sx)), max(2, int(h * sy))), Image.BILINEAR)).astype(np.float32) / 255.0


class Picker:
    """同字多实例轮转；间隔不足才复用。"""

    def __init__(self, lib, min_gap=200.0, rng=None):
        self.lib = lib
        self.min_gap = min_gap
        self.rng = rng or np.random.default_rng()
        self.last = {}

    def take(self, ch, x, y):
        pool = self.lib[ch]
        idx = int(self.rng.integers(0, len(pool)))
        if len(pool) > 1:
            rec = self.last.get(ch)
            if rec is not None:
                li, lx, ly = rec
                near = (x - lx) ** 2 + (y - ly) ** 2 < self.min_gap ** 2
                if near and idx == li:
                    idx = (idx + 1) % len(pool)
        self.last[ch] = (idx, x, y)
        # 连"源字形身份"一起返回：倾角缓存必须锚在源字形上，而 vary() 之后
        # 数组已被随机形变，从数组本身再也反推不出这是哪个实例。
        return (ch, idx), pool[idx]


def blend(arr, a, x, y):
    """把 alpha 图 a 以落点 (x, y) 混到 arr 上。

    ⛔ **返回的是"实际落下多少"，不是 arr** —— 这是必须的：调用方（_render_line）
    落墨后要记 recs 用于字数对账，而原来的 blend 在完全越界时**静默返回原数组**，
    调用方无从判断。后果是落点算出画布外的字被记进 recs、实际墨迹不在图上，
    字数对账还报告"相符"——用户以为内容都在，其实少了几笔。

    返回 (是否落下, 横向被裁掉的比例)：
      · 完全在画布外或被裁到零宽 → (False, 1.0)
      · 部分落下                 → (True, 裁剪比例)  比例超过阈值调用方该告警
    """
    if x < 0 or y < 0 or y >= arr.shape[0] or x >= arr.shape[1]:
        return False, 1.0
    ah, aw = a.shape
    y1, x1 = min(arr.shape[0], y + ah), min(arr.shape[1], x + aw)
    ah2, aw2 = y1 - y, x1 - x
    if ah2 <= 0 or aw2 <= 0:
        return False, 1.0
    arr[y:y1, x:x1] = (arr[y:y1, x:x1] * (1 - a[:ah2, :aw2][..., None])
                       + np.array(INK) * a[:ah2, :aw2][..., None])
    # 裁掉的比例：宽高各算一次取大的 —— 一个字被切掉一半宽比被切掉一半高更伤。
    cut = max(1.0 - aw2 / float(aw), 1.0 - ah2 / float(ah))
    return True, float(cut)


# 一个字被裁掉多少比例才算"落墨不完整"，值得报出来。
# 0.5 的依据：裁掉一半以上时，成品上的字形已经明显不是那个字了，
# 而用户对着一张 1200 字的成品图根本看不出哪几笔短了 —— 必须程序说。
# 小于这个比例的裁剪（切掉一两像素边缘）属正常排版容差，不报。
BLEND_CLIP_WARN = 0.5


# ---------------------------------------------------------------- 纸面

def paper_profile(background):
    A = np.asarray(background).astype(np.float32)
    H, W = A.shape[:2]
    L = A.mean(axis=2)
    chroma = A.max(axis=2) - A.min(axis=2)
    mask = (L > 150) & (chroma < 60)
    if mask.mean() > 0.985:
        return [(y, 0, W - 1) for y in range(0, H, 4)], [(x, H - 1) for x in range(0, W, 8)], mask
    mask = ndimage.binary_closing(mask, structure=np.ones((15, 15)))
    lb, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(mask, lb, range(1, n + 1))
        mask = lb == (int(np.argmax(sizes)) + 1)
    mask = ndimage.binary_fill_holes(mask)
    rows = []
    for y in range(0, H, 4):
        xs = np.where(mask[y])[0]
        if len(xs) < 40:
            rows.append((y, -1, -1))
        else:
            xs = np.sort(xs)
            rows.append((y, int(xs[int(len(xs) * 0.01)]), int(xs[int(len(xs) * 0.99)])))
    bot = []
    for x in range(0, W, 8):
        ys = np.where(mask[:, x])[0]
        bot.append((x, int(ys.max()) if len(ys) > 5 else -1))
    return rows, bot, mask


# 纸面轮廓量不出来时的保守可用范围（像素）。
# 原来兜底给 (40, 10**6) —— 可用宽度因此变成一百万，排版以为"一行能写到底"，
# 于是整篇不折行、字全挤成一行冲出画布。给一个真实的纸宽上限才安全：
# 宁可折行折早一点，也不能把内容排到画面外。
PAPER_FALLBACK = (40, 1200)


def range_at(rows, y):
    """取 y 这一行的纸面可用横向范围 (left, right)；量不出来时给保守兜底。"""
    best = None
    for yy, lo, hi in rows:
        if lo < 0:
            continue
        d = abs(yy - y)
        if best is None or d < best[0]:
            best = (d, lo, hi)
    return PAPER_FALLBACK if best is None else (best[1], best[2])


def bottom_at(bot, x):
    """取 x 附近的纸面下缘 y；量不出来时给保守兜底（而非 10**6）。

    兜底给 10**6 时，"是否超出一页"的判断永远为假 —— 内容被画到画面外
    也不会有任何提示。给一个有限值（fallback 上限）能让截断检测正常触发。
    """
    vals = [by for bx, by in bot if by > 0 and abs(bx - x) < 40]
    return max(vals) if vals else PAPER_FALLBACK[1] * 2


# ---------------------------------------------------------------- 自动排版（页面流）

# 一页末尾留多少像素以内就不算"大片空白"。
# 为什么是 240：这是"再塞得下一行正文"的量级 —— 正文行高 40、行距 1.8 倍
# 约 82px，240px 够放两行还富余。低于这个数说明页面本来就排满了，
# 再去把下一页的内容搬上来只会让两页都变挤。
FLOW_SLACK = 240

# 空白段判决的最小高度。比这更矮的空隙是行距自然形成的，不算"大段空白"。
FLOW_MIN_GAP = 300


# 字距的期望倍数：_render_line 里 gap = h * 0.22 * uniform(0.85, 1.2)。
GAP_MEAN = 0.22 * ((0.85 + 1.2) / 2.0)
# 量行高这一侧的余量：vary() 的尺寸抖动会改宽高比（宽 0.96~1.05、高 0.96~1.04），
# 取最宽实例 + 再乘这个系数，保证估算落在"不窄于实际"的一侧。
GAP_EST_MARGIN = 1.10


def token_metrics(c, h, w_ratio=None, sub=False):
    """一个 token 在纸上占多宽 —— **量行高与贴字共用的唯一口径**。

    返回 (类别, 宽度 px, 有效字号 px)：类别是 "space" / "dot" / "miss" / "glyph"，
    只有 "glyph" 会用到有效字号（贴字时要按它算落点高度）。

    ⛔ 为什么必须共用：原来量行高用一张固定倍率表（汉字 0.98h、字母 0.62·LATIN·h…），
    贴字用"实测字形宽 + 随机字距"，而**字距那一项在估算表里根本没有** ——
    汉字实际推进 1.09~1.26h、估算只有 0.98h，量出来的行数系统性偏少
    （每 5 行少算 1 行），分页偏乐观，最后只能靠"截断"收场。
    两处各写一套必然漂移，所以口径归到这里一处。w_ratio 是字形宽高比（宽/高）。
    """
    if c == " ":
        return "space", h * 0.30, None
    if w_ratio is None:
        # 库里没有这个字符：标点走程序画点兜底，其余算"缺字"
        if c in DOT_CHARS:
            return "dot", h * 0.40, None
        return "miss", h * 0.60, None
    size = h * (LATIN if c.isascii() else 1.0)
    if sub:
        size *= 0.55
    return "glyph", max(2.0, float(w_ratio) * size), size


def glyph_wh_ratio(lib, c):
    """字形宽高比（宽/高），取该字**所有实例里最宽的那个**。

    取最宽的而不是取中位：量行高宁可估宽一点（早折行），
    不能估窄 —— 估窄了会排完了才发现折行，版面就对不上了。
    """
    if not lib:
        return None
    pool = lib.get(c)
    if not pool:
        return None
    best = 0.0
    for arr in pool:
        sh = np.asarray(arr).shape
        if len(sh) >= 2 and sh[0] > 0:
            best = max(best, float(sh[1]) / float(sh[0]))
    return best or None


def measure_line_span(tokens, h, rows, bot, y, x_left, avail, no_wrap, indent=0.0, lib=None):
    """算出这一行文字实际要占的纵向跨度（不落墨）。

    只做宽度累加，不做字形挑选 —— 排版只需要"会不会折行、折几行"，
    不需要知道每个字长什么样。所以**不能**用 _render_line：那会推进 picker
    的随机状态，导致"量一遍、画一遍"两次结果不一致（行高变了，排版就对不上）。

    ⚠ 但**宽度口径必须与 _render_line 一致**（两边都走 token_metrics）。
    2026-09-18 之前这里用固定倍率表、且漏了字距项，实测低估 10~22%。
    lib 传 None 时退回旧表（只为兼容老调用；正式路径必须传 lib）。
    """
    lines = 0
    cur = indent
    yy = y
    a = avail
    for c, s, _o in tokens:
        wr = glyph_wh_ratio(lib, c)
        kind, tw, _sz = token_metrics(c, h, wr, s)
        # 字距只有"真贴字形"这一支才有 —— 与 _render_line 的 x += tw + gap 对应
        gap = h * GAP_MEAN * GAP_EST_MARGIN if kind == "glyph" else 0.0
        w = tw + gap
        if cur + w > a and cur > indent + 1:
            if no_wrap:
                # 整行会被缩到一行里，跨度就是一行，不必再往下量
                break
            lines += 1
            yy += int(h * 1.8) + 10
            lo, hi = range_at(rows, yy + h // 2)
            a = max(60, (hi - 20) - (lo + 20))
            cur = 0.0
        cur += w
    return (lines + 1), (yy + int(h * 1.8) + 10)


def plan_pages(items, rows, bot, page_h, no_wrap):
    """把若干"行"贪心装进页面，尽量不让任何一页尾巴上留大片空白。

    items：[(spec 原文行, 纵向跨度)] 的有序表。
    返回：每页包含的行索引列表，例如 [[0,1,2],[3,4]]。

    为什么用贪心而不是均分：均分会让"3 行 + 1 行长行"这种组合变成每页各 2 行，
    长行那页照样留白。贪心（能塞就塞）才是"把后续内容提上来填掉空白"的本意。
    """
    pages, cur, used = [], [], 0
    for idx, (_ln, span) in enumerate(items):
        if cur and used + span > page_h:
            pages.append(cur)
            cur, used = [], 0
        cur.append(idx)
        used += span
    if cur:
        pages.append(cur)
    return pages or [[]]


def blank_bands(canvas_arr, rows, bot, min_h=None, ink_thresh=0.008):
    """找出一页成品里"成片没有墨迹"的纵向区段。

    返回 [{"y0":.., "y1":.., "h":..}]，按从上到下排序。

    为什么按行统计墨迹、而不是按连通域找字：这一页的文字是逐字贴上去的，
    连通域会把一个字拆成好几块、又会把相邻行连成一片，判"哪里是空白段"
    反而失准。逐行扫"这一行还有没有墨"既便宜又直接对应用户看到的东西。

    纸面本身的横线/红格不是墨迹 —— 判据用**相对整页**的暗度差，
    纸纹是均匀的浅色，不会越过 ink_thresh。
    """
    if min_h is None:
        min_h = FLOW_MIN_GAP
    A = np.asarray(canvas_arr)
    if A.ndim == 3:
        L = A.astype(np.float32).mean(axis=2)
    else:
        L = A.astype(np.float32)
    Hs, Ws = L.shape
    # 只统计纸面可用横向范围，避免把桌子/阴影算成"有内容"
    lo, hi = range_at(rows, Hs // 2)
    if lo < 0 or hi <= lo:
        lo, hi = 0, Ws
    lo = max(0, min(lo, Ws - 1))
    hi = max(lo + 1, min(hi, Ws))
    strip = L[:, lo:hi]
    if strip.size == 0:
        return []
    # 每行的"墨迹比例"：暗于本页中位亮度一定的像素算墨
    bgv = float(np.median(strip))
    inkrow = (strip < bgv - 60).mean(axis=1)
    has = inkrow > ink_thresh
    bands = []
    run = None
    for yy in range(Hs):
        if not has[yy]:
            if run is None:
                run = yy
        else:
            if run is not None:
                if yy - run >= min_h:
                    bands.append({"y0": run, "y1": yy, "h": yy - run})
                run = None
    if run is not None and Hs - run >= min_h:
        bands.append({"y0": run, "y1": Hs, "h": Hs - run})
    return bands


def render_flow(spec_path, lib_path, bg_path, out_path, seed=20260914,
                no_wrap=False, slack=None):
    """自动排版渲染：把整篇内容按"填满一页再进下一页"重新分配。

    与 render() 的区别：render() 是"一行接一行往下写，写不下就截断"，
    内容短时最后一页会留一大片空白；render_flow() 先把每行量一遍，
    再贪心装箱到多页，空白被后续内容填掉。

    ⚠ 这里的页是**同一张纸面模板的多份拷贝**，不是把上一页的字搬到下一页 ——
    所以每页都是完整的一张纸（纸面纹理/边缘都在），不会出现"半张纸"。

    返回 (输出文件列表, 被截断的页码列表)。页码非空 = 有内容没落到任何一页上，
    调用方必须给出非零退出码 —— 详见下面"截断必须吵"的说明。
    """
    lib = load_library(lib_path)
    bg = open_rgb(bg_path)
    rows, bot, _ = paper_profile(bg)
    W, H = bg.size
    # 纸面下缘（可用范围末端）。量不出来时按画布高度兜个底。
    page_bot = bottom_at(bot, W // 2)
    if page_bot > H:
        page_bot = H - 40
    START_Y = 90
    page_h = max(200, page_bot - START_Y - 60)

    raw = [ln.rstrip("\n") for ln in read_text(spec_path).splitlines() if ln.strip()]
    # 注释行不参与渲染（与 render 同口径），但它们仍要留在原位 ——
    # 所以这里只标记"哪些行要排"，行号保持与 raw 对齐。
    #
    # ⚠ `#gap` / `#wave` 这些**是指令不是注释**：render() 里它们在 `#` 分支之前
    # 就被拦下了。这里若拿 `startswith("#")` 一刀切，`#gap` 会被当成注释吃掉 ——
    # 版面少了留白，而且**不报错**（只是字挤在一起）。所以显式排除指令前缀。
    text_idx = [i for i, ln in enumerate(raw)
                if not ln.startswith("#") or ln.startswith(("#gap", "#wave"))]

    missing = {}
    page_chars = set()
    for i in text_idx:
        # 指令行不是正文，别把它当字去查库 —— 否则 `#gap 40` 会报
        # "缺字：# g a p"，用户按这个去补字就白跑一趟。
        if raw[i].startswith(("#gap", "#wave")):
            continue
        body = raw[i][1:] if raw[i].startswith("*") else raw[i]
        if body.startswith(">"):
            body = body[1:]
        for ch in body:
            if ch in SKIP_CHARS:
                continue
            if ch not in lib:
                missing[ch] = missing.get(ch, 0) + 1
            else:
                page_chars.add(ch)
    if missing:
        print("缺字：%s" % " ".join(sorted(missing)), file=sys.stderr)

    # 全篇共用一个墨量标尺。**这里必须是全篇而不是逐页**：
    # 逐页算的话，第 1 页和第 2 页的标尺不同、同一批字在两页上粗细就不一样，
    # 而用户是连着翻这两页看的。全篇一个标尺，页与页之间才连得上。
    ink_target = page_ink_target(lib, page_chars)

    # ---- 第一趟：只量不画 ----
    # ⚠ items 里存的是**每一行的增量高度**（这一行自己占多少），不是累计 y。
    # 装箱要的是增量；存累计值会让"能塞就塞"的判据变成"累计 > 页高"，
    # 结果第一行就超限、每页只放一行（实测就是这样把 5 行拆成 2 页的）。
    items = []
    y = 90
    for i in text_idx:
        ln = raw[i]
        if ln.startswith("#gap"):
            arg = ln.split()[1] if len(ln.split()) > 1 else ""
            try:
                d = int(arg)
            except ValueError:
                print("提示：#gap 后面的数字看不懂（%r），本行按默认 40 处理。" % arg,
                      file=sys.stderr)
                d = 40
            items.append((i, d))
            y += d
            continue
        if ln.startswith("#wave"):
            items.append((i, 150))
            y += 150
            continue
        h = 40
        lead = 0.0
        if ln.startswith(">"):
            lead = 40 * 2.0
            ln = ln[1:].lstrip()
        if ln.startswith("*"):
            h, ln = 34, ln[1:].strip()
        lo, hi = range_at(rows, y + h // 2)
        avail = max(60, (hi - 20) - (lo + 20))
        toks = _tokens(ln)
        _nl, span_end = measure_line_span(toks, h, rows, bot, y, lo + 20, avail,
                                         no_wrap, indent=lead)
        span = span_end - y          # 增量
        items.append((i, span))
        y = span_end

    pages = plan_pages(items, rows, bot, page_h, no_wrap)

    # ---- 第二趟：按装箱结果逐页渲染 ----
    outs = []
    base, ext = os.path.splitext(out_path)
    n_expect_all = n_prog_all = 0
    reports = []
    # 被截断的页：(页码, 原文行号(0基), 没渲染的行数)
    # 为什么必须记下来：第一趟已把行分配到固定的页，某页装不下时那些行
    # **不会**被挪到后面的页，也不会出现在任何输出里。原实现只把 truncated
    # 置真然后 break，之后就再没人读它 —— 用户拿到的是少了几行的成品，
    # 而那一页的审计还写着"已压到阈值内"。（单页 render() 早在 v0.1 就修好了
    # 这个问题，render_flow 是新路径，把这条保证丢了。）
    trunc_pages = []
    for pi, page_idx in enumerate(pages):
        if len(pages) == 1:
            op = out_path
        else:
            op = "%s_p%d%s" % (base, pi + 1, ext)
        rng = np.random.default_rng(seed + pi * 1000)
        picker = Picker(lib, rng=rng)
        canvas = np.asarray(bg).astype(np.float32)
        recs, clipped = [], []
        # ink_target 是全篇共用的（上面算好的），这里不再按页重算 —— 见那里的注释。
        y = 90
        rendered_lines = 0
        truncated = False
        truncated_at = None       # (原文行号(0基), 本页没能渲染的行数)
        n_expect = n_prog = 0
        for pos, i in enumerate(page_idx):
            ln = raw[i]
            if ln.startswith("#gap"):
                arg = ln.split()[1] if len(ln.split()) > 1 else ""
                try:
                    y += int(arg)
                except ValueError:
                    y += 40
                continue
            if ln.startswith("#wave"):
                lo, hi = range_at(rows, int(y + 60))
                tmp = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
                dr = ImageDraw.Draw(tmp)
                waveform(dr, lo + 40, y + 10,
                         min(hi - 40, lo + 760) - (lo + 40), 110, rng)
                canvas = np.asarray(tmp).astype(np.float32)
                y += 150
                continue
            h = 40
            lead = 0.0
            if ln.startswith(">"):
                lead = 40 * 2.0
                ln = ln[1:].lstrip()
            if ln.startswith("*"):
                h, ln = 34, ln[1:].strip()
            toks = _tokens(ln)
            lo, hi = range_at(rows, y + h // 2)
            avail = max(60, (hi - 20) - (lo + 20))
            if bottom_at(bot, int(lo + 30)) < y + int(h * 1.4) + 20:
                truncated = True
                truncated_at = (i, len(page_idx) - pos)
                break
            x_left = float(lo + 20)
            base_y = y + h
            rest = list(toks)
            first = True
            while rest:
                canvas, rr, used, nch, xend, cut_chars = _render_line(
                    canvas, rest, lib, x_left, base_y, h, avail, rng, picker, y,
                    indent=(lead if first else 0.0),
                    drift_amp=BASE_DRIFT, no_wrap=no_wrap, ink_target=ink_target)
                recs.extend(rr)
                clipped.extend(cut_chars)
                if used <= 0:
                    break
                for c, _s, _o in rest[:used]:
                    if c == " " or c in SUP_CHARS:
                        continue
                    if c in SKIP_CHARS or c in lib:
                        n_expect += 1
                        if c in DOT_CHARS and c not in lib:
                            n_prog += 1
                rest = rest[used:]
                if not rest:
                    break
                y += int(h * 1.8) + 10
                rendered_lines += 1
                lo, hi = range_at(rows, y + h // 2)
                avail = max(60, (hi - 20) - (lo + 20))
                if bottom_at(bot, int(lo + 30)) < y + int(h * 1.4) + 20:
                    truncated = True
                    # 折行折到这一行的后半段时也放不下了：同样要记账（这一行算没渲染完）
                    truncated_at = (i, len(page_idx) - pos)
                    break
                x_left = float(lo + 20)
                base_y = y + h
                first = False
            if truncated:
                break
            y += int(h * 1.8) + 10
            rendered_lines += 1
        img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(0.15))
        img.save(op)
        n_lost = len(clipped)
        rep, nbad = audit(recs, expected=max(0, n_expect - n_lost), n_prog=n_prog)
        used_h = y
        # 本页尾部的剩余空间。这就是"大片空白"的量化口径 ——
        # 自动排版存在的意义就是把它压到 FLOW_SLACK 以内。
        tail = max(0, page_h + 60 - used_h)
        # 实测一遍空白段：上面那个 tail 是"按排版游标算的"，
        # 这里再拿**成品像素**复核一次。两者不一致说明有行没落墨
        # （落点出画布、或纸面量错），必须报出来而不是信游标。
        bands = blank_bands(canvas, rows, bot)
        tail_band = [b for b in bands if b["y1"] >= H - 6]
        px_tail = tail_band[-1]["h"] if tail_band else 0
        head = ("【自动排版 第 %d/%d 页】本页 %d 行、%d 字；"
                "正文占 %dpx / 页高 %dpx，页尾空 %dpx（实测 %dpx）%s\n"
                % (pi + 1, len(pages), len(page_idx), len(recs),
                   used_h, page_h + 60, tail, px_tail,
                   "（已压到阈值内）" if tail <= (slack or FLOW_SLACK) else "（仍偏大）"))
        if bands:
            head += ("  本页空白段 %d 处：%s\n"
                     % (len(bands),
                        "、".join("%d..%d(%dpx)" % (b["y0"], b["y1"], b["h"])
                                  for b in bands[:6])))
        reports.append(head + rep)
        outs.append(op)
        n_expect_all += n_expect
        n_prog_all += n_prog
        if truncated_at is not None:
            _ln_no, _n_miss = truncated_at
            trunc_pages.append((pi + 1, _ln_no, _n_miss))
            print("警告：自动排版第 %d/%d 页装不下 —— 从原文第 %d 行起、共 %d 行没有渲染，"
                  "它们不会出现在任何一页上。" % (pi + 1, len(pages), _ln_no + 1, _n_miss),
                  file=sys.stderr)
            print("  请加大 --flow-slack、把内容拆成多个文件，或适当精简。", file=sys.stderr)
            reports[-1] = ("【内容被截断】第 %d/%d 页还有 %d 行没能渲染"
                           "（从原文第 %d 行起），这些行不会出现在任何一页上。"
                           "请加大 --flow-slack、拆成多个文件，或精简内容。\n"
                           % (pi + 1, len(pages), _n_miss, _ln_no + 1)) + reports[-1]

    rp = os.path.splitext(out_path)[0] + ".audit.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write("".join(reports))
    print("".join(reports), end="")
    print("质检报告：%s" % rp)
    print("自动排版：%d 行 -> %d 页" % (len(items), len(pages)))
    if trunc_pages:
        print("⚠ %d 页装不下（第 %s 页），有内容没有落地 —— 详见上方告警与质检报告。"
              % (len(trunc_pages), "、".join(str(p) for p, _l, _n in trunc_pages)))
    for p in outs:
        print(p)
    return outs, trunc_pages


# ---------------------------------------------------------------- 波形

def waveform(dr, x0, y0, w, h, rng):
    def jline(pts):
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            n = max(2, int(np.hypot(bx - ax, by - ay) / 7))
            px, py = ax, ay
            for k in range(1, n + 1):
                t = k / float(n)
                qx = ax + (bx - ax) * t + rng.uniform(-0.9, 0.9)
                qy = ay + (by - ay) * t + rng.uniform(-0.9, 0.9)
                dr.line([px, py, qx, qy], fill=INK, width=int(rng.choice([2, 2, 2, 3])))
                px, py = qx, qy

    hi, lo = y0 + h * 0.16, y0 + h * 0.84
    pts = [(x0 + rng.uniform(-1.5, 1.5), lo + rng.uniform(-2.5, 2.5))]
    x, lvl = x0, lo
    while x < x0 + w - 12:
        x2 = min(x0 + w, x + float(rng.uniform(0.07, 0.34)) * w)
        pts.append((x2, lvl + rng.uniform(-1.5, 1.5)))
        lvl = (hi if lvl > y0 + h / 2 else lo) + rng.uniform(-2.5, 2.5)
        pts.append((x2, lvl))
        x = x2
    pts.append((x0 + w, lvl))
    jline(pts)


# ---------------------------------------------------------------- 渲染

def _audit_class(ch):
    """质检分组用的字类。分成 cjk / latin / punct 三组，各自取基准。

    为什么不能只分"中日韩 vs 其余"两类：拉丁字母在渲染里被 `LATIN = 0.95`
    缩放（见 render 里 `size = h * LATIN if c.isascii() else h`），
    而标点与汉字都是 1.0 倍。三类字的**渲染尺度各不相同**，
    尺度不同 → dens/sharp 的分布就不同 → 必须各自取基准。

    ⚠ 判据是 **CJK 码点区**，不是 `isascii()`：`⊕`（U+2295）、`→`、`·`
    这些符号都不是 ASCII，但它们的渲染尺度跟标点一样（1.0 倍），
    按 `isascii()` 一刀切会把它们归进"汉字组"当基准 —— 实测 `⊕` 就是
    这么混进去的，还把 `⊕` 报成了"汉字报警"。
    """
    if _is_cjk_char(ch):
        return "cjk"
    return "latin" if ch.isascii() and ch.isalnum() else "punct"


def audit(recs, expected=None, n_prog=0):
    """对成品逐字质检：糊（笔画粘连）/ 碎（笔画断裂）/ 浅（浓度不足）/ 缺。

    指标直接在**字形自身的 alpha** 上算（渲染时记录），不受相邻行墨迹干扰。

    expected 给出"本该渲染出多少个字"，n_prog 是其中由程序画的（标点等）。
    有了它才能做"应渲染 vs 实际"对账 —— 原来的审计只统计已渲染的字，
    内容被截断或缺字时审计里完全看不出来，用户以为一切正常。

    注：原来第一个参数是成品图（img），但从头到尾没被读过 —— 所有指标都来自
    recs。删掉它，免得让人以为"审计会再看一眼成品图"。
    """
    rows = []
    for r in recs:
        # 程序自己画的标点（prog=True）不参与糊/碎/浅判定 ——
        # 它是纯几何图形，用"手写笔画"的判据去量只会白白报错。
        #
        # 被跳过的字形（墨量过低，没有贴上去）直接落"跳过"这个结论：
        # 它**不在纸上**，不算合格；但它也不该被拿去比 dens/sharp
        # （没有墨迹可比），所以下面的判定循环会跳过已有结论的行。
        rows.append([r["ch"], r["dens"], r["sharp"],
                     "跳过" if r.get("skipped") else "OK", bool(r.get("prog"))])
    # 分组基准先声明在 if 外 —— 下面的报告段要用它印"各组判据"，
    # 声明在 if 里的话"页面上一个字都没有"时会 NameError。
    grp, ref = {}, {}
    if rows:
        hand = [r for r in rows if not r[4]]
        # ⛔ **基准必须按字类分组取**（2026-09-17 修）。
        #
        # 为什么：`dens` / `sharp` 这两个量**随字号系统变化**，
        # 而拉丁字母被 `LATIN = 0.95` 缩到 0.95 倍字号渲染 ——
        # 笔画在小一号的框里相对更宽 → dens 系统性偏高、sharp 系统性偏低。
        # 全页取一个中位当基准，就等于**拿两类不同尺度的字互比**：
        # 实测同一页 407 字里，拉丁组 237 字有 17 字越过 dens 判据，
        # 而标点组 163 字**一个都没有** —— 这不是字形质量差异，是度量口径差异。
        # 报出来的 5 个"糊"（= A B E ⊕）全是拉丁/符号，汉字 0 个，
        # 用户拿到报告会以为字母写坏了，其实图是好的。
        #
        # 分组后每组用**自己的中位**当基准，比的是"同类字之间谁异常"，
        # 这才是这条断言本来想量的事。
        #
        # ⚠ 兜底：某类字少于 MIN_GROUP_N 个时，它的中位没有统计意义
        # （1~2 个字时中位就是它自己，永远判"正常"），退回全页基准。
        MIN_GROUP_N = 4
        for r in hand:
            grp.setdefault(_audit_class(r[0]), []).append(r)
        for g, rr in grp.items():
            if len(rr) >= MIN_GROUP_N:
                ref[g] = (float(np.median([x[1] for x in rr])),
                          float(np.median([x[2] for x in rr])))
        ds_all = np.array([r[1] for r in hand], dtype=np.float32) if hand else np.array([1.0])
        sh_all = np.array([r[2] for r in hand], dtype=np.float32) if hand else np.array([1.0])
        d_all, s_all = float(np.median(ds_all)), float(np.median(sh_all))
        for r in hand:
            if r[3] != "OK":
                continue                     # 已被标成"跳过"：不参与糊/碎/浅判定
            ch, dens, sharp = r[0], r[1], r[2]
            d_ref, s_ref = ref.get(_audit_class(ch), (d_all, s_all))
            if dens < 0.03:
                r[3] = "缺"
            elif dens < 0.05 * d_ref:
                r[3] = "碎"
            elif dens > 1.75 * d_ref or sharp < 0.55 * s_ref:
                r[3] = "糊"
            else:
                r[3] = "OK"
    bad = [r for r in rows if r[3] != "OK"]
    by_ch = {}
    for ch, _d, _s2, v, _p in rows:
        if v != "OK":
            by_ch.setdefault(ch, {}).setdefault(v, 0)
            by_ch[ch][v] += 1
    n_hand = sum(1 for r in rows if not r[4])
    n_prog_drawn = len(rows) - n_hand
    lines = ["逐字质检：共 %d 字（手写 %d / 程序画点 %d），合格 %d，不合格 %d"
             % (len(rows), n_hand, n_prog_drawn,
                len(rows) - len(bad), len(bad))]
    if expected is not None:
        nb = n_prog if n_prog else 0
        n_have = len(rows)
        n_want = int(expected)
        if n_want == n_have:
            lines.append("字数对账：应渲染 %d 字（其中程序画点 %d 字），实渲染 %d 字 —— 相符。"
                         % (n_want, nb, n_have))
        else:
            gap = n_want - n_have
            lines.append("字数对账：应渲染 %d 字（其中程序画点 %d 字），实渲染 %d 字 —— "
                         "差 %d 字没有出现在成品上。"
                         % (n_want, nb, n_have, gap))
            lines.append("  %s" % ("多渲染了" if gap < 0 else "少渲染了") + "，"
                         "常见原因是内容超出一页被截断，或字库里缺字被跳过。")
    ds = [r[1] for r in rows if not r[4]] or [0.0]
    ss = [r[2] for r in rows if not r[4]] or [0.0]
    # 判据按字类分别报 —— 报告里只印一个"中位"，用户没法判断
    # "我这个字为什么被判糊"。分开印才知道它是跟**同类字**比的。
    if ref:
        for g in ("cjk", "latin", "punct"):
            if g not in ref:
                continue
            d_g, s_g = ref[g]
            n_g = len(grp.get(g, []))
            lines.append("  %s 组基准（%d 字）：dens 中位 %.3f → 判据 %.3f ／ "
                         "sharp 中位 %.3f → 判据 %.3f"
                         % ({"cjk": "汉字", "latin": "字母数字", "punct": "符号"}[g],
                            n_g, d_g, 1.75 * d_g, s_g, 0.55 * s_g))
        lines.append("  注：判据按字类各自取基准。拉丁字母渲染字号为 0.95 倍"
                     "（LATIN），dens 天然偏高、sharp 天然偏低，")
        lines.append("      与汉字同用一条基准会把整块字母误报成「糊」。")
    else:
        lines.append("墨迹密度（字形自身 alpha）：中位 %.3f，P90 %.3f，最大 %.3f；判据为相对值 1.75×中位"
                     % (float(np.median(ds)), float(np.percentile(ds, 90)), float(max(ds))))
        lines.append("边缘锐度：中位 %.3f，P10 %.3f，最低 %.3f；判据为相对值 0.55×中位"
                     % (float(np.median(ss)), float(np.percentile(ss, 10)), float(min(ss))))
    # 被纸面边界裁掉过半的字形：这类字**没被算成"缺"**（它确实落墨了），
    # 但用户在一整页成品里根本看不出哪几笔短了 —— 必须程序点名。
    # （原实现把比例写进了 rec["clipped"] 之后再没人读，BLEND_CLIP_WARN 的
    # 注释写着"必须程序说"，实际一个字都没说。）
    part = [r for r in recs if r.get("clipped")]
    if part:
        worst = max(float(r["clipped"]) for r in part)
        lines.append("被纸面边界裁掉过半的字形：%d 个（最严重裁掉 %.0f%%），涉及：%s"
                     % (len(part), worst * 100,
                        "".join(sorted({r["ch"] for r in part}))[:40]))
        lines.append("  常见原因是纸面可用范围没量准（见上方纸面告警），或内容排到了画布边缘。")
    # 墨量过低被跳过的字形：不在纸上，必须点名（否则用户只知道"少了几个字"）
    skip = [r for r in recs if r.get("skipped")]
    if skip:
        bywhy = {}
        for r in skip:
            bywhy.setdefault(r["skipped"], []).append(r["ch"])
        for why, cs in bywhy.items():
            lines.append("有 %d 个字因为「%s」被跳过、没有贴上去：%s"
                         % (len(cs), why, "".join(sorted(set(cs)))[:40]))
        lines.append("  这类字形贴上去会被放大成一坨黑块，所以宁可空着。"
                     "用 handglyph shape --lib <库> 可以列出它们，处置是重写后并入。")
    for ch, kinds in sorted(by_ch.items(), key=lambda kv: -sum(kv[1].values())):
        lines.append("  %s：%s  ×%d" % (ch, "/".join("%s%d" % (k, v) for k, v in kinds.items()), sum(kinds.values())))
    weak = [ch for ch in by_ch if ch in WEAK]
    if weak:
        lines.append("源字形本身质量偏低，建议重写后并入字库：%s" % "".join(weak))
        lines.append("    handglyph form \"%s\" -o 补字单.png" % "".join(weak))
    if not bad and not part:
        lines.append("结论：成品合格。")
    elif not bad:
        lines.append("结论：逐字质检合格，但有 %d 个字形被边界裁掉过半（见上）。" % len(part))
    else:
        lines.append("结论：%d 处不合格，见上表。" % len(bad))
    return "\n".join(lines) + "\n", len(bad)


# 标点优先级：库里有这个标点的字形就用它（那是用户真手写的），
# 没有才由程序画一个点/圈兜底。DOT_CHARS 里原本是一律画点，
# 结果用户把逗号、句号写进字库也永远用不上，画出来的点长得都一样。
# 程序兜底时按字符形状选样式：句号画空心小圈，其余画实心点。
DOT_RING = "。"
DOT_FALLBACK = "，、·."

# 字形"有效墨量"下限：贴到版面上时，一个字形至少要有这么多墨点像素，
# 否则单字还原度太低，放大后就是一坨实心黑。
#
# 为什么必须管这件事：`render` 贴字时的缩放倍率是
# `tw = 字形宽 × size / 字形高`，也就是**把字形拉到目标字号那么高**。
# 一个 18×11、只有 47 个墨点像素的碎片（实测是 `·` 的入库字形），
# 被拉到 40px 高、横向 65px 后，47 个墨点铺满 40×65 的格子 → 实心黑块。
# 同一个碎片如果目标字号小一点、或它本身宽高比正常，症状会轻得多 ——
# 所以这不是"某个标点坏了"，而是"**过小的源字形被过度放大会变成墨块**"，
# 换任何一个字都成立。判据因此写成通用的墨量检查，而不是给 `·` 打补丁。
GLYPH_MIN_INK = 90

# ---------------------------------------------------------------- 墨量/笔画归一
#
# 解决的问题（2026-09-17 用户指出"最后的效果粗细看起来要大概一致"）：
#
# 字库里的字形**大小差得很远** —— 本库字高从 11px 到 74px（6.7 倍）。
# 渲染的做法是把字形**拉到目标字号那么高**（`tw = 字形宽 × size / 字形高`），
# 等比缩放意味着"源字形在自己那个尺寸下有多粗"会被原样带到纸上。实测：
#     字形 `F`  高 64px、半径 3.00 → 缩 0.63 倍 → 纸上半径 1.88
#     字形 `端` 高 48px、半径 1.00 → 缩 0.83 倍 → 纸上半径 0.83
# **同一个字，光这一项就差 5.5 倍**（归一半径 0.66 ~ 3.64）。
# 用户看到的"粗细不统一"，主因在这里 —— 不是抖动，也不是浓淡。
#
# 一层修正（原先计划两层，宽度那层整块删了）：
#   ink_normalize —— 改**墨色浓淡**（alpha 量，靠乘系数）
#
# ⛔ 删掉的另一层是 stroke_normalize（改笔画宽度）。它前后写过三版，三版全废，
#    2026-09-18 整块移除。删它**不是**因为它没写完，而是因为**这件事在数据上做不到**：
#
#    · **细笔画（半径 <1.5px，本库的大多数）根本不可修**：1~1.5px 的条子做一次
#      `binary_erosion` 就整条消失。实测 1px 竖条要"变细"，输出墨量 40→40 点，
#      一点没动 —— 不是代码懒得动，是**退无可退**。加粗方向能走一格，但"3px 条
#      变 2px 条"这种跳变，观感上不是"更齐"，是"这根笔画没了"。
#    · **实心笔画只能整格跳**：`dilation` 一次就是一整圈像素。实测条宽 6/10/16
#      加粗时墨量 384→448、640→704、1024→1088，涨的全是"宽 1px"的量。
#      想要"从 1.25 拉到 1.5"（半格）？做不到，离散数据上没有半格。
#    · **量尺本身在细笔画上没有分辨率**：`stroke_radius` 在 2px 条和 3px 条上
#      量出来**都是 1.0000**。于是"半径纹丝不动"里有很大一部分**是量尺没变化，
#      不是墨没变化** —— 条宽 4 加粗时墨量 160→200（真变了 25%），半径读数照样不动。
#      这也意味着当初把"半径朝目标移动"写成断言，本身就不成立。
#
#    为什么前两版没被测出来：断言用的字形是**人为造的大竖条**（源半径 4.5），
#    而真实字库里的笔画半径多在 1~2px。**测的东西不像真东西，测过也不能算数。**
#
#    实测数据留在 gen_out/v07_in/_sn_check.txt 与 _sn_diag*.txt，别重开这条线。
#    粗细不齐这件事，靠**写的时候一致**（见文档"从源头"那一节），不靠程序事后修。
INK_NORM_GAIN = 0.5
INK_NORM_CAP = 0.32


# 正文字号（像素）。**唯一真源** —— 原来 `h = 40` 散在三处
# （render / render_flow / 量行高），改一处漏两处就会让
# "笔画宽度标尺折算到哪个字高"和"实际贴的字号"对不上。
# 标题 `*` 用 34，见各处 `h, ln = 34, ...`。
BODY_H = 40


def _too_thin(a):
    """字形墨量是否低到不该直接贴版（是则返回墨点像素数，否则 None）。

    判据是**绝对值**（墨点像素数），不是比例：比例会漏掉"小图小墨"这种情况
    —— 一个 18×11 的碎片墨占比 0.237，看着不低，但绝对只有 47 个像素，
    拉到 40px 高就必糊。反过来一个 46×55 的 `A` 有 569 个墨点，随便放大都撑得住。
    """
    if a is None or a.size == 0:
        return None
    n = int((a > 0.5).sum())
    return n if n < GLYPH_MIN_INK else None


def stroke_radius(a, thr=0.5):
    """字形的**笔画半径**（像素，在字形自身的尺度上）。

    ⛔ 这个度量换过三次，前两次都被实测否掉，别再退回去：

    ① 用**墨量总量**当粗细 → 错。墨量 = 厚度 × 长度，`日` 和 `口`
       笔画一样粗但墨量差很多。
    ② 用**逐行落墨游程的中位数** → 也错。它数的是"有多少段横着的连续墨"，
       斜画多的字（如 `A`）被切得碎，得出偏小的值。
    ③ 现在这个：**距离变换**。对每个墨点算它到最近背景的欧氏距离，
       取所有墨点的**中位数** —— 中位数落在笔画中轴线上，直接就是半径。
       这是"粗细"的定义式，与有多少笔、笔多长无关。

    为什么取中位数而非均值：均值会被交叉口（笔画交汇处离背景最远）和
    抗锯齿边缘一起拉偏；中位数落在笔画主体上，稳。

    ⚠ **它现在只被 `batch` 用来做批间一致性报告，不再喂给任何"修正"环节**
    （那个环节 2026-09-18 删了，理由见常量区 INK_NORM_* 上方的长注释）。
    留在文件里的原因：它是那套报告里"人写得多稳"的唯一量化出口。
    但要知道它的**分辨率在细笔画上很差** —— 2px 条与 3px 条都量出 1.0000，
    所以别拿它做小于 1px 的判断（这正是它当初被误信、写出假断言的原因）。
    """
    m = np.asarray(a) > thr
    if not m.any():
        return 0.0
    d = ndimage.distance_transform_edt(m)
    v = d[m]
    return float(np.median(v)) if v.size else 0.0


def folded_stroke_radius(a, ref_h=BODY_H):
    """字形"折算到参考字高 ref_h"的笔画半径 —— **全文件唯一的折算口径**。

    为什么必须折算：渲染是把字形拉到目标字号那么高，真正决定"贴到纸上多粗"的
    是 `半径 × 目标字高 / 字形高`。直接比原尺寸半径，比的是苹果和橘子。

    ⛔ 这个口径在文件里曾经有两份，而且不一致：
      · render 侧（page_stroke_target）写成 `r * (ref_h / 40)` —— 分母是常量 40，
        而调用方传的 ref_h 就是 40，于是系数恒为 1，等于**没折算**；
      · batch 侧（_stroke_of_alpha）写成 `r * (40 / 字形高)` —— 这个是**对的**。
    现在两处都走这一个函数，再漂就没地方漂了。
    （`page_stroke_target` 那份已随 width 归一一起删除，只剩 batch 这一个调用方。）
    """
    arr = np.asarray(a, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., -1] / 255.0 if arr.dtype == np.uint8 else arr[..., -1]
    if arr.size == 0 or arr.shape[0] < 8:
        return None
    r = stroke_radius(arr)
    if r <= 1e-6:
        return None
    return r * float(ref_h) / float(arr.shape[0])



# ---------------------------------------------------------------- 字库分批与批间比粗细
#
# 用户 2026-09-17 的需求（原话）：
#   「字库有可能新补充一批，跟原来的字库它的轻重就不一样，但是新补充的字库里面的字
#     之间一般它的轻重是一样的。我们存储字库的时候，解决方式可以设置一个分批，
#     就知道它到底是什么时候存储的。然后比对一下跟之前的库里粗细的差异，
#     如果真在一个不可接受的范围内，就让他重写一下之前的，跟这些新的一起重写。」
#
# 三个设计要点：
#   ① **同批内齐、批间可能不齐** —— 一次写的字（同一支笔、同一天、同一种状态）
#      内部轻重是一致的；换一次写就可能整体偏粗或偏细。
#      所以比对口径必须是**批与批之间比中位数**，绝不能全库取一个中位然后逐字比
#      —— 那样一整批的偏移会被摊成"每个字都异常"，看不出真正的问题。
#   ② **粗细要折算到统一字高再比** —— 渲染是把字形拉到目标字号那么高，
#      真正决定"贴到纸上多粗"的是 `半径 × 目标字高 / 字形高`。
#      直接比原尺寸半径会把"字形大小差异"误当成"笔画粗细差异"（第 ② 版翻车的原因）。
#   ③ **只提示、不自动改** —— 阈值超了要**让人重写**，因为程序改是把笔画缩放，
#      缩放会连带改字号、改清晰度（字母一放大就糊，见分辨率闸门）。
#      重写才是根治，程序只负责把"差多少"算清楚说出来。
BATCH_STROKE_REF_H = 40.0     # 比粗细时的参考字高（与 BODY_H 同口径）
# 批间粗细差容忍带：两批的中位半径相差超过这个比例就提醒重写。
# 标定：同一批内各字的半径差实测在 ±10% 以内；换一批写常差到 15~30%。
# 取 20% 是"明显不是一支笔写的"这一档；低于它多半只是正常落笔起伏。
BATCH_STROKE_TOL = 0.20
# 单批内至少要这么多字才值得量中位（太少的中位不稳）
BATCH_MIN_CHARS = 4


def _stroke_of_alpha(a, ref_h=None):
    """一个字形"折算到参考字高"的笔画半径（走唯一的折算口径）。"""
    return folded_stroke_radius(a, ref_h or BATCH_STROKE_REF_H)


def _batch_chars(man, batch):
    """一个批次包含哪些相对路径（从 instances 里按 batch 字段反查）。"""
    bid = batch.get("id")
    out = []
    for rel, info in (man.get("instances") or {}).items():
        if isinstance(info, dict) and info.get("batch") == bid:
            out.append(rel)
    return out


def _batch_stroke_median(lib_dir, rels):
    """量一个批次的**中位笔画半径**（折算到统一字高）。

    rels 可以是字符集合（旧调用）或相对路径集合（新调用），两种都认：
    给的是字符就去 chars 里取其全部实例，给的是路径就直接读那个文件。
    """
    vals = []
    if not rels:
        return None
    man = None
    mp = os.path.join(lib_dir, "manifest.json")
    try:
        with open(mp, encoding="utf-8-sig") as f:
            man = json.load(f)
    except Exception:
        man = None
    for item in rels:
        if man is not None and item in (man.get("chars") or {}):
            paths = list(man["chars"][item])
        else:
            paths = [item]
        for rel in paths:
            p = os.path.join(lib_dir, rel.replace("/", os.sep))
            if not os.path.isfile(p):
                continue
            try:
                a = load_alpha(p)
            except Exception:
                continue
            v = _stroke_of_alpha(a)
            if v is not None:
                vals.append(v)
    if not vals:
        return None
    return round(float(np.median(vals)), 4)


def compare_batches(man):
    """比各批次之间的粗细。返回 [(批次id, 与基准的偏差比例, 中位, 字符集)]。

    基准取**所有批次中位数的中位数**（不是全库逐字中位）——
    这样"某一批整体偏粗"会表现为该批偏离基准，而不会被自己稀释掉。
    """
    bs = [b for b in (man.get("batches") or []) if b.get("stroke_med")]
    if len(bs) < 2:
        return []
    ref = float(np.median([b["stroke_med"] for b in bs]))
    out = []
    for b in bs:
        d = (b["stroke_med"] - ref) / max(ref, 1e-9)
        out.append((b["id"], d, b["stroke_med"], b.get("chars") or []))
    return out


def batch_report(lib_path, tol=None):
    """给用户看的"批间粗细比对"报告。返回 (文本行列表, 是否超限)。"""
    tol = BATCH_STROKE_TOL if tol is None else tol
    man = load_manifest(lib_path)
    bs = [b for b in (man.get("batches") or []) if b.get("stroke_med")]
    lines = []
    if not bs:
        lines.append("字库还没有分批信息（老库升级上来的），无法比对批间粗细。")
        lines.append("下次用 --merge 补字时会自动建立批次。")
        return lines, False
    if len(bs) == 1:
        b = bs[0]
        lines.append("字库目前只有 1 个批次（%s，%s，%d 字），无批间差异可比。"
                     % (b["id"], b.get("date", "?"), len(b.get("chars") or [])))
        lines.append("  本批中位笔画半径（折算到 %dpx 字高）：%.2f"
                     % (BATCH_STROKE_REF_H, b["stroke_med"]))
        return lines, False
    ref = float(np.median([b["stroke_med"] for b in bs]))
    lines.append("批间粗细比对（参考字高 %dpx，基准 %.2f）" % (BATCH_STROKE_REF_H, ref))
    lines.append("  %-6s %-12s %6s %10s %10s  %s"
                 % ("批次", "日期", "字数", "中位半径", "与基准差", "判定"))
    bad = []
    for b in bs:
        d = (b["stroke_med"] - ref) / max(ref, 1e-9)
        flag = "正常"
        if abs(d) > tol:
            flag = "⚠ 偏离过大"
            bad.append((b["id"], d, b.get("chars") or [], b.get("date", "?")))
        lines.append("  %-6s %-12s %6d %10.2f %9.1f%%  %s"
                     % (b["id"], b.get("date", "?"), len(b.get("chars") or []),
                        b["stroke_med"], d * 100, flag))
    if bad:
        lines.append("")
        lines.append("以下批次的落笔轻重跟其它批明显不是一路（超过 %.0f%%），"
                     "建议**重写**：" % (tol * 100))
        for bid, d, chs, date in bad:
            lines.append("  %s（%s，%d 字，%+.1f%%）" % (bid, date, len(chs), d * 100))
            lines.append("    涉及字符：%s" % "".join(chs))
        lines.append("")
        lines.append("重写方法：把这些字跟新要写的字**放在同一张补字单上一起写** ——")
        lines.append("  handglyph form \"<上面的字符>加其它要写的字\" --copies 3 -o 补字单.png")
        lines.append("  handglyph build 补字单照片.jpg --lib %s --form-sheet --merge --replace"
                     % lib_path)
        lines.append("  （--replace 会把旧实例换成新写的；一起写能保证它们粗细一致）")
        return lines, True
    lines.append("")
    lines.append("各批次粗细都在容忍带内（±%.0f%%），无需重写。" % (tol * 100))
    return lines, False


def ink_normalize(a, target=None, gain=None, cap=None):
    """把字形的**墨色浓淡**朝 target 调一点。

    ⚠ 只改颜色深浅，**不改笔画宽度**。原计划里"改宽度"由 `stroke_normalize`
    负责，但那件事在离散数据上做不到（见常量区 INK_NORM_* 上方的长注释，
    2026-09-18 整块删除）。**所以现在粗细一律不动，只修浓淡。**

    淡浓和粗细分开看，是因为它们本来是两件不同的事：
      · 粗细 = 笔画有多宽（几何量）—— **本程序不再碰**
      · 浓淡 = 笔画颜色有多深（alpha 量，改它只乘系数）
    """
    if target is None or a is None or a.size == 0:
        return a
    gain = INK_NORM_GAIN if gain is None else gain
    cap = INK_NORM_CAP if cap is None else cap
    cur = float(a.sum())
    if cur <= 1e-6:
        return a
    k = 1.0 + (float(target) / cur - 1.0) * gain
    k = float(np.clip(k, 1.0 - cap, 1.0 + cap))
    return a if abs(k - 1.0) < 1e-3 else np.clip(a * k, 0, 1)


def page_ink_target(lib, chars):
    """算一篇文字的**目标墨量**（本页用到的全部字形实例的中位数）。

    ⚠ 口径是**墨量（alpha 总量）**，对应"浓淡"，不是"粗细"。
    """
    vs = []
    for ch in chars:
        for a in (lib.get(ch) or []):
            s = float(np.asarray(a, dtype=np.float32).sum())
            if s > 1e-6:
                vs.append(s)
    if not vs:
        return None
    return float(np.median(vs))



def _draw_dot(arr, cx, cy, r, ring=False, rng=None):
    """程序画标点：ring=True 画空心小圈（句号），否则画实心点。"""
    n = int(r * 2) + 6
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.sqrt((yy - n / 2.0) ** 2 + (xx - n / 2.0) ** 2)
    if ring:
        # 空心圈：外圈 r、内圈 r*0.52 之间的环
        dot = np.clip((r - d) / 1.4, 0, 1) * np.clip((d - r * 0.52) / 1.4, 0, 1)
    else:
        dot = np.clip((r - d) / 1.4, 0, 1)
    jx = rng.uniform(-1.5, 1.5) if rng is not None else 0.0
    jy = rng.uniform(-1.5, 1.5) if rng is not None else 0.0
    blend(arr, dot, int(cx + jx), int(cy + jy))
    return arr


def _tokens(ln):
    """把一行文本切成 (字符, 是否下标, 是否上标) 序列。

    标点**不再**在这里被强制改写成"．" —— 那是渲染阶段该决定的事
    （库里有就用库里的字形）。这里只负责解析上下标语法。
    """
    toks, over, sub, i = [], False, False, 0
    while i < len(ln):
        c = ln[i]
        if c == "~":
            over = True
            i += 1
            continue
        if c == "^":
            sub = True
            i += 1
            continue
        toks.append((c, sub, over))
        over = sub = False
        i += 1
    return toks


def _render_line(canvas, toks, lib, x0, base_y, h, avail, rng, picker, y, indent=0.0,
                 drift_amp=0.0, no_wrap=False, ink_target=None):
    """排一行（必要时折行）并贴字。

    返回 (canvas, 记录列表, 本行用掉的 token 数, 本行渲染的字数, 行末 x, 越界丢的字)。

    ⚠ 这是"一次布局 + 一次绘制"：原来量宽和正式贴字是两套代码，
    量宽那套用**每行都重置成同一个种子**的 srng，正式渲染用连续 rng，
    两边 vary() 的随机状态不同 → 量出来的宽度 ≠ 实际宽度（一行字算得下、
    实际却压到边上）；而且量宽路径还会对字形做**双重 vary**。
    现在只走这一条路径：先算坐标，边算边落墨，测量与实际必然一致。

    ⚠ 参数名是 canvas 不是 img：它现在自始至终是 float32 数组（画布真源），
    不再是一个 Pillow 图像。见 render() 开头的说明。
    """
    recs = []
    clipped = []            # 落点算出画布外、因此没画上去的字
    x = float(x0 + indent)
    n_tok, n_ch = 0, 0
    cand = 0
    while cand < len(toks):
        c, s, o = toks[cand]
        # --- 1. 决定这个位置画什么、多宽 ---
        if c == " ":
            _kind, w, _sz = token_metrics(c, h)
            x += w
            cand += 1
            n_tok += 1
            continue
        if c in DOT_CHARS and c not in lib:
            # 库里没有这个标点 → 程序画点兜底
            _kind, w, _sz = token_metrics(c, h)
            if not no_wrap and x + w > x0 + avail and x > x0 + indent + 1:
                break
            arr = canvas
            dot_cx, dot_cy, dot_r = x + h * 0.12, base_y - h * 0.34, h * 0.10
            _draw_dot(arr, dot_cx, dot_cy, dot_r, ring=(c in DOT_RING), rng=rng)
            # 程序画的点也要进 recs —— 它是成品上真实存在的墨迹。
            # 不记的话 audit 的"实渲染字数"会凭空少掉几个标点，
            # 对账就会把"程序画点"误报成"内容丢失"。
            recs.append({"ch": c, "x": int(dot_cx), "y": int(dot_cy - dot_r),
                         "w": int(dot_r * 2), "h": int(dot_r * 2),
                         "dens": float((dot_r * 2) ** 2) / max(1.0, float(h * h)),
                         "sharp": 0.0, "dark": 1.0, "prog": True})
            x += w
            cand += 1
            n_tok += 1
            n_ch += 1
            continue
        if c not in lib:
            _kind, w, _sz = token_metrics(c, h)
            x += w
            cand += 1
            n_tok += 1
            continue
        # 库里有：取字形（同字多实例轮转）。宽度口径走 token_metrics，
        # 与 measure_line_span 同一份实现 —— 两边各写一套必然漂移。
        # 顺带把"源字形身份"一起取出来，透传给 vary → upright 的倾角缓存。
        gid, src = picker.take(c, x, y)
        a = vary(src, rng, gid=gid)
        # 归一必须落在 vary 之后（vary 自己也会改宽度和墨量），
        # 且要放在 _too_thin 之前 —— 那个检查要看的是**归一之后真实贴版**的墨量。
        #
        # 这里**只剩墨色浓淡归一**（改 alpha 深浅，不改几何）。
        # ⛔ 笔画宽度归一（stroke_normalize）已于 2026-09-18 整块移除，别再往回加：
        #    实测它在本库上是"要么不动、要么把 1px 笔画整条削没"。详见原来那段
        #    常量注释的位置（现在写着删除理由）。
        a = ink_normalize(a, ink_target)
        thin = _too_thin(a)
        if thin is not None and c in DOT_FALLBACK:
            w = h * 0.40
            if not no_wrap and x + w > x0 + avail and x > x0 + indent + 1:
                break
            dot_cx, dot_cy, dot_r = x + h * 0.12, base_y - h * 0.34, h * 0.10
            _draw_dot(canvas, dot_cx, dot_cy, dot_r, ring=(c in DOT_RING), rng=rng)
            recs.append({"ch": c, "x": int(dot_cx), "y": int(dot_cy - dot_r),
                         "w": int(dot_r * 2), "h": int(dot_r * 2),
                         "dens": float((dot_r * 2) ** 2) / max(1.0, float(h * h)),
                         "sharp": 0.0, "dark": 1.0, "prog": True})
            x += w
            cand += 1
            n_tok += 1
            n_ch += 1
            continue
        if thin is not None:
            # 墨量过低的**非标点**字形：不硬贴。
            # 硬贴会被拉伸到 h 那么高，笔画不足 1px 的部分会被插值成一坨黑块 ——
            # 正是当初做"墨量闸门"要避免的事。这里跳过，并记一条进 recs：
            # audit() 会把它标成"跳过"、在报告里点名（原实现只处理了标点兜底，
            # 文档却写着"或跳过并在审计报告里说明" —— 那条路径实际不存在）。
            recs.append({"ch": c, "x": int(x), "y": int(base_y - h), "w": 0,
                         "h": int(h), "dens": 0.0, "sharp": 0.0, "dark": 1.0,
                         "skipped": "墨量过低"})
            x += h * 0.60               # 与"缺字"占位一致，别让后面的字挤上来
            cand += 1
            n_tok += 1
            continue
        # 宽度与字号都走**共用口径** token_metrics —— 与 measure_line_span 同一份
        # 实现（原来量行高用固定倍率表、贴字用实测宽度，两边差 10~22%）。
        _kind, tw_f, size_eff = token_metrics(
            c, h, a.shape[1] / float(max(1, a.shape[0])), s)
        tw = max(2, int(tw_f))
        gap = h * 0.22 * float(rng.uniform(0.85, 1.2))
        # --- 2. 放不下就折行（不再整行等比缩小 —— 一行字多了整行变小，观感很假）---
        if not no_wrap and x + tw + gap > x0 + avail and x > x0 + indent + 1:
            break
        # --- 3. 落墨 ---
        # 画布重构：arr / canvas 是**唯一真源**，全程只在 float32 数组上落墨，
        # 到 render 末尾才物化一次 Pillow 图。原来每个字都做
        # "整页 float32 转换 → 落墨 → 回写 uint8"，一次转换搬 26MB
        # （1240×1754×3×4B），每页几百字就是几百次 → 实测这是渲染最大的一笔开销。
        arr = canvas
        # 基线漂移：整行沿一条低频正弦走，模拟手沿纸面缓慢起伏（真实书写不是每字独立跳）
        drift = drift_amp * np.sin((x - x0) / max(1.0, avail) * 2.4 + 0.7) if drift_amp else 0.0
        yoff = int(base_y - size_eff + (h * 0.34 if s else 0.0) + drift + rng.uniform(-0.9, 0.9))
        t = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (tw, max(2, int(size_eff))), Image.BILINEAR)).astype(np.float32) / 255.0
        pk = float(t.max())
        if pk > 0.03:
            t = np.clip(t / (0.45 * pk), 0, 1)
        t = np.clip((t - 0.32) / 0.42, 0, 1)
        landed, cut = blend(arr, t, int(x), yoff)
        ink_g = t > 0.5
        dens = float(ink_g.mean()) if ink_g.size else 0.0
        rec = {"ch": c, "x": int(x), "y": yoff, "w": tw, "h": int(size_eff),
               "dens": dens, "sharp": edge_ratio(ink_g),
               "dark": float(t[ink_g].mean()) if ink_g.any() else 1.0}
        # 只有真的落在画布上才记进 recs —— 否则字数对账会把"落点算到画布外
        # 因此没画上去"的字算成已渲染，报"相符"，用户在成品上找不到那几个字。
        if landed:
            if cut > BLEND_CLIP_WARN:
                rec["clipped"] = round(float(cut), 3)
            recs.append(rec)
        else:
            clipped.append(c)
        if o:
            bar = np.ones((max(2, int(h * 0.075)),
                           max(4, int(tw * rng.uniform(0.85, 1.0)))), np.float32)
            blend(arr, bar, int(x + tw * 0.05), int(yoff - h * 0.12))
        x += tw + gap
        cand += 1
        n_tok += 1
        n_ch += 1
    return canvas, recs, n_tok, n_ch, x, clipped




def render(spec_path, lib_path, bg_path, out_path, seed=20260914, size=None):
    lib = load_library(lib_path)
    bg = open_rgb(bg_path)
    if size:
        bg = bg.resize(size, Image.LANCZOS)
    rows, bot, _ = paper_profile(bg)
    # 纸面轮廓没量出来（整幅都是纸、或有大片阴影）时，可用范围会退化成兜底值。
    # 这不是"正常运行"—— 排版宽度是错的，成品可能整行冲出画布。
    # 必须说出来，不能静默按兜底值排完还报"合格"。
    fallback_rows = all(lo < 0 for _yy, lo, _hi in rows) if rows else True
    if fallback_rows:
        print("警告：没能从背景图里量出纸面可用范围，排版按保守宽度 %d px 处理。"
              % (PAPER_FALLBACK[1] - PAPER_FALLBACK[0]), file=sys.stderr)
        print("  成品可能比预期更早折行，或右侧留白偏大。", file=sys.stderr)
        print("  换一张背景更干净/对比更明显的纸面图可避免这种情况。", file=sys.stderr)
    rng = np.random.default_rng(seed)
    picker = Picker(lib, rng=rng)

    raw = [ln.rstrip("\n") for ln in read_text(spec_path).splitlines() if ln.strip()]

    # 纯注释行（# 开头但不是指令）不参与渲染。extract 生成的骨架文件通篇都是
    # 这种提示行，用户忘删时提示语会被当成正文写进成品里。
    text_lines = [ln for ln in raw if not ln.startswith("#")]

    # 整篇共用一个墨量标尺（本页/本篇用到的字才算），让同一页里的字彼此齐。
    # 按整篇而不是按行：同一行内当然要齐，行与行之间更要齐 ——
    # 只按行算的话，某行恰好都是粗字、另一行都是细字，两行之间反而更不齐。
    page_chars = set()
    for ln in text_lines:
        for c, _s, _o in _tokens(ln[1:] if ln.startswith(("*", ">")) else ln):
            if c in lib:
                page_chars.add(c)
    ink_target = page_ink_target(lib, page_chars)

    missing = {}
    for ln in text_lines:
        body = ln[1:] if ln.startswith("*") else ln
        # 缩进指令 ">" 是语法标记，不是要写的字
        if body.startswith(">"):
            body = body[1:]
        for ch in body:
            if ch in SKIP_CHARS:
                continue
            if ch not in lib:
                missing[ch] = missing.get(ch, 0) + 1
    if missing:
        print("缺字：%s" % " ".join(sorted(missing)), file=sys.stderr)

    # ⚠ **画布真源只有一个**：canvas 是 float32 数组，全程直接在它上面落墨，
    # 到函数末尾才物化一次 Pillow 图。原来的写法是每贴一个字都
    # "np.asarray(img) 整页转 float32 → 落墨 → Image.fromarray 回写 uint8"，
    # 一页 1240×1754×3 的 float32 就是 26MB，一个字搬两趟 —— 这是本工具
    # 逐字渲染热路径上最大的一笔无谓开销（与"画什么"完全无关的纯搬运）。
    canvas = np.asarray(bg).astype(np.float32)
    recs = []
    clipped = []          # 落点算出画布外、因此没画上去的字（P1-B）
    y = 90
    rendered_lines = 0
    truncated = False
    n_expect = 0          # 应渲染的总字数（含程序画的点）；按实际画出的字累加
    n_prog = 0            # 其中由程序画的
    for ln in raw:
        if ln.startswith("#wave"):
            lo, hi = range_at(rows, int(y + 60))
            # 波形要直接画到 Pillow 上（都是画线原语），画完立刻并回 canvas。
            # 这类整页性操作一页只出现几次，单独物化一次是划算的。
            tmp = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
            dr = ImageDraw.Draw(tmp)
            waveform(dr, lo + 40, y + 10, min(hi - 40, lo + 760) - (lo + 40), 110, rng)
            canvas = np.asarray(tmp).astype(np.float32)
            y += 150
            continue
        if ln.startswith("#gap"):
            # 页内留白指令：`#gap 40`。缺了数字或数字非法时不能崩 ——
            # 这是人手写的排版指令，写错一个数字就 traceback 太粗暴了。
            arg = ln.split()[1] if len(ln.split()) > 1 else ""
            try:
                y += int(arg)
            except ValueError:
                print("提示：#gap 后面的数字看不懂（%r），本行按默认 40 处理。" % arg,
                      file=sys.stderr)
                y += 40
            continue
        if ln.startswith("#"):
            continue
        h = 40
        # 段落缩进指令：行首 ">" 表示缩进两字宽（写正文段首那种）
        lead = 0.0
        if ln.startswith(">"):
            lead = 40 * 2.0
            ln = ln[1:].lstrip()
        if ln.startswith("*"):
            h, ln = 34, ln[1:].strip()
        toks = _tokens(ln)
        lo, hi = range_at(rows, y + h // 2)
        avail = max(60, (hi - 20) - (lo + 20))
        if bottom_at(bot, int(lo + 30)) < y + int(h * 1.4) + 20:
            truncated = True
            break
        x_left = float(lo + 20)
        base_y = y + h
        rest = list(toks)
        first = True
        while rest:
            canvas, rr, used, nch, xend, cut_chars = _render_line(
                canvas, rest, lib, x_left, base_y, h, avail, rng, picker, y,
                indent=(lead if first else 0.0),
                drift_amp=BASE_DRIFT,
                no_wrap=NO_WRAP, ink_target=ink_target)
            recs.extend(rr)
            clipped.extend(cut_chars)
            if used <= 0:
                break
            # 对账口径：把"这趟排上版面的 token"折算成**应出现的字数**：
            #   库里有该字 / 该标点需要程序画点 → 计入应渲（前者落墨、后者也算画出来了）
            #   空格、上下标语法符、缺字 → 不计（它们本就不该在成品里占位置）
            # 然后与 recs 的实际落墨数比对，"差几字"就是渲染环节真的丢了几个。
            for c, _s, _o in rest[:used]:
                if c == " " or c in SUP_CHARS:
                    continue
                if c in SKIP_CHARS or c in lib:
                    n_expect += 1
                    if c in DOT_CHARS and c not in lib:
                        n_prog += 1
            rest = rest[used:]
            if not rest:
                break
            # 折行：进入下一行，重新取页面上可用的横向范围
            y += int(h * 1.8) + 10
            rendered_lines += 1
            lo, hi = range_at(rows, y + h // 2)
            avail = max(60, (hi - 20) - (lo + 20))
            if bottom_at(bot, int(lo + 30)) < y + int(h * 1.4) + 20:
                truncated = True
                break
            x_left = float(lo + 20)
            base_y = y + h
            first = False
        if truncated:
            break
        y += int(h * 1.8) + 10
        rendered_lines += 1
        if bottom_at(bot, int(lo + 30)) < y + 20:
            truncated = True
            break
    # 全部落墨结束，物化**一次**成 Pillow 图（画布重构的收尾）
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(0.15))
    img.save(out_path)
    # 越界丢的字必须从"应渲染"里扣掉，否则字数对账会把它们算成"已渲染"。
    # 这是 P1-B 的正面修法：blend 现在会说"我没落下"，这里据此修正口径，
    # 同时把它们单列出来告警 —— 用户在一页成品里找不到几个字，必须有人告诉他。
    n_lost = len(clipped)
    if n_lost:
        uniq = "".join(sorted(set(clipped)))
        print("警告：有 %d 个字算出来的落点在画布外，没能画上去：%s"
              % (n_lost, uniq[:40]), file=sys.stderr)
        print("  常见原因是内容太长、或纸面可用范围没量准（见上面的纸面告警）。",
              file=sys.stderr)
    report, nbad = audit(recs, expected=max(0, n_expect - n_lost), n_prog=n_prog)
    if truncated:
        miss_n = len(text_lines) - rendered_lines
        print("警告：内容超出一页，后面作者可能还有内容没有渲染"
              "（本页放下 %d 行 / 共 %d 行，另有 %d 行没渲染；共 %d 字）。"
              % (rendered_lines, len(text_lines), miss_n, len(recs)), file=sys.stderr)
        print("  这些内容不会出现在图上。请拆成多页分别渲染，或精简内容。", file=sys.stderr)
        report = ("【内容被截断】输入 %d 行文字，本页只渲染 %d 行（%d 字），"
                  "后面的内容没有出现在图上。请拆页或精简内容。\n"
                  % (len(text_lines), rendered_lines, len(recs))) + report
    if LEAN_TOL > 0 and lib:
        # 只扫**本页真的用到**的字。
        # 原来这里是 for c in lib —— 整个字库每个字都跑一遍 lean_report
        # （37 次旋转的粒度扫描），而用户看的这一页可能只用了 30 个字。
        # 扫描结果按整库报"倾角中位/最大"也就没有意义：那个中位数含几百个
        # 本页根本没写的字。改成页面涉字之后，报出来的数字才真的是"这一页的观感"。
        used_chars = {r["ch"] for r in recs} & set(lib)
        angs = []
        acted = 0
        for c in sorted(used_chars):
            g = lib[c][0]
            if g.max() <= 0.02 or g.shape[0] < 14:
                continue
            o, conf = lean_report(g)
            if conf >= LEAN_GAIN and abs(o) > LEAN_TOL:
                acted += 1
            angs.append(abs(o))
        if angs:
            rv = np.array(angs, dtype=np.float32)
            # 只报确定的两个数：扫描了多少字、多少字会被扶正。
            # 曾经写过"字库中 N/M 字存在长竖画"，但那个 N 其实是"参与扫描的字数"，
            # 与没有长竖画的字也被算进去了，措辞与事实不符。
            report += ("竖画（容差 %.1f°，上限 %.1f°，置信 %.0f%%，仅本页用到的字）："
                       "扫描 %d 字，%d 字需要扶正；倾角 中位 %.2f° / 最大 %.2f°\n"
                       % (LEAN_TOL, LEAN_CAP, LEAN_GAIN * 100,
                          len(angs), acted,
                          float(np.median(rv)), float(rv.max())))
        else:
            report += ("竖画（容差 %.1f°）：本页用到的字都没有可判定的长竖画，未做统计。\n"
                       % LEAN_TOL)
    rp = os.path.splitext(out_path)[0] + ".audit.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report, end="")
    print("质检报告：%s" % rp)
    # 返回 (成品路径, 是否被截断)：调用方（main）据此给非零退出码。
    # 原来只返回路径，于是"内容超出页面被截断"这件事在脚本里完全看不见 ——
    # 审计报告里写了，可退出码仍是 0，`render ... && 下一步` 会照常往下走。
    return out_path, truncated


# ---------------------------------------------------------------- 建库与补字单

def robust_scale(vals, window=1.7, iters=30):
    """主簇稳健尺度估计（对数空间均值漂移）：少数杂块拉不动基准。"""
    v = np.log(np.maximum(np.asarray(vals, dtype=np.float32), 1e-3))
    c = float(np.median(v))
    for _ in range(iters):
        w = np.abs(v - c) <= np.log(window)
        if w.sum() < 2:
            break
        nc = float(v[w].mean())
        if abs(nc - c) < 1e-4:
            c = nc
            break
        c = nc
    return float(np.exp(c))


def next_index(d):
    """取目录里已有 PNG 的最大编号 +1，作为新实例的文件名。

    不能用 len(manifest 列表)：prune 删掉 00.png 后列表只剩 2 项，
    下次 merge 会写出 02.png 覆盖幸存的旧实例，manifest 里还出现两条指向同一文件。
    编号只增不重用空洞 —— 宁可编号变大，也不能覆盖。
    """
    mx = -1
    try:
        for f in os.listdir(d):
            if f.lower().endswith(".png"):
                stem = os.path.splitext(f)[0]
                if stem.isdigit():
                    mx = max(mx, int(stem))
    except OSError:
        pass
    return mx + 1


def build(sample, out_dir, expect=None, pad=5, merge=False, debug=False, replace=False,
          form_sheet=False, cols=12, cell=CELL, rebuild=False):
    """把一张字版照片切成字形库。expect 给出字版上的字符顺序时自动命名。

    form_sheet=True 时改用"格子感知"切分（配合 make_form 生成的补字单），
    见 build_form_sheet 的说明。默认 False 走原来的自由切分。
    """
    if form_sheet and not (expect or "").strip():
        print("补字单模式必须给 --expect。")
        print("")
        print("  为什么不能省：补字单的字符名**只能**来自 --expect —— 它是格位顺序")
        print("  对应的字符表。不给的话程序只能编造名字（u0000、u0001…），")
        print("  而且不知道要读几行，只读第一行 12 格。")
        print("")
        print("  正确用法（先让程序告诉你要写哪些字）：")
        print("    handglyph coverage --lib library -t \"你要写的文字\"     # 看缺哪些字")
        print("    handglyph form \"缺的那些字\" -o 补字单.png              # 打印补字单")
        print("    handglyph build 拍的照片.jpg --form-sheet --expect \"缺的那些字\" -o library --merge")
        sys.exit(2)
    if form_sheet:
        return build_form_sheet(sample, out_dir, expect=expect, pad=pad,
                                merge=merge, debug=debug, replace=replace,
                                cols=cols, cell=cell, rebuild=rebuild)
    os.makedirs(out_dir, exist_ok=True)
    rep = {"raw": 0, "drop_small": 0, "drop_wide": 0, "drop_thin": 0, "drop_area": 0, "drop_fill": 0, "drop_atypical": 0, "drop_bad": 0, "split": 0, "kept": 0}
    drops = []
    im = open_rgb(sample)
    g = np.asarray(im.convert("L")).astype(np.float32)
    bg = ndimage.uniform_filter(g, size=81)
    ink = ((bg - g) > 25)
    ink = ndimage.binary_opening(ink, structure=np.ones((2, 2)))
    ink2 = ndimage.binary_dilation(ink, structure=np.ones((3, 1)))
    core = ndimage.binary_opening(ink2, structure=np.ones((1, 80)))
    if core.any():
        ink = ink & ~ndimage.binary_dilation(core, structure=np.ones((9, 1)))
    lab, n = ndimage.label(ink, structure=np.ones((3, 3)))
    objs, sizes = ndimage.find_objects(lab), ndimage.sum(ink, lab, range(1, n + 1))
    H, W = g.shape
    rep["raw"] = n
    comps = []
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        box = [x0, y0, x1, y1]
        if sizes[i] < 20:
            rep["drop_small"] += 1
            drops.append(("small", box))
            continue
        if (x1 - x0) > 0.4 * W:
            rep["drop_wide"] += 1
            drops.append(("wide", box))
            continue
        if (x1 - x0) > 2.5 * (y1 - y0) and (y1 - y0) < 18:
            rep["drop_thin"] += 1
            drops.append(("thin", box))
            continue
        comps.append(box)
    comps.sort(key=lambda b: ((b[1] + b[3]) / 2 // 40, b[0]))
    H, W = g.shape
    med = float(np.median([b[3] - b[1] for b in comps])) if comps else 20
    lines = []
    for b in comps:
        cy = (b[1] + b[3]) / 2
        for ln in lines:
            if abs(ln["cy"] - cy) < max(14, 0.5 * med):
                ln["b"].append(b)
                ln["cy"] = float(np.mean([(x[1] + x[3]) / 2 for x in ln["b"]]))
                break
        else:
            lines.append({"cy": cy, "b": [b]})
    glyphs = []
    for ln in sorted(lines, key=lambda l: l["cy"]):
        cur = None
        for b in sorted(ln["b"], key=lambda x: x[0]):
            if cur and b[0] - cur[2] <= 7 and min(cur[3], b[3]) - max(cur[1], b[1]) > 0.3 * (cur[3] - cur[1]):
                cur[2], cur[1], cur[3] = max(cur[2], b[2]), min(cur[1], b[1]), max(cur[3], b[3])
            else:
                if cur:
                    glyphs.append(cur)
                cur = list(b)
        if cur:
            glyphs.append(cur)
    if glyphs:
        areas = np.array([max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for b in glyphs], dtype=np.float32)
        heights = np.array([max(1.0, b[3] - b[1]) for b in glyphs], dtype=np.float32)
        a_ref = float(np.quantile(areas, 0.75))
        h_ref = float(np.quantile(heights, 0.75))
        keep = []
        for b in glyphs:
            a = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            h = max(1.0, b[3] - b[1])
            if not (0.30 * a_ref <= a <= 3.5 * a_ref) or not (0.60 * h_ref <= h <= 1.8 * h_ref):
                rep["drop_area"] += 1
                drops.append(("area", b))
                continue
            sub = ink[b[1]:b[3], b[0]:b[2]]
            if sub.size and float(sub.mean()) > 0.55:
                rep["drop_fill"] += 1
                drops.append(("fill", b))
                continue
            keep.append(b)
        glyphs = keep
    if expect:
        want = len([c for c in expect if c not in " \t"])
        if len(glyphs) == want and glyphs:
            ys = [(b[1] + b[3]) / 2.0 for b in glyphs]
            mh = float(np.median([b[3] - b[1] for b in glyphs]))
            if max(ys) - min(ys) < 1.5 * max(1.0, mh):
                glyphs.sort(key=lambda b: b[0])
        elif len(glyphs) > want:
            a_ref = robust_scale([max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for b in glyphs])
            h_ref = robust_scale([max(1.0, b[3] - b[1]) for b in glyphs])
            def dev(b):
                a = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                h = max(1.0, b[3] - b[1])
                return abs(np.log(a / a_ref)) + abs(np.log(h / h_ref))
            ranked = sorted(glyphs, key=dev)
            kept = set()
            for b in ranked[:want]:
                kept.add((b[0], b[1], b[2], b[3]))
            rest = [b for b in glyphs if (b[0], b[1], b[2], b[3]) not in kept]
            for b in rest:
                rep["drop_atypical"] += 1
                drops.append(("atypical", b))
            glyphs = [b for b in glyphs if (b[0], b[1], b[2], b[3]) in kept]
    rep["kept"] = len(glyphs)
    if debug:
        dbg = im.copy()
        dr2 = ImageDraw.Draw(dbg)
        for reason, b in drops:
            dr2.rectangle([b[0], b[1], b[2], b[3]], outline=(220, 60, 60), width=2)
        for b in glyphs:
            dr2.rectangle([b[0], b[1], b[2], b[3]], outline=(30, 170, 60), width=3)
        dbg.save(os.path.join(out_dir, "build_debug.png"))
        print("切分诊断：连通域 %d → 保留 %d" % (rep["raw"], rep["kept"]))
        print("  丢弃：太小 %d / 整行宽 %d / 细长 %d / 面积或字号离群 %d / 实心墨块 %d / 非典型多余块 %d / 不可信 %d"
              % (rep["drop_small"], rep["drop_wide"], rep["drop_thin"], rep["drop_area"], rep["drop_fill"], rep["drop_atypical"], rep["drop_bad"]))
        if rep["split"]:
            print("  切掉邻字墨块 %d 处" % rep["split"])
        print("  红框=丢弃，绿框=保留，见 build_debug.png")
    if expect:
        want = len([c for c in expect if c not in " \t"])
        if want != len(glyphs):
            print("警告：字版上的字符数 %d 与切出的字形数 %d 不一致，命名可能错位，请先看 atlas.png。" % (want, len(glyphs)))
    els = [c for c in (expect or "") if c not in " \t"]
    keep_boxes, keep_chars = [], []
    for k, (x0, y0, x1, y1) in enumerate(glyphs):
        ch = els[k] if k < len(els) else "u%04d" % k
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        crop = im.crop((cx0, cy0, min(W, x1 + pad), min(H, y1 + pad)))
        # alpha 用"局部背景差分"而非灰度反转，原因见 _store_glyphs 里的说明
        gsub = np.asarray(crop.convert("L")).astype(np.float32)
        bsub = bg[cy0:cy0 + gsub.shape[0], cx0:cx0 + gsub.shape[1]]
        if bsub.shape != gsub.shape:
            bsub = np.full(gsub.shape, 255.0, np.float32)
        alpha = np.clip((bsub - gsub) / 255.0, 0, 1)
        alpha = np.where(alpha < 0.16, 0, alpha)
        m = alpha > 0.5
        if m.sum() > 8 and crop.width > 12:
            lb2, n2 = ndimage.label(m, structure=np.ones((3, 3)))
            for i2, sl2 in enumerate(ndimage.find_objects(lb2), start=1):
                if sl2 is None:
                    continue
                hh2 = sl2[0].stop - sl2[0].start
                ww2 = sl2[1].stop - sl2[1].start
                if ww2 >= 0.80 * crop.width and hh2 <= 4:
                    alpha[sl2][lb2[sl2] == i2] = 0.0
        ink2 = alpha > 0.30
        if ink2.sum() > 8:
            ys2, xs2 = np.where(ink2)
            alpha = alpha[max(0, ys2.min() - 1):ys2.max() + 2, max(0, xs2.min() - 1):xs2.max() + 2]
        alpha, sp = split_bleed(alpha)
        rep["split"] += sp
        alpha = normalize_ink(alpha)
        # 用"裁好的字形"预判可信度：这一步只筛掉残片，真正的落库由 _store_glyphs
        # 重算一遍同样的 alpha（多算一次换来的是一份落库实现，避免两处逻辑漂移）
        bad_reason = plaus(alpha, ch=ch)
        if bad_reason:
            rep["drop_bad"] += 1
            drops.append(("bad", [x0, y0, x1, y1]))
            if bad_reason != "残片":
                # 非残片的拒收少见但不该静默 —— 用户看到"少了一个字"却不知为什么
                rep.setdefault("bad_detail", []).append((ch, bad_reason))
            continue
        keep_boxes.append([x0, y0, x1, y1])
        keep_chars.append(ch)

    # 落库统一走 _store_glyphs —— 原来 build 自己还有一份几乎一样的落库代码，
    # 两份并存的结果是：补字单路径加了实例档案，自由字版路径却漏了。
    # 合并成一处，两边行为必然一致。
    res = _store_glyphs(im, g, bg, keep_boxes, keep_chars, out_dir, expect=expect,
                        pad=pad, merge=merge, replace=replace, source=sample,
                        rebuild=rebuild)
    if res.get("refused"):
        # 必须把 refused 原样带出去：调用方（main）靠它决定退出码。
        # 之前这里重新拼了一个字典却漏掉 refused，结果是**拒绝信息印出来了，
        # 退出码却是 0** —— 脚本里 `handglyph build ... && 下一步` 会照常往下走，
        # 把"没入库"当成"入库成功"。这正是 P0-4 想杜绝的静默失败，只是换了个形态。
        return {"glyphs": 0, "chars": res.get("chars", 0), "atlas": None,
                "weak": "", "refused": True,
                "reason": res.get("reason", "字库已存在且未指定 --merge / --rebuild")}
    man = read_json(os.path.join(out_dir, "manifest.json"))
    if merge:
        old_soft, new_soft = [], []
        for ch, rels in man["chars"].items():
            for rel in rels:
                p = os.path.join(out_dir, rel)
                if not os.path.exists(p):
                    continue
                a = load_alpha(p)
                m = metrics_of(a)
                if m:
                    (new_soft if ch in set(els) else old_soft).append(m["soft"])
        if old_soft and new_soft:
            o, nw = float(np.median(old_soft)), float(np.median(new_soft))
            if nw > 1.6 * max(o, 1e-3):
                print("提示：本张字版的笔画边缘比现有字库软 %.0f%%（发虚 %.3f 对 %.3f）。"
                      % ((nw / max(o, 1e-3) - 1) * 100, nw, o))
                for ln in refill_notice(list(els), "这张字版拍得偏软，入库后会拉低字库清晰度。"):
                    print(ln)
    # 入库即时形状校验：本次新入的字若被压扁，当场报出来
    bad = []
    for ch in sorted(set(man["chars"])):
        for rel in man["chars"][ch]:
            p = os.path.join(out_dir, rel)
            if not os.path.exists(p):
                continue
            a = load_alpha(p)
            if squeeze_report(ch, a):
                bad.append(ch)
    if bad:
        bad = sorted(set(bad))
        print("提示：本张字版有 %d 个字被压扁（宽高比不对）：%s" % (len(bad), "".join(bad)))
        print("  原因：切分时抓取的区域偏窄，或这个字本身就写得太瘦长。")
        for ln in refill_notice(bad, "这些字需要重写并把字写进格子里（横向写足、不要写太瘦）。"):
            print(ln)
    return {"glyphs": res["glyphs"], "chars": res["chars"],
            "atlas": res["atlas"], "weak": res["weak"]}


# ---------------------------------------------------------------- 补字单：格子感知切分
# 解决的问题（原 build 在补字单上基本不可用）：
#
# 补字单是程序自己打印的 A4 表单：顶部有说明文字、每个格子里还印着一个浅灰参考字。
# 原 build 把它当"手写自由字版"处理，拿墨迹阈值 25 一卡 —— 打印说明文字（RGB 70，
# 跟钢笔墨色差不多深）和浅灰参考字（175 灰，对比度 80，远超阈值）全被算成笔迹。
# 两组实测：
#   · 空白补字单（一个字没写）：切出 528 个连通域，最终"保留 6 个"，全是说明文字碎片，
#     只因宽高比超限才侥幸没入库；
#   · 填了两个黑字的补字单：手写字反被判成"面积/字号离群"丢掉，入库 0 个字符 ——
#     因为说明文字一行十几个同样式的字构成"典型簇"多数派，格子里孤零零的大字倒成了离群值。
#
# 治本思路：补字单的几何是**已知的**（make_form 自己画的），根本不用猜。
#   1. 用横竖投影把格子阵列的行列边界检出来（照片有白边/轻微透视也能自适应）；
#   2. 格位顺序即字符顺序，命名不再依赖"切出个数刚好等于 expect 长度"，
#      命名错位、对 --expect 的顺序依赖这两个老毛病一并消失；
#   3. 每个格子里只认"最深的那团墨" —— 手写约 30 灰、参考字 175 灰，
#      用平均墨深一刀切开，参考字整批出局，不会混进字形库；
#   4. 格外的连通域（说明文字、页眉、页码、桌面纹理）从头到尾不参与。

# 手写墨迹与浅灰参考字的分界（灰度，越小越深）。
# 取值依据：钢笔/中性笔在均匀光下笔画芯部实测 25~60 灰；补字单参考字现已提到 200 灰。
# 门槛放 100 是给"笔尖偏细 + 光线偏亮"留余量，两边都不会误判。
#
# ⚠️ 这道门槛**依赖 form 生成端的灰度**，所以旧版打印件必须作废重打：
# v0.2 之前参考字是 70 灰（比门槛还深），会被当成手写收进库 —— 实测一张
# 一个字没写的旧表单会吐出两个完整的参考字（与 53x58、门 30x57），
# 输出还显示"有手写墨迹 2 格 / 入库 2 个字形"，没有任何异常信号。
# 这条无法用灰度阈值解决：旧参考字 70 灰与"下笔很轻的手写"70 灰像素同值。
# 可行判据只剩"位置一致性"（印刷参考字每格都落在同一相对位置、墨色方差极小），
# 需新标定，暂不做。
FORM_INK_DEPTH = 100

# 补字单的格子几何（与 make_form 共用）。x0/y0 是第一个格子的左上角，
# 之后按 cell 步进铺格。build --form-sheet 在检不到格线时就用这组常数兜底。
FORM_X0 = 90
FORM_Y0 = 560

# 补字单纸张尺寸（A4 @200dpi）。make_form 画纸用它，--cols / 字数的上限也由它推出来：
# 列数超了格子会被画到纸外（PIL 静默裁掉），build 端再把画外的格位当空位跳过 ——
# 表现是"入库的字少了几个"，全程没有任何报错。
FORM_W = 1654
FORM_H = 2339

# 格周期合理性带宽：实测周期必须落在 cell 的 [0.75, 1.35] 倍内，否则否决。
# 依据：自相关会把"说明文字行距"也当成周期（实测误检成 40/50/60），而真格宽
# 恒为 CELL —— 误检值全在带外。留这么宽是给照片透视/缩放吸收余量。
GRID_PERIOD_LO = 0.75
GRID_PERIOD_HI = 1.35

# 行列周期"应当接近"的倍数：格子是方的，差到 1.5 倍就丢小的一方。
# 用 >= 比较：实测 60 对 40 恰为 1.5 倍，严格大于会把两个错周期一起放过。
GRID_SQUARE_RATIO = 1.5

# 残片闸门：格内墨迹的高度跨度不足 cell 的这个比例，判为"被误检窗口切断的残片"，
# 不计入入库、并入空格。依据（实测）：正确窗口下正常字占 cell 的 0.52，
# 被 50px 矮窗切断的残片只有 0.19 —— 两倍多的余量，比按"窗口高一半"判稳得多
# （后者实测 0.48 对 0.50，只差 4%，全靠周期误检落点碰巧才生效）。
FORM_FRAG_SPAN = 0.35

# `form --kind symbols` 用的字符集：数字 + 常用标点 + 数学/箭头符号。
# 为什么单独列一份：这些字符在字库里通常缺且必须补，而用户手打一长串符号
# 很容易漏；一份常量能让补字单把整套符号一次铺完。
# 引号类（" ' ）故意不入表 —— 它们在 Python 字符串里要转义，写进 shell 命令
# 更容易出错，需要时用 form "..." 手打即可。
SYMBOLS = ("0123456789"
           "abcdefghijklmnopqrstuvwxyz"
           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "+-*/=<>"
           ".,:;!?()[]{}"
           "^~%&#@$"
           "±×÷≠≤≥→←↑↓∑∏√∞°"
           "、。，；：！？“”（）【】《》…—")

# 非法路径穿越防护：manifest 里的 rel 一律当作不可信输入。
# manifest 是纯文本、可手改、也会从别处拷来，里面写 "../../etc/passwd"
# 就能让 prune/clean 删到字库外面的文件 —— 读库前必须把住。
_MANIFEST_ENTRY_CACHE = {}


def detect_grid(g, cell=CELL):
    """定位补字单的格子阵列，返回 (列周期, 行周期)，检不到的一方给 None。

    注：返回的是**周期**，不是"列数"。原来签名上挂着 cols / expect_n 两个参数，
    函数体里一次都没读 —— 是早期设计（想直接数出格数）的残留。留在那里会让人
    以为 `--cols` 会影响检测结果，实际完全不影响：列数在调用侧按已知几何算。

    为什么不用"投影找贯穿线"：试过了，在**稀疏补字单上不成立**。补字单只有几个字时，
    竖格线总长才 200 多像素，而说明文字那一行是横贯整幅的长线 —— 投影里凸出来的是
    文字行和十字中线，不是格线（实测把步距认成了 27.5 px）。
    格线的"贯穿假设"只在格子铺满整页时才成立，而补字单恰恰经常是稀的。

    改用**局部自相关**：格子阵列是严格周期的，把灰度背景场去掉后的响应做
    自相关，周期处会出现尖峰。这对"线多长"不敏感，只依赖"间距相同"这一条，
    正好是格阵最本质的性质。检不出周期时返回 None，调用方用 make_form 的
    已知几何（FORM_X0/FORM_Y0/cell）兜底 —— 反正表单是自己画的。

    ⚠️ **周期合理性校验（必须有）**：自相关对"非格线的强周期"一样敏感 ——
    说明文字的行距就是现成的反例（实测 40~50 px 出峰）。而单边检出时
    原有的交叉校验（行列周期比对）根本管不到它，于是 108×50 的窗口会把字
    拦腰切断，**残片还能通过后续检查被静默命名入库**（看着像"入库 2 个字形，
    完全成功"）。所以这里必须用"表单是程序画的、格宽必然是 cell"这条先验把住：
    落在 cell 的 ±35% 之外一律否决，回退内置几何。
    """
    h, w = g.shape
    bg2 = ndimage.uniform_filter(g, size=41)
    resp = np.clip((bg2 - g) / 255.0, 0, 1)
    resp = ndimage.gaussian_filter(resp, sigma=1.0)

    def period_of(prof, lo, hi):
        """在 [lo, hi] 像素的周期范围内找主周期；找不到给 None。"""
        if prof.size < 4 * hi:
            return None
        x = prof - prof.mean()
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        if ac[0] <= 0:
            return None
        ac = ac / ac[0]
        lo, hi = max(4, int(lo)), min(int(hi), len(ac) - 1)
        if hi <= lo + 2:
            return None
        seg = ac[lo:hi]
        k = int(np.argmax(seg)) + lo
        # 峰值要明显高于邻近谷值，否则只是噪声
        near = ac[max(1, k - max(3, k // 4)):k].min() if k > 2 else 0.0
        if ac[k] < 0.25 or ac[k] < near + 0.12:
            return None
        return k

    cp = period_of(resp.sum(axis=0), 40, 200)      # 列周期：格宽 40~200px
    rp = period_of(resp.sum(axis=1), 40, 200)      # 行周期

    # ① 周期性合理区间：格宽是已知的（表单由 make_form 生成），±35% 之外否决。
    # 允许这么大的带宽是为了吸收照片的轻微透视/缩放；而实测的误检值
    # （40、50、60）全部落在 108×[0.75, 1.35] = [81, 146] 之外，够用。
    glo, ghi = cell * GRID_PERIOD_LO, cell * GRID_PERIOD_HI
    if cp is not None and not (glo <= cp <= ghi):
        cp = None
    if rp is not None and not (glo <= rp <= ghi):
        rp = None

    # ② 交叉校验：补字单的格子是方的，行列周期应当接近。
    # 用 >= 而非 >：周期恰为 1.5 倍时（实测 60 对 40）严格大于会两个都放过。
    if cp and rp and cp >= GRID_SQUARE_RATIO * rp:
        rp = None
    elif cp and rp and rp >= GRID_SQUARE_RATIO * cp:
        cp = None
    return cp, rp


def _grid_origin(resp_prof, period):
    """给定周期，找格线的相位（第一条格线落在哪）。

    做法：把投影按周期折叠（取模累加），格线相位处会叠出最高峰。

    注：原来还有个 span 参数，从来没被读过 —— 相位只由投影与周期决定，
    与"铺多少格"无关。删除以免误导（让人以为可以限制搜索范围）。
    """
    n = len(resp_prof)
    fold = np.zeros(period, dtype=np.float64)
    for i in range(n):
        fold[i % period] += resp_prof[i]
    return int(np.argmax(fold))


def _cell_ink(g, bg, lo, hi):
    """量一个格子里的墨：返回 (alpha, 平均墨深, 掩码)；格内没有可辨墨迹给 None。"""
    sub = g[lo[1]:hi[1], lo[0]:hi[0]]
    if sub.size == 0:
        return None
    bsub = bg[lo[1]:hi[1], lo[0]:hi[0]]
    alpha = np.clip((bsub - sub) / 255.0, 0, 1)
    m = alpha > 0.16
    if m.sum() < 12:
        return None
    return alpha, float(sub[m].mean()), m


def build_form_sheet(sample, out_dir, expect=None, pad=5, merge=False, debug=False,
                     replace=False, cols=12, cell=CELL, rebuild=False):
    """补字单专用切分：先检格子，再按格取字（原理见上方长注释）。"""
    os.makedirs(out_dir, exist_ok=True)
    im = open_rgb(sample)
    g = np.asarray(im.convert("L")).astype(np.float32)
    H, W = g.shape
    bg = ndimage.uniform_filter(g, size=81)
    ink = ((bg - g) > 25)
    ink = ndimage.binary_opening(ink, structure=np.ones((2, 2)))

    el = [c for c in (expect or "") if c not in " \t"]
    cper, rper = detect_grid(g, cell=cell)

    # 列：检到周期就用"已知的起点 + 实测的周期"；检不到就整组用 make_form 的常数。
    # 起点始终优先用 FORM_X0 —— 表单是程序自己画的，左边距不会有别的可能；
    # 照片的偏移/透视靠周期自适应就能吸收大半，不必也不该去猜起点。
    bg2 = ndimage.uniform_filter(g, size=41)
    resp = ndimage.gaussian_filter(np.clip((bg2 - g) / 255.0, 0, 1), sigma=1.0)
    if cper:
        ph = _grid_origin(resp.sum(axis=0), cper)
        # 相位换算成"第一个格子的左边界"：格线相位是边界，取第一个 >= FORM_X0 的
        k = int(np.ceil((FORM_X0 - ph) / float(cper)))
        cx0 = float(ph + k * cper)
        cw = float(cper)
        cdet_src = "实测周期 %d px" % cper
    else:
        cx0, cw = float(FORM_X0), float(cell)
        cdet_src = "内置几何（未检出格线周期）"
    if rper:
        ph = _grid_origin(resp.sum(axis=1), rper)
        k = int(np.ceil((FORM_Y0 - ph) / float(rper)))
        ry0 = float(ph + k * rper)
        rh = float(rper)
        rdet_src = "实测周期 %d px" % rper
    else:
        ry0, rh = float(FORM_Y0), float(cell)
        rdet_src = "内置几何（未检出格线周期）"
    if el:
        nrow = max(1, (len(el) + int(cols) - 1) // int(cols))
    else:
        # 没有 --expect 就不知道用户写了几个字，只能从纸上量能放几行。
        # 原来写死 nrow=1，后果是**只读第一行 12 格**：第 2 行起写的字
        # 被彻底忽略，而命令退出码仍是 0、输出还说"入库 N 个字形"。
        # 现在按纸面下缘估行数，宁可多排几个空格位也不漏行。
        nrow = max(1, int((H - FORM_Y0) // max(1.0, rh)))
        print("提示：这次没给 --expect，按纸面尺寸估了 %d 行（最多读 %d 格）。"
              % (nrow, int(cols) * nrow))
        print("  字符名只能编造（u0000、u0001…），渲染时对不上号，所以不建议入库。")
        print("  补字单**必须**带 --expect，否则字符名只能编造。")
        print("  请改用：--expect \"按格位顺序排列的字符\"。")
        print("  若你只是想看看切分对不对，可以继续；要入库请重跑并带上 --expect。")

    total = max(len(el), int(cols) * nrow)
    boxes, slots = [], []
    for k in range(total):
        cx = int(round(cx0 + (k % int(cols)) * cw))
        cy = int(round(ry0 + (k // int(cols)) * rh))
        boxes.append((cx, cy, int(round(cw)), int(round(rh))))
        slots.append(el[k] if k < len(el) else "u%04d" % k)

    kept, kept_chars, empty, depths = [], [], 0, []
    frag = 0
    for (cx, cy, bw, bh), ch in zip(boxes, slots):
        mx, my = int(bw * 0.08), int(bh * 0.08)     # 内缩 8%，把格线排除在外
        lo = (max(0, cx + mx), max(0, cy + my))
        hi = (min(W, cx + bw - mx), min(H, cy + bh - my))
        if hi[0] - lo[0] < 6 or hi[1] - lo[1] < 6:
            empty += 1
            continue
        got = _cell_ink(g, bg, lo, hi)
        if got is None:
            empty += 1
            continue
        _a, depth, mask = got
        depths.append(depth)
        if depth > FORM_INK_DEPTH:      # 整格都是浅灰：没写，或只剩浅灰参考字
            empty += 1
            continue
        ys, xs = np.where(mask)
        if len(ys) < 12:
            empty += 1
            continue
        # 残片闸门：窗口被误检的短周期压扁时，字会被拦腰切断，而残片能通过
        # 上面所有检查被静默命名入库 —— 比报错糟得多，因为输出看着是成功的
        # （"入库 2 个字形"），之后每次 render 这两个字都是坏的。
        # 判据锚在**已知格宽**上而不是窗口高：实测正常字占 cell 的 0.52，
        # 残片只占 0.19，比"窗口高一半"（0.48 对 0.50）稳得多。
        if (ys.max() - ys.min() + 1) < FORM_FRAG_SPAN * cell:
            empty += 1
            frag += 1
            continue
        x0, y0 = lo[0] + int(xs.min()), lo[1] + int(ys.min())
        x1, y1 = lo[0] + int(xs.max()) + 1, lo[1] + int(ys.max()) + 1
        kept.append([x0, y0, x1, y1])
        kept_chars.append(ch)

    print("格子感知切分：列 %s（格宽 %d px），行 %s（行高 %d px），共排布 %d 个格位。"
          % (cdet_src, int(cw), rdet_src, int(rh), len(boxes)))
    print("  有手写墨迹 %d 格，空着或只有浅灰参考字 %d 格。" % (len(kept), empty))
    if frag:
        print("  ⚠️ 有 %d 格里的墨迹太扁（不足格高的 %.0f%%），像是被切断的残片，已丢弃。"
              % (frag, FORM_FRAG_SPAN * 100))
        print("  这通常意味着格线周期没认准。请确认用的是当前版本 form 打印的补字单，")
        print("  并正对纸面拍（别斜拍、别让格子出画）。")

    if not kept:
        print("  → 没有一个格子里找到手写墨迹。这张补字单可能还没写，或下笔太轻。")
        if depths:
            print("  实测格内墨色平均 %d 灰（手写通常 25~60，浅灰参考字约 175）。" % int(np.median(depths)))
            print("  若字迹确实比 %d 灰还浅：下笔重一点，或换张曝光正常的照片。" % FORM_INK_DEPTH)
        if debug:
            Image.fromarray(np.asarray(g, dtype=np.uint8)).save(os.path.join(out_dir, "build_debug.png"))
        # empty=True 让调用方把退出码置为非 0。
        # 为什么必须这样：这种情况一个字都没入库，上层 `handglyph build ... && 下一步`
        # 会照常往下走，把"什么都没干"当成"建库成功" —— 和 P0-4 的拒绝覆盖是同一类
        # 静默失败，只是换了个触发条件（旧库非空 vs 本次零入库）。
        return {"glyphs": 0, "chars": 0, "atlas": None, "weak": "", "form_sheet": True,
                "empty": True, "message": "补字单上没有检测到手写墨迹"}

    if el and len(el) != len(kept):
        gap = [el[i] for i in range(len(el)) if i >= len(kept_chars)]
        print("提示：--expect 给了 %d 个字，实际只读到 %d 个有墨的格子。" % (len(el), len(kept)))
        if gap:
            print("  按格位顺序推断，这些字还没写或没识别到：%s" % "".join(gap))
            print("  它们不会入库，下次补字单记得把这几个字写全。")

    if debug:
        dbg = im.copy()
        dr2 = ImageDraw.Draw(dbg)
        for cx, cy, bw, bh in boxes:
            dr2.rectangle([cx, cy, cx + bw, cy + bh], outline=(200, 200, 60), width=2)
        for b in kept:
            dr2.rectangle([b[0], b[1], b[2], b[3]], outline=(30, 170, 60), width=3)
        dbg.save(os.path.join(out_dir, "build_debug.png"))
        print("  黄框=识别到的格位，绿框=入库字形，见 build_debug.png")

    res = _store_glyphs(im, g, bg, kept, kept_chars, out_dir, expect=expect,
                        pad=pad, merge=merge, replace=replace, source=sample,
                        rebuild=rebuild)
    if res.get("refused"):
        # reason 必须一起带出去 —— 只给退出码不给原因，用户只知道"失败了"，
        # 不知道是"库已存在"还是"参数冲突"。build() 里已经踩过这个坑。
        return {"glyphs": 0, "chars": res["chars"], "atlas": None, "weak": "",
                "form_sheet": True, "refused": True,
                "reason": res.get("reason", "")}
    return res


def _write_atlas(im, boxes, out_dir, pad, cols=12, cell=CELL):
    """画图集：入库字形按顺序铺开，供人工核对切分对不对。"""
    W, H = im.size
    rowsn = (len(boxes) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, max(1, rowsn) * cell), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    f = cjk_font(15)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        c = im.crop((max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)))
        sc = min((cell - 26) / max(1, c.width), (cell - 26) / max(1, c.height))
        c = c.resize((max(1, int(c.width * sc)), max(1, int(c.height * sc))), Image.LANCZOS)
        cx, cy = (k % cols) * cell, (k // cols) * cell
        sheet.paste(c, (cx + (cell - c.width) // 2, cy + 22))
        dr.text((cx + 3, cy + 3), str(k), fill=(200, 0, 0), font=f)
    sheet.save(os.path.join(out_dir, "atlas.png"))


def write_atlas_from_lib(lib_path, cols=12, cell=None):
    """从字库里已有的字形重画一份 atlas.png（不需要原始照片）。

    为什么需要独立入口：原来 atlas.png 只在 _store_glyphs 里画，也就是
    **只有 build 的那一次**会产图鉴。而 atlas --fix 这条路的操作顺序恰恰是
    "先看图鉴、再写映射"—— 图鉴是它唯一的输入。用户改了字库（改名、清理、
    从别处拷来）之后就没图鉴可看了，而 atlas 子命令又只会打印用法。

    序号语义与 build 时一致：优先用 instances 里的 seq（全局唯一）；
    老字库没有 instances 时退化成"按 chars 的遍历顺序编号"，并在图上标明。
    返回 (图鉴路径, 画的字形数)；字库为空时返回 (None, 0)。
    """
    cell = CELL if cell is None else cell
    man = load_manifest(lib_path)
    items = []                      # [(序号, 字, 绝对路径)]
    inst = man.get("instances") or {}
    seq = 0
    # 没有实例档案的老字库：序号只能退化成"遍历顺序"，语义与 build 时的全局序号
    # **不同**，必须在图上写明 —— 否则用户会照着红字去写 `序号=字`，而那个序号
    # 在老字库里根本不是全局唯一的（见 atlas_fix 的同名防护）。
    no_inst = not bool(inst)
    for ch, rels in man.get("chars", {}).items():
        for rel in rels:
            p = safe_rel(lib_path, rel)
            if p is None or not os.path.isfile(p):
                continue
            rec = inst.get(rel) or {}
            idx = rec.get("seq")
            if idx is None:
                idx = seq
            items.append((int(idx), ch, p))
            seq += 1
    if not items:
        return None, 0
    items.sort(key=lambda t: t[0])
    rowsn = (len(items) + cols - 1) // cols
    foot = cell if no_inst else 0
    sheet = Image.new("RGB", (cols * cell, max(1, rowsn) * cell + foot), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    f = cjk_font(15)
    for k, (idx, ch, p) in enumerate(items):
        a = load_alpha(p)
        # ⛔ 不能把 alpha 直接当灰度画：alpha 是"墨有多少"，墨处 = 1.0 → 255，
        # 画出来是**黑底白字**；再叠加 paste 不传 mask（整块矩形替换到白纸上），
        # 每个格位会变成一个黑方块 —— 与 build 时的图鉴（纸色底、深色笔画）
        # 对比度完全相反，而图鉴唯一的用途就是看笔画粗细/发虚，反相后判断就失真了。
        # 正确做法两条：
        #   墨色 = 反相(1 - a) → 墨处黑、纸处白
        #   形状 = a 当 mask   → 只贴笔画本身，"非墨"处露出底下的白纸
        gimg = Image.fromarray(np.clip((1.0 - a) * 255, 0, 255).astype(np.uint8), "L")
        mask = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8), "L")
        sc = min((cell - 26) / max(1, gimg.width), (cell - 26) / max(1, gimg.height))
        size2 = (max(1, int(gimg.width * sc)), max(1, int(gimg.height * sc)))
        gimg = gimg.resize(size2, Image.LANCZOS)
        mask = mask.resize(size2, Image.LANCZOS)
        cx, cy = (k % cols) * cell, (k // cols) * cell
        sheet.paste(Image.merge("RGB", (gimg, gimg, gimg)),
                    (cx + (cell - gimg.width) // 2, cy + 22), mask)
        dr.text((cx + 3, cy + 3), str(idx), fill=(200, 0, 0), font=f)
        dr.text((cx + cell - 18, cy + 3), ch, fill=(60, 90, 180), font=f)
    if no_inst:
        # 分两行写：PIL 不换行，窄图鉴（--cols 小时）一行会被裁掉后半句。
        dr.text((6, max(1, rowsn) * cell + 6),
                "注意：本字库没有实例档案（manifest 缺 instances），红字是「遍历顺序」，",
                fill=(200, 0, 0), font=f)
        dr.text((6, max(1, rowsn) * cell + 24),
                "不是 build 时的全局序号 —— 写映射请用 `字符/序号` 形式（如 确/0）。",
                fill=(200, 0, 0), font=f)
    out = os.path.join(lib_path, "atlas.png")
    sheet.save(out)
    return out, len(items)


def _store_glyphs(im, g, bg, boxes, chars, out_dir, expect=None, pad=5, merge=False,
                  replace=False, source=None, rebuild=False):
    """把 (框, 字符) 落成字形库。两种切分共用这段落库逻辑。

    merge 时顺手把每个实例的档案（score / src / date）记进 manifest["instances"]，
    这样 merge 之后的"新字版 vs 老字版"质量对比可以持续追踪，
    prune 也能按来源（src）整批淘汰某一张拍坏的字版，而不是只能按分数散着删。

    ⛔ **拒绝毁库**：out_dir 里已有非空字库、而本次既没 merge 也没明确要求 rebuild 时，
    直接报错退出。这是必须拦的一道 —— 原实现从空骨架起步、末尾无条件回写
    manifest.json，所以"换一张照片重跑 build"会把旧 chars 映射整表抹掉，
    而旧 PNG 还留在磁盘上（变成谁也找不到的孤儿），输出里一个警告都没有。
    实测：4 个字的字库被覆盖成 2 个新字，退出码仍是 0。
    """
    H, W = g.shape
    mp = os.path.join(out_dir, "manifest.json")
    old_man = None
    if os.path.isfile(mp):
        try:
            with open(mp, encoding="utf-8-sig") as _f:
                old_man = json.load(_f)
        except Exception:
            old_man = None
    if man_has_glyphs(old_man) and not merge and not rebuild:
        n_old = len(old_man.get("chars") or {})
        print("拒绝执行：目标目录里已经有一个 %d 字的字库：%s" % (n_old, mp))
        print("")
        print("  直接往里 build 会把旧的字形映射**整表覆盖**，旧 PNG 会变成没人")
        print("  找得到的孤儿文件。这不是你想要的，所以这里停下了。")
        print("")
        print("  想干什么就选哪个参数：")
        print("    · 往字库里**加**新字（最常用）   加 --merge")
        print("    · 只**重写**照片上这几个字       加 --merge --replace")
        print("    · 就是想**推倒重建**一个字库     加 --rebuild（确认旧字都不要了）")
        return {"glyphs": 0, "chars": n_old, "atlas": None, "weak": "",
                "refused": True, "reason": "字库已存在且未指定 --merge / --rebuild"}

    man = old_man if (merge and old_man is not None) else None
    if man is None:
        man = {"source": os.path.basename(str(out_dir)), "chars": {}}
    man.setdefault("instances", {})
    src_name = os.path.basename(source) if source else man.get("source", "")
    today = time.strftime("%Y-%m-%d")

    # ---- 分批登记（2026-09-17 用户要求）----
    #
    # 为什么要有"批次"：用户补字是**分次写**的。不同时候写的字，落笔轻重会变
    # （换了笔、换了纸、隔了几天手感不同），而**同一批里各个字之间是齐的**。
    # 所以比"库内粗细一致性"时，口径不能是全库取中位 —— 那会把一整批的偏移
    # 摊到每个字上，反而看不出"这批跟老库不是一路粗细"。
    # 正确的比对是**批与批之间**比。
    #
    # batch_id 的生成规则：本库现有批次数 + 1，形如 "b3"。
    # 同一批内所有实例共享一个 batch_id，所以能与实例的 date/src 对上。
    # merge 时开新批次（本次写的就是新的一批）；rebuild 时为 b1。
    batches = man.setdefault("batches", [])
    if merge and batches:
        batch_id = "b%d" % (len(batches) + 1)
    else:
        batch_id = "b1"
        batches.clear()
    # 每批记下：编号、来源照片、日期、本次入库的字符数（后面填）
    this_batch = {"id": batch_id, "src": src_name, "date": today,
                  "chars": [], "stroke_med": None}
    batches.append(this_batch)

    qual, fresh, bad_n = {}, {}, 0
    bad_detail = []         # 被拒的字：(字, 原因)，非残片的拒收要说出来
    seq = 0                 # 本次入库的全局序号，写进实例档案供 atlas --fix 定位
    for (x0, y0, x1, y1), ch in zip(boxes, chars):
        d = os.path.join(out_dir, "glyphs", str(ord(ch)) if len(ch) == 1 else ch)
        os.makedirs(d, exist_ok=True)
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        crop = im.crop((cx0, cy0, min(W, x1 + pad), min(H, y1 + pad)))
        p = os.path.join(d, "%02d.png" % next_index(d))
        # alpha 用"局部背景差分"而非灰度反转：255-L 会把不均匀光照与纸纹阴影
        # 以 0.2~0.4 的低 alpha 灰雾一起烤进字形，是入库字形"发虚"的源头。
        gsub = np.asarray(crop.convert("L")).astype(np.float32)
        bsub = bg[cy0:cy0 + gsub.shape[0], cx0:cx0 + gsub.shape[1]]
        if bsub.shape != gsub.shape:
            bsub = np.full(gsub.shape, 255.0, np.float32)
        alpha = np.clip((bsub - gsub) / 255.0, 0, 1)
        alpha = np.where(alpha < 0.16, 0, alpha)
        m = alpha > 0.5
        if m.sum() > 8 and crop.width > 12:
            lb2, n2 = ndimage.label(m, structure=np.ones((3, 3)))
            for i2, sl2 in enumerate(ndimage.find_objects(lb2), start=1):
                if sl2 is None:
                    continue
                hh2 = sl2[0].stop - sl2[0].start
                ww2 = sl2[1].stop - sl2[1].start
                if ww2 >= 0.80 * crop.width and hh2 <= 4:
                    alpha[sl2][lb2[sl2] == i2] = 0.0
        ink2 = alpha > 0.30
        if ink2.sum() > 8:
            ys2, xs2 = np.where(ink2)
            alpha = alpha[max(0, ys2.min() - 1):ys2.max() + 2,
                          max(0, xs2.min() - 1):xs2.max() + 2]
        alpha, _sp = split_bleed(alpha)
        alpha = normalize_ink(alpha)
        bad_reason = plaus(alpha, ch=ch)
        if bad_reason:
            bad_n += 1
            if bad_reason != "残片":
                bad_detail.append((ch, bad_reason))
            continue
        rgba = np.dstack([np.full(alpha.shape + (3,), ink_rgb(), np.uint8),
                          (alpha * 255).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(p)
        rel = os.path.relpath(p, out_dir).replace("\\", "/")
        fresh.setdefault(ch, []).append(rel)
        sc = score_alpha(alpha)
        qual.setdefault(ch, []).append(sc)
        # 全局序号 seq 是 atlas.png 上的编号（本次切分的第几个字），
        # 与文件名无关 —— 文件名在不同字符目录下会重名（都叫 00.png），
        # 拿它当序号会让 atlas --fix 定位错乱。
        man["instances"][rel] = {"score": round(float(sc), 4),
                                 "src": src_name, "date": today, "seq": seq,
                                 "batch": batch_id}
        seq += 1

    # replace 在循环外统一落地：本次该字一个都没成功时，旧字形完好无损。
    # 同时清理 instances 里指向已删文件 / 已不在 chars 里的孤儿条目。
    for ch, rels in fresh.items():
        if replace:
            for old_rel in man["chars"].get(ch, []):
                fp = os.path.join(out_dir, old_rel)
                if os.path.isfile(fp) and old_rel not in rels:
                    os.remove(fp)
                man["instances"].pop(old_rel, None)
            man["chars"][ch] = list(rels)
        else:
            man["chars"].setdefault(ch, []).extend(rels)
    alive = {r for rels in man["chars"].values() for r in rels}
    for rel in list(man["instances"]):
        if rel not in alive:
            man["instances"].pop(rel, None)

    # ---- 登记本批次的字符，并量出"本批的笔画粗细"（供批间比对）----
    this_batch["chars"] = sorted(fresh)
    this_batch["stroke_med"] = _batch_stroke_median(out_dir, fresh)
    # 老批次若还没量过粗细（老库升级上来的），补量一次，否则没得比。
    for b in man.get("batches", []):
        if b is not this_batch and b.get("stroke_med") is None:
            b["stroke_med"] = _batch_stroke_median(out_dir, _batch_chars(man, b))

    # 注：这里曾有一行 `drift = compare_batches(man)`，结果从未被读过 ——
    # 而真正要用的报告在下面用 batch_report() 又重算了一遍（还要重读一次 manifest）。
    # 一次白做的全库笔画扫描，删掉。
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)

    _write_atlas(im, boxes, out_dir, pad)
    n_new = sum(len(v) for v in fresh.values())
    # 字库变了，质检缓存必须作废 —— 否则 quality/prune 拿旧分布当基准，
    # 新并入的这批字会被旧中位数判成离群（详见 invalidate_metrics 的说明）。
    # 只在真的写进了字形时才删：一张空补字单不该把缓存清掉重算一遍。
    if n_new:
        invalidate_metrics(out_dir)
    weak = {c: max(v) for c, v in qual.items() if max(v) < 0.62}
    for c in weak:
        WEAK.add(c)
    print("入库：%d 个字形 / %d 个不同字符" % (n_new, len(man["chars"])))
    if bad_n:
        print("  （另有 %d 个格子写得太差，未入库）" % bad_n)
    if bad_detail:
        # 非残片的拒收必须列出来 —— 用户看到"少了一个字"却不知道原因时，
        # 除了重拍一遍没有别的办法。列出字与原因，至少能对症（写小了？写太瘦？）
        seen = {}
        for c, why in bad_detail:
            seen.setdefault(why, []).append(c)
        for why, cs in seen.items():
            print("  其中 %d 个是「%s」，未入库：%s"
                  % (len(cs), why, "".join(sorted(set(cs)))))
        if any(why == "形状可疑" for why in seen):
            print("  「形状可疑」指这个字的宽高比明显超出正常范围 ——")
            print("  常见原因是裁到了半个字、或按住了笔尖把字写成一个点。")
            print("  补字单模式下请确认字写在格子正中、占七成大小。")
    if weak and expect:
        order = sorted(weak, key=lambda c: weak[c])
        print("清晰度偏低，建议重写这 %d 个字：" % len(order))
        print("  " + " ".join(order))
        print("  handglyph form \"%s\" -o 补字单.png" % "".join(order))
    elif weak:
        print("有 %d 个字形清晰度偏低；带上 --expect 命名后才能给出可用的补字清单。" % len(weak))
    else:
        print("清晰度检验：全部通过。")

    # ---- 批间粗细比对（只在多批次时才有意义）----
    # 为什么放在入库的末尾提示：用户刚写完一批新字，正是他"还能再写一遍"的时候。
    # 等到下次渲染才发现"这批字跟老库不是一路粗细"，就得重新摆纸笔了。
    drift_over = False
    if len([b for b in man.get("batches", []) if b.get("stroke_med")]) >= 2:
        b_lines, drift_over = batch_report(out_dir)
        if drift_over:
            print("")
            for ln in b_lines:
                print(ln)
    return {"glyphs": n_new, "chars": len(man["chars"]),
            "atlas": os.path.join(out_dir, "atlas.png"),
            "weak": "".join(sorted(weak, key=lambda c: weak[c])), "form_sheet": True,
            "batch": batch_id, "batch_drift": drift_over,
            "_weak": weak}


def make_form(chars, out_path, cols=12, cell=CELL, copies=1):
    """生成补字单。

    说明文字用 #B4B4B4（180 灰）、格内参考字用 #C8C8C8（200 灰），都刻意浅于
    FORM_INK_DEPTH(100)。原来的说明文字是 70 灰 —— 跟钢笔墨色几乎一样深，
    build 会把它们当成笔迹吃进字库。现在改成浅灰是**双重保险**：
    即便走的不是 --form-sheet 模式，纯靠墨深也能把打印内容滤掉。

    格子几何（cols / cell / 起点）与 build --form-sheet 共用 FORM_X0 / FORM_Y0 /
    cell 三个常数，两边永远对得上。

    copies：每个字连着排几格，让用户一次抄几遍（默认 1 遍）。
    为什么要这个：同一个字写多遍，入库时才有**多实例**可轮转 —— 渲染时同一个字
    在页面上出现两次就不会长得一模一样（那是"盖章感"的来源），
    quality/prune 也能在多个实例里择优。只写一遍的话这个机制无从发挥。
    格位顺序 = chars 里每个字重复 copies 次后的顺序，所以 build 的 --expect
    也要按同样的顺序写（用 form_expect() 生成，别手敲）。
    """
    if not chars:
        print("没有要写的字，补字单是空的，未生成。")
        print("  用法：handglyph form \"要补的那些字\" -o 补字单.png")
        print("  想看该补哪些字：handglyph coverage --lib library -t \"你要写的文字\"")
        return None
    copies = max(1, int(copies))
    # ---- 容量校验：列数/行数超出纸面时必须拒绝，不能画到纸外 ----
    # 格子画到纸外会被 PIL 静默裁掉，而 build 端会把画外的格位当"空位"跳过，
    # 表现是"入库的字少了几个"，全程没有报错。--copies 会让格位翻倍，
    # 所以必须按**展开后**的格数算。
    n_cells = len(chars) * copies
    max_cols = max(1, int((FORM_W - FORM_X0) // max(1, cell)))
    if not (1 <= int(cols) <= max_cols):
        print("列数 %d 超出一页补字单的可用宽度：最多 %d 列（格宽 %d px，纸宽 %d px）。"
              % (cols, max_cols, cell, FORM_W))
        print("  请把 --cols 改到 1~%d；build --form-sheet 必须用同一个值。" % max_cols)
        return None
    max_rows = max(1, int((FORM_H - FORM_Y0) // max(1, cell)))
    n_rows = (n_cells + int(cols) - 1) // int(cols)
    if n_rows > max_rows:
        cap = max_rows * int(cols)
        print("一页放不下：%d 个字 × %d 遍 = %d 个格位，按 %d 列排需要 %d 行，"
              "最多只放得下 %d 行（%d 个格位）。"
              % (len(chars), copies, n_cells, cols, n_rows, max_rows, cap))
        print("  请拆成几张补字单（每张不超过 %d 个格位），或加大 --cols（上限 %d 列）。"
              % (cap, max_cols))
        return None
    # 字体缺失时说明文字会画成一串方块或干脆不显示 —— 用户拿到一张空白格子的
    # 补字单，完全不知道该怎么写。这里把话说在前面。
    for ln in font_warning():
        print(ln)
    cx0, cy0, cell = FORM_X0, FORM_Y0, cell
    sheet = Image.new("RGB", (FORM_W, FORM_H), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    ft = cjk_font(40, bold=True)
    fs = cjk_font(30)
    fg = cjk_font(62)
    dr.text((90, 80), "补字单", font=ft, fill=(120, 120, 120))
    dr.text((90, 145), "照抄下面每个字，写在对应格子里。写得好不好直接决定成图效果，请按下面要求写：", font=fs, fill=(150, 150, 150))
    tips = [
        "写：用平时写作业的那支笔，整张纸同一支笔写完",
        "写：每个字写进格子里、占七成大小，不要顶格、不要压格线",
        "写：一笔一画写清楚，笔画不要粘连；不要潦草、不要写连笔草书",
        # ⛔ 这条是用户 2026-09-17 明确要求加的：**粗细要一致**。
        # 为什么必须写进提示词而不是靠程序后期修：程序改粗细靠"整字等比缩放"，
        # 而缩放会连带着改字号、改清晰度（细笔画字形一放大就糊，见分辨率闸门），
        # 是被动补救。写的时候手上有意识地控制轻重，才是根治。
        "写：**每个字的笔画粗细要尽量一致**（落笔轻重要均匀），不要有的字粗、有的字细，也不要一个字里一半粗一半细",
        "写：按平时速度写，字与字之间留一点缝",
        "拍：正对纸面拍，别斜拍别隔远，字要占画面三分之一以上",
        "拍：光线均匀、不开闪光灯、别让一边有阴影；对焦在字上",
        "拍：拍完放大看一眼，笔画边缘要清楚、不能发灰",
        "拍：发原图，不要压缩、不要美颜滤镜",
    ]
    if copies > 1:
        # 遍数只用来**加字库容量**：同一个字多几个实例，排版时轮着用，就不至于
        # 满篇同一副样子。它不负责"变粗细" —— 粗细是 vary() 在微妙档里做的。
        #
        # 为什么是 3 不是 5：抄到第 4、5 遍人已经开始烦了，越写越敷衍，
        # 那几个实例的质量反而比前 3 遍差，等于往字库里掺次品。
        # 用户 2026-09-17 的原话：「抄多了他心里会厌烦，抄 5 遍的话，
        # 他还不如自己写呢。所以我们定的抄 3 遍。」
        # ⚠ 索引用"找第一条 '写：按平时速度' 之前插入"，不要写死数字 ——
        # 上面 tips 增删一条就会让写死的索引插错位置（宁可多一次查找）。
        at = next((i for i, t in enumerate(tips) if "按平时速度" in t), len(tips))
        tips.insert(at, "写：同一个字连着 %d 格都写一遍，每遍按平时那么写就行，"
                        "不用刻意变样（写多了反而会敷衍，就是 %d 遍）" % (copies, copies))
    if copies >= 3:
        at2 = next((i for i, t in enumerate(tips) if "按平时速度" in t), len(tips))
        tips.insert(at2, "写：越往后越要跟第一遍一样，粗细也保持一致。"
                         "敷衍的后几遍会把字库拉低，不如不写")
    # 行距随条数自适应：条数会随 copies 变，写死 40 会把最后几条挤出画面。
    lh = 40 if len(tips) <= 10 else 34
    for i, t in enumerate(tips):
        dr.text((90, 195 + i * lh), t, font=fs, fill=(165, 165, 165))
    top = cy0
    # 每个字展开成 copies 格：E E E  No. 或者汉字「永 永 永」
    cells_ch = [ch for ch in chars for _ in range(copies)]
    for i, ch in enumerate(cells_ch):
        cx, cy = cx0 + (i % cols) * cell, top + (i // cols) * cell
        dr.rectangle([cx, cy, cx + cell, cy + cell], outline=(120, 120, 120))
        dr.line([cx + cell // 2, cy, cx + cell // 2, cy + cell], fill=(205, 205, 205))
        dr.line([cx, cy + cell // 2, cx + cell, cy + cell // 2], fill=(205, 205, 205))
        bb = dr.textbbox((0, 0), ch, font=fg)
        dr.text((cx + (cell - (bb[2] - bb[0])) / 2 - bb[0],
                 cy + (cell - (bb[3] - bb[1])) / 2 - bb[1]), ch, font=fg, fill=(200, 200, 200))
    sheet.save(out_path)
    return out_path


def form_expect(chars, copies=1):
    """补字单的格位顺序（打印在抬头供 --expect 照抄）。

    与 make_form 的展开方式**必须一致**：改了一处忘了另一处，--expect 就会和
    格位错开，入库时每个字都被安到错的名字上 —— 而且不报错（字数对得上）。
    所以这个展开只写在这一处，两边都从它取。
    """
    copies = max(1, int(copies))
    return "".join(ch for ch in chars for _ in range(copies))



SHOOT_TIPS = [
    "1  正对纸面拍，别斜着拍，也别隔太远（字要占画面三分之一以上）",
    "2  光线均匀、别开闪光灯、别让一边有阴影",
    "3  对焦在字上，拍完放大看一眼：笔画边缘要清楚，不能发灰",
    "4  发原图，不要压缩、不要美颜滤镜",
    "5  用黑笔写，别用太细的笔；纸铺平，不要折皱",
    "6  每个字写进格子里、占七成大小、字间留缝；不要压在格线上",
    "7  一次写不完可以分几张，但每张都要按左到右、上到下的顺序",
]


def shoot_guide():
    return ["【拍摄要求】"] + ["  " + t for t in SHOOT_TIPS]


def refill_notice(chars, reason):
    """统一的补库提示：说明这是样张问题，并给出可直接执行的命令。"""
    s = "".join(chars)
    out = ["", "【补库提示】%s" % reason,
           "  这类问题出在你的字版样张上，不是程序算错：",
           "  · 发虚/糊 → 拍得太远、对焦不准、光线不匀，笔画边缘发灰",
           "  · 残片/尺寸偏小 → 裁到了半个字，或字写得太小、太挤",
           "  · 命名错位 → 字版上有打印抬头、墨点、格线干扰",
           "  按下面三步替换旧字形（--replace 会去掉旧实例，不并存）："]
    if s:
        out.append("    handglyph form \"%s\" -o 补字单.png" % s)
        out.append("    handglyph build 补字单照片.jpg --expect \"%s\" --merge --replace -o library" % s)
    else:
        out.append("    handglyph form <缺的字> -o 补字单.png")
        out.append("    handglyph build 补字单照片.jpg --expect <同样的字> --merge --replace -o library")
    out.append("    handglyph clean --lib library")
    out += shoot_guide()
    return out


def metrics_of(a):
    """字形客观指标（与具体人、具体字无关）：尺寸、实度、填充、发虚、锐度、碎片。"""
    ink = a > 0.55
    n = int(ink.sum())
    if n < 8:
        return None
    h, w = a.shape
    comps = ndimage.label(ink, structure=np.ones((3, 3)))[1]
    edge = ink & ~ndimage.binary_erosion(ink, structure=np.ones((3, 3)))
    mid = (a > 0.18) & (a < 0.82)
    return {
        "h": int(h),
        "w": int(w),
        "ar": float(h) / float(max(1, w)),
        "fill": float(n) / float(a.size),
        "solid": float(a[ink].mean()),
        "soft": float(mid.sum()) / max(1.0, float(edge.sum())),
        "sharp": edge_ratio(ink),
        "frag": float(comps) / max(1.0, n / 1000.0),
        "comps": int(comps),
        "ink": n,
    }


METRIC_KEYS = ("sharp", "soft", "frag", "solid", "fill")
LOWER_BETTER = ("soft", "frag")
METRIC_FLOOR = {"sharp": 0.08, "soft": 0.02, "frag": 0.5, "solid": 0.01, "fill": 0.02}


def script_class(ch):
    if ch.isascii():
        return "latin" if ch.isalnum() else "punct"
    return "cjk"


# ---------------------------------------------------------------- 字形形状校验
# 解决的问题：裁切时若抓取区域偏窄/偏扁，字形会被"压扁"存进字库，
# 渲染时又按该比例忠实放大，成品里就出现一个变形的字（如 确、能）。
# 这类缺陷与"发虚/碎片/浓度"无关，原有质检指标全部看不见，必须单独查。

# 宽高比偏差阈值。取值权衡：真被压扁的字（确 1.89 倍）必须抓住，
# 而"这位书写者的某个字天生偏窄"（门 1.80 倍）会一起被抓 —— 实测无法用阈值干净分开，
# 故阈值放在 1.7 并配合"只报告不改动"，由用户看报告自行判断要不要重写。
SHAPE_RATIO_LIMIT = 1.7
_ref_cache = {}


def _ref_font(px=200):
    """参照字体。找不到中文字体时返回 None，形状校验自动跳过（不报错、不误报）。"""
    path = cjk_font_path()
    if not path:
        return None
    try:
        return ImageFont.truetype(path, px)
    except Exception:
        return None


def _ref_ar(ch):
    """参照宽高比：用系统字体渲染该字，量墨迹框的 高/宽。"""
    if ch in _ref_cache:
        return _ref_cache[ch]
    f = _ref_font()
    val = None
    if f is not None:
        try:
            px = 200
            im = Image.new("L", (px * 3, px * 3), 0)
            d = ImageDraw.Draw(im)
            bb = d.textbbox((0, 0), ch, font=f)
            d.text((px - bb[0], px - bb[1]), ch, font=f, fill=255)
            arr = np.asarray(im) > 128
            ys, xs = np.where(arr)
            if len(ys):
                h = ys.max() - ys.min() + 1
                w = xs.max() - xs.min() + 1
                val = float(h) / float(max(1, w))
        except Exception:
            val = None
    _ref_cache[ch] = val
    return val


def squeeze_report(ch, a, limit=None):
    """判断字形是否被压扁，返回 dict 或 None（无法判定时）。

    判据只有一条：**实际宽高比 / 参照宽高比 超过 limit**（默认 1.6）。

    为什么只查汉字：字母、符号（`-`、`(`、`1`）在手写里与印刷体的比例差别本来就极大，
    用字体当参照必然误报，实测 `-` 被判偏窄 2.37 倍而它完全正常。
    汉字的方块字形有稳定的宽高比预期，才适合用字体参照。

    为什么不加"结构间隙"旁证：实测系统字体的 `确`（石+角）在 200px 下两部件相触、
    字内没有竖向缝隙，而手写的 `门` 反而没有缝隙 —— 用"间隙有无"当旁证，
    会同时漏掉 确 又误伤 门，不可靠。

    ⛔ 检测出来**不能靠横向拉伸修复**：裁窄时丢掉的横向细节已经不存在，
    拉宽只会把笔画拉胖（实测 确 31→59px 后变成一坨）。唯一正确的处置是重写。
    """
    if ch.isascii() or not ch.isalnum():
        return None          # 只查汉字
    if min(a.shape) < 12:
        return None          # 太小量不准
    r = _ref_ar(ch)
    if r is None:
        return None
    limit = SHAPE_RATIO_LIMIT if limit is None else limit
    h, w = a.shape
    ar = h / float(max(1, w))
    ratio = ar / r
    if ratio < limit:
        return None
    return {"h": int(h), "w": int(w), "ar": ar, "ref_ar": r, "ratio": ratio}


def _is_cjk_char(ch):
    """是不是一个汉字（CJK 统一表意文字主区）。

    只判主区 4E00–9FFF：这个项目的字库只装常用汉字 + 拉丁字母 + 数学符号，
    扩展区（B 及以后）的字不在考虑范围内，多判反而会给 `_ref_ar`（按字体量
    参照宽高比）喂进它取不到参照的字，白白走一遍异常路径。
    """
    return len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"


def thin_report(ch, a):
    """判断非汉字字形是否墨量低到不可用，返回 dict 或 None。

    为什么和 squeeze_report 分开：汉字有稳定的宽高比预期，可以拿字体当参照；
    字母和符号没有（`-`、`(`、`1` 的手写比例与印刷体差得远），拿字体比必然误报。
    但这些字形**不是没问题**，只是问题不在比例上，而在**墨量绝对值** ——
    裁切时如果只框到一小段残笔，字形尺寸会小、墨点也少，贴版放大就成了黑块。

    判据只用墨点像素数，不看宽高比。阈值取自 GLYPH_MIN_INK，与 render 的
    贴版前检查**共用同一个常量** —— 两处如果各写一个数，改了一处另一处就漏，
    "shape 报了警但 render 照贴"这种自相矛盾的状态会很难查。

    ⚠ 检出后**不能靠拉伸修复**，和压扁一样只能重写：残笔丢掉的笔画已经不存在了。

    ⚠ 这里的守卫是"**是不是汉字**"，不是"是不是 ASCII"。
    `·`（U+00B7）**不是** ASCII 字符，用 `ch.isascii()` 当筛子会把它连同汉字
    一起挡在门外 —— 而它恰恰是本题要抓的坏字形（实测 18×11、47 个墨点）。
    这类"看着像 ASCII 其实不是"的字符正是最容易被漏掉的一批，所以判据取
    CJK 区段（汉字有稳定宽高比，才走得到 squeeze_report；其余一律按墨量查）。
    """
    if _is_cjk_char(ch):
        return None                       # 汉字走 squeeze_report，不重复报
    if a is None or a.size == 0:
        return None
    n = int((a > 0.5).sum())
    if n >= GLYPH_MIN_INK:
        return None
    h, w = a.shape
    return {"h": int(h), "w": int(w), "ink": n, "min_ink": GLYPH_MIN_INK}

def scan_shapes(lib_path, limit=None):
    """全库扫描有问题的字形。返回 [(字, rel, shape, 报告), ...]。

    两类问题：
      · **被压扁**（squeeze_report）—— 只查汉字，用字体宽高比当参照。
      · **墨量太低**（thin_report）—— 查字母与符号。这类字**根本没法用宽高比判**
        （`-` 被判偏窄 2.37 倍而完全正常，见 squeeze_report 的说明），
        但它们的墨量绝对值是可以直接量的：`·` 实测 18×11 只有 47 个墨点，
        贴到版面上必然糊成实心块。这是原实现的一个判据盲区 ——
        `shape` 原先只走 squeeze_report，于是 `·` 这种坏字形一路静默通过。
    """
    out = []
    try:
        with open(os.path.join(lib_path, "manifest.json"), encoding="utf-8-sig") as f:
            man = json.load(f)
    except Exception:
        return out
    for ch, rels in man.get("chars", {}).items():
        for rel in rels:
            p = os.path.join(lib_path, rel)
            if not os.path.exists(p):
                continue
            a = load_alpha(p)
            rep = squeeze_report(ch, a, limit=limit)
            if rep is None:
                rep = thin_report(ch, a)
                if rep:
                    # 标出问题种类，报告里两种混排时才看得出区别。
                    rep = dict(rep, kind="thin")
            else:
                rep = dict(rep, kind="squeeze")
            if rep:
                out.append((ch, rel, a.shape, rep))
    return out


def shape_lines(lib_path, limit=None):
    """形状校验报告。返回 (报告行列表, 异常字列表)。

    连字符列表一起返回，调用方（quality --form）才能把形状异常的字
    也收进补字单 —— 原先是靠解析自己打印的报告文本来猜，既脆又漏。
    """
    found = scan_shapes(lib_path, limit=limit)
    if not found:
        return ["形状校验：全部字形的宽高比与墨量都在正常范围，未发现不可用的字形。"], []
    chars = sorted({ch for ch, _r, _s, _i in found})
    squeeze = [t for t in found if t[3].get("kind") == "squeeze"]
    thin = [t for t in found if t[3].get("kind") == "thin"]
    lines = ["形状校验：发现 %d 个字形不可用：" % len(found)]
    if squeeze:
        lines.append("  — 被压扁（宽高比异常）%d 个：" % len(squeeze))
        for ch, rel, shp, rep in squeeze:
            lines.append("    %s  %dx%d  宽高比 %.2f（参照 %.2f，偏窄 %.2f 倍）"
                         % (ch, shp[1], shp[0], rep["ar"], rep["ref_ar"], rep["ratio"]))
    if thin:
        lines.append("  — 墨量太低（贴到版面上会糊成黑块）%d 个：" % len(thin))
        for ch, rel, shp, rep in thin:
            lines.append("    %s  %dx%d  墨点 %d 个（下限 %d）"
                         % (ch, shp[1], shp[0], rep["ink"], rep["min_ink"]))
    lines.append("  这两类问题**都不能靠拉伸修复** —— 裁切时丢掉的细节已经没了，"
                 "拉宽只会把笔画拉胖。必须重写这些字。")
    lines += refill_notice(chars, "字库里有 %d 个字形的形状或墨量不对。" % len(chars))
    lines.append("  重写后如仍报此错，检查写的时候每个字是否都写进了格子、没有写得太瘦长，"
                 "标点是否写得太小。")
    return lines, chars


def calibrate(lib_path, k=3.0):
    """稳健离群校准：按字类分组，用中位数 + MAD 定正常范围。与具体人、具体字无关。"""
    man = load_manifest(lib_path)
    per_ch = {}
    for ch, items in man.get("chars", {}).items():
        best = None
        for rel in items:
            p = os.path.join(lib_path, rel)
            if not os.path.exists(p):
                continue
            a = load_alpha(p)
            m = metrics_of(a)
            if m and (best is None or (m["sharp"] + m["fill"]) > (best["sharp"] + best["fill"])):
                best = m
        if best:
            per_ch[ch] = best
    if not per_ch:
        return None
    groups = {}
    for ch, m in per_ch.items():
        groups.setdefault(script_class(ch), []).append(m)
    if len(groups.get("cjk", [])) < 8:
        groups = {"all": list(per_ch.values())}
        per_ch_class = {ch: "all" for ch in per_ch}
    else:
        per_ch_class = {ch: script_class(ch) for ch in per_ch}
    stats = {}
    for g, ms in groups.items():
        st = {}
        for key in METRIC_KEYS:
            vals = np.array([m[key] for m in ms], dtype=np.float32)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            st[key] = [med, max(mad * 1.4826, METRIC_FLOOR[key])]
        stats[g] = st
    out = {"k": k, "groups": stats, "class": per_ch_class, "chars": per_ch}
    with open(os.path.join(lib_path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def normalize_ink(a, floor=0.28, span=0.42):
    """墨迹浓度归一：先把墨迹 P95 拉到 1.0，再用对比曲线压掉中间灰带。

    目的：让库内所有实例的"发虚度"落在同一水平（老库 0.05 一档），
    否则新并入的字会明显发浅，质检会把它们全判成离群。
    """
    m = a > 0.2
    if m.sum() < 12:
        return a
    ref = float(np.percentile(a[m], 95))
    if ref <= 0.05:
        return a
    a = np.clip(a / ref, 0, 1)
    return np.clip((a - floor) / span, 0, 1)


def split_bleed(a, gapmin=2, gapmax=12, minor=0.42):
    """切掉贴着边缘的邻字墨块。

    判据：字形主体与边缘墨块之间通常有一条 2–12px 的竖向空白，
    且被切掉的一侧墨量占少数（< 42%）。这样既治"邻字侵入"，
    又不会误伤 儿、八、川 这类本来就左右分离的字。
    """
    m = a > 0.5
    if m.sum() < 40:
        return a, 0
    cols = m.any(axis=0)
    idx = np.where(cols)[0]
    if len(idx) < 3:
        return a, 0
    x0, x1 = int(idx[0]), int(idx[-1])
    seg = cols[x0:x1 + 1]
    gaps, run, start = [], 0, 0
    for i, v in enumerate(seg):
        if not v:
            if run == 0:
                start = i
            run += 1
        else:
            if gapmin <= run <= gapmax:
                gaps.append((start, start + run - 1))
            run = 0
    if gapmin <= run <= gapmax:
        gaps.append((start, start + run - 1))
    for gs, ge in gaps:
        left_ink = int(m[:, x0:x0 + gs].sum())
        right_ink = int(m[:, x0 + ge + 1:x1 + 1].sum())
        tot = left_ink + right_ink
        if tot < 20:
            continue
        if left_ink < minor * tot:
            return a[:, x0 + ge + 1:x1 + 1], 1
        if right_ink < minor * tot:
            return a[:, x0:x0 + gs], 1
    return a, 0


def plaus(a, ref_ink=0, ref_h=0, ch=None):
    """字形可信度检查：返回问题描述，空串表示可信。

    ⛔ 宽高比闸门不能一刀切（原实现 asp<0.25 或 >2.6 就判"形状可疑"）：
    单笔画字的正常宽高比本来就在区间外 —— 实测 `一`（55×8，asp=6.9）、
    `丨`（8×55，asp=0.15）、`-`（30×6，asp=5.0）全被判可疑，
    也就是**用户写的这几个字永远进不了字库**，而它们恰恰是最常用的标点与笔画。

    修法分两步：
      1. "残片"与"比例"两件事分开判 —— 残片看**绝对墨量**（小于 40 像素的墨点
         无论什么形状都不可信），比例看**与同类的关系**；
      2. 比例判据改成**离群判定**：只有在"同类的字都很方、就它极度细长"时才算异常。
         单笔画字自身就是一个合法的细长形状，不该因为"细长"被否。

    ch 给出字符时用得上这条先验：单笔画/标点类字符（一丨丨丶亅乙…-—_()）的
    宽高比本就极端，对它们**直接跳过比例闸门**，只保留残片判据。
    不传 ch 时退回"混合判据"：仍然拦极端的（asp<0.06 或 >16），但把阈值放宽到
    只拦真正的噪点（一条 1 像素线、一个孤立小黑块）。
    """
    m = a > 0.5
    n = int(m.sum())
    if n < 40:
        return "残片"
    h, w = a.shape
    asp = w / float(max(1, h))
    if ch is not None and is_single_stroke(ch):
        # 单笔画字：比例天然极端，只看残片与相对墨量
        if ref_ink and n < 0.45 * ref_ink:
            return "墨量偏少"
        return ""
    # 非单笔画字：比例闸门放宽到只拦噪点级别。
    # 原来的 0.25 / 2.6 会把"写得很瘦长的门、日"一起误杀，
    # 而真正需要拦的"一条横线碎片"（asp 常见 20+）在新阈值下依然被拦。
    if asp < 0.08 or asp > 12.0:
        return "形状可疑"
    if ref_ink and n < 0.45 * ref_ink:
        return "墨量偏少"
    if ref_h and h < 0.5 * ref_h:
        return "尺寸偏小"
    return ""


# 单笔画/极端比例的字符：这些字的"正常形状"就很细长，不能用方块的宽高比去衡量。
# 依据：手写 `一` 实测 55×8（asp 6.9）、`-` 30×6（asp 5.0）、`丨` 8×55（asp 0.15）。
#
# ⚠ 这里**必须**显式分段 + 转义，不要写成一大串混合引号。原来的写法是
#     set("…；：""''")
# 作者的意图是加全角引号 “ ”，实际写成了两个 ASCII 双引号 —— 靠 Python 的
# 相邻字符串字面量拼接才没报 SyntaxError，代价是 “” 根本没进集合
# （实测 38 个字符，应为 40），而且任何人再往里补一个 ASCII 双引号都会当场炸。
SINGLE_STROKE_CHARS = set(
    "一丨丶丿亅乀乁乙"                    # 单笔画汉字
    "—–-_~"                               # 各类横线、波浪、连字符
    "|/\\()[]{}<>"                        # 竖线与各种括弧
    "！lI1'" + '"' +                       # 全角叹号、窄拉丁字母、两种 ASCII 引号
    "·．.,、。；：；“”"                     # 标点（含全角引号）
)


def is_single_stroke(ch):
    """这个字符的正常形状是否本来就很细长（不该用宽高比闸门卡它）。"""
    if not ch or len(ch) != 1:
        return False
    if ch in SINGLE_STROKE_CHARS:
        return True
    # 拉丁字母里的窄字形也常常是细长的一条（l、i、j、!），归到同一类
    return ch in "lIij!1"


# 底部印刷格线的判据阈值。
#
# 为什么需要这条（2026-09-17 用户报）：用户看到"每个 A 下面都有一根小横线"。
# 根因是采样单纸面上有印刷横线，抠字时连同字形一起抠了进来。
#
# ⛔ 已有的"按连通域删横线"（见 clean 里那段）**抓不到这个 case**：
# 它要求横线是一个**孤立**连通域；而 A 是手写体的尖角字形，
# 那条印刷线跟 A 的右下角**粘连**在一起，整字只有一个连通域 ——
# 于是 find_objects 拿到的框是整个字，高度不满足"≤4 行"，永远删不掉。
#
# 所以这里改用**不依赖连通性**的按行判据。
# 核心量是 **溢出量 = 命中行的横向跨度 − 字形主体的横向跨度**：
_CUT_RULE_THR = 0.35         # 墨迹阈值
_CUT_RULE_FILL = 0.80        # 命中行墨点须铺满其跨度的 80%（排除零星几点）
_CUT_RULE_MAXROWS = 4        # 横线厚度上限（行）
_CUT_RULE_OVERFLOW = 6.0     # 溢出量下限（px）


def _cut_bottom_rule(a, dry=False):
    """抹掉字形底部那条**与主体粘连**的印刷格线。返回抹掉的行数（0 = 没动）。

    ⛔ 判据为什么用"溢出量"而不是"占字宽比"（2026-09-17 实测踩过）：
    一开始用"命中行宽度 ≥ 字宽 85%"，结果 8 个候选字**全部 100% 命中** ——
    因为印刷线一直铺到字格边缘，占字宽比恒为 100%，这个判据**没有区分力**。
    真正的区分量是"**比字形主体宽出多少**"：
      · 印刷线：比主体宽 9~11px（实测 A/与/个/时，整齐得不像手写）
      · 手写底横：溢出量 ≈ 0（实测 `得` 溢出 0、`联` 溢出 5，都被正确放过）
    手写笔画不可能恰好比字的主体宽出固定的一截、且恰好贴到格边。
    """
    m = np.asarray(a) > _CUT_RULE_THR
    if not m.any():
        return 0
    rows = m.sum(axis=1)
    wmax = int(rows.max())
    if wmax <= 4:
        return 0
    # ① 跳过尾部的空白行（裁边常留 1~2 行空白）。
    #    ⛔ 不跳这一步就永远扫不到横线：从底往上第一行是全 0，直接 break。
    y = len(rows) - 1
    while y >= 0 and rows[y] == 0:
        y -= 1
    if y < 0:
        return 0
    # ② 从最后一个有墨的行往上，收集连续的"满行"
    full = []
    while y >= 0 and rows[y] >= 0.85 * wmax:
        full.append(y)
        y -= 1
    if not full or len(full) > _CUT_RULE_MAXROWS:
        return 0
    y0, y1 = min(full), max(full) + 1
    sub = m[y0:y1]
    cols = np.where(sub.any(axis=0))[0]
    if cols.size == 0:
        return 0
    span = int(cols.max() - cols.min() + 1)
    # ③ 命中行内部必须"实心"（不是只有零星几点，否则可能是笔锋扫过）
    fill_ratio = sub.sum() / float(max(1, span * (y1 - y0)))
    if fill_ratio < _CUT_RULE_FILL:
        return 0
    # ④ 溢出量：命中行是否比字形主体还宽
    rest = m.copy()
    rest[y0:y1] = False
    rcols = np.where(rest.any(axis=0))[0]
    if rcols.size == 0:
        return 0
    rspan = int(rcols.max() - rcols.min() + 1)
    if span - rspan < _CUT_RULE_OVERFLOW:
        return 0
    if dry:
        return y1 - y0
    arr = np.asarray(a)
    if arr.ndim == 3:
        arr[y0:y1, :, 3] = 0
    else:
        arr[y0:y1, :] = 0.0
    return y1 - y0


def clean(lib_path, dry=False):
    """对既有字库做统一清理：去格线、切邻字、裁边、剔残片。

    残片判据用**全库同类中位墨量**做基准（比"该字自身最大墨量"稳健：
    否则带邻字墨块的坏实例会把基准抬高，真残片反而躲过）。
    """
    man = load_manifest(lib_path)
    fixed = trimmed = split = dropped = 0
    dropped_list = []
    prepared = {}
    inks = {"cjk": [], "other": []}
    for ch, items in man["chars"].items():
        rows = []
        for rel in items:
            p = safe_rel(lib_path, rel)
            if p is None or not os.path.exists(p):
                continue
            a = load_alpha(p)
            m = a > 0.5
            if m.sum() > 8 and a.shape[1] > 12:
                lb2, n2 = ndimage.label(m, structure=np.ones((3, 3)))
                for i2, sl2 in enumerate(ndimage.find_objects(lb2), start=1):
                    if sl2 is None:
                        continue
                    hh2 = sl2[0].stop - sl2[0].start
                    ww2 = sl2[1].stop - sl2[1].start
                    if ww2 >= 0.80 * a.shape[1] and hh2 <= 4:
                        a[sl2][lb2[sl2] == i2] = 0.0
                        fixed += 1
                # ⛔ 上面那段只能删**孤立**的横线（靠连通域）。
                # 2026-09-17 实测踩到它的盲区：`A` 底部那条印刷线跟两条腿**粘连**，
                # 整个字只有一个连通域 → find_objects 抓到的框是"整个字"，
                # hh2 等于字高、不满足 <=4 → 永远删不掉，用户就看见
                # "每个 A 底下都多一根小横线"。
                # 补一条**按行**的判据（不依赖连通性）：
                #   底部有 1~4 行，其横向跨度**比字形主体的跨度还宽**（溢出 ≥6px）、
                #   且墨点铺满该行跨度的 80% 以上 → 判为印刷格线，抹掉。
                # 为什么用"溢出量"而不是"占字宽比"：
                #   实测字格的横线会一直铺到格子边缘，比字的主体宽 9~11px；
                #   手写底横不会比主体宽出这么多（`得` 的溢出量是 0，被正确放过）。
                fixed += _cut_bottom_rule(a)
            a, sp = split_bleed(a)
            split += sp
            ink2 = a > 0.30
            if ink2.sum() > 8:
                ys2, xs2 = np.where(ink2)
                a = a[max(0, ys2.min() - 1):ys2.max() + 2, max(0, xs2.min() - 1):xs2.max() + 2]
                trimmed += 1
            n_ink = int((a > 0.5).sum())
            a = normalize_ink(a)
            fill = n_ink / float(max(1, a.shape[0] * a.shape[1]))
            rows.append((rel, a, n_ink, fill))
            inks["cjk" if not ch.isascii() else "other"].append(fill)
        prepared[ch] = rows
    ref = {}
    for k, v in inks.items():
        ref[k] = float(np.median(v)) if v else 0.0
    for ch, rows in prepared.items():
        base = ref["cjk" if not ch.isascii() else "other"]
        final = []
        for rel, a, n_ink, fill in rows:
            low_fill = bool(base) and fill < 0.5 * base and len(rows) > 1
            if low_fill:
                prob = "笔画过稀" if plaus(a) == "" else plaus(a)
                if not dry:
                    p = safe_rel(lib_path, rel)
                    if p and os.path.exists(p):
                        os.remove(p)
                dropped += 1
                dropped_list.append((ch, rel, prob))
                continue
            if not dry:
                rgba = np.dstack([np.full(a.shape + (3,), ink_rgb(), np.uint8), np.clip(a * 255, 0, 255).astype(np.uint8)])
                Image.fromarray(rgba, "RGBA").save(os.path.join(lib_path, rel))
            final.append(rel)
        man["chars"][ch] = final
    if not dry:
        write_json(os.path.join(lib_path, "manifest.json"), man)
        mp = os.path.join(lib_path, "metrics.json")
        if os.path.exists(mp):
            os.remove(mp)
    return {"lines": fixed, "trimmed": trimmed, "split": split, "dropped": dropped, "list": dropped_list, "ref": ref}


def lib_quality(lib_path, k=3.0, min_hits=2):
    """字库质检。返回 (报告文本, 需重写的字列表)。

    返回值里带上结构化清单，quality --form 直接用它生成补字单，
    不必再去解析自己打印的报告文本。
    """
    mp = os.path.join(lib_path, "metrics.json")
    # read_json 读不了会返回 None（这是它作为"容错读取"的设计）。所以这里必须把
    # "缓存不存在"和"缓存读坏了"都当作**需要重算**，而不是当成"字库为空"：
    # 原来的写法 `read_json(mp) if os.path.exists(mp) else calibrate(...)` 让一个
    # 损坏的 metrics.json 直接报出"字库为空。" —— 用户会以为字库丢了，
    # 实际只是质检缓存坏了，重算一次就好。
    data = read_json(mp) if os.path.exists(mp) else None
    if not data:
        data = calibrate(lib_path, k)
    if not data:
        return "字库为空。\n", []
    stats, cls, per_ch = data["groups"], data["class"], data["chars"]
    flags, sev = {}, {}
    for ch, m in per_ch.items():
        st = stats[cls[ch]]
        hits = []
        for key in METRIC_KEYS:
            med, sd = st[key]
            z = abs(m[key] - med) / sd
            if z > k:
                hits.append(key if key not in LOWER_BETTER or m[key] > med else key + "-低")
        if hits:
            flags[ch] = hits
            sev[ch] = sum(abs(m[key] - st[key][0]) / st[key][1] for key in METRIC_KEYS)
    rows = sorted(per_ch.items(), key=lambda kv: -sev.get(kv[0], 0.0))
    need = [c for c, _m in rows if len(flags.get(c, [])) >= min_hits or sev.get(c, 0) > 4.5]
    lines = ["字库质检：%d 个字符，需重写 %d（稳健离群校准 k=%.1f，按字类分组）"
             % (len(per_ch), len(need), data["k"])]
    for g in sorted(stats):
        lines.append("  %s 组基准： " % g + "  ".join("%s=%.3f±%.3f" % (key, stats[g][key][0], stats[g][key][1]) for key in METRIC_KEYS))
    for ch, m in rows[:12]:
        lines.append("  %s  碎片%.1f 锐度%.2f 发虚%.2f 高%dp  偏离%.1fσ  %s"
                     % (ch, m["frag"], m["sharp"], m["soft"], m["h"], sev.get(ch, 0.0), "/".join(flags.get(ch, [])) or "-"))
    if need:
        lines.append("建议照抄重写（偏离最大的排前）：%s" % "".join(need))
    else:
        lines.append("全部通过：没有离群字形，字库内部风格一致。")
    # 形状校验（宽高比）与上面的浓度/锐度指标互不重叠，单独跑一遍
    shp_lines, shp_bad = shape_lines(lib_path)
    lines += [""] + shp_lines
    # 补字单要同时收进两类问题：浓度/锐度离群 + 宽高比异常。
    # 原来只收前者，被压扁的字（如确、能）反而进不了补字单。
    need_all = list(need) + [c for c in shp_bad if c not in need]
    if need_all:
        lines += refill_notice(need_all, "有 %d 个字的字形不合格，需要重写。" % len(need_all))
    return "\n".join(lines) + "\n", need_all


def res_dir():
    """资源目录：源码运行时是脚本所在目录，打包成 exe 后是解包目录。"""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


# 内置纸面。名字 -> 文件名主干，实际扩展名在 .png/.jpg/.jpeg 里找第一个存在的。
# 仓库里放的是 JPEG（体积只有 PNG 的约 1/9），但用户自己换成 PNG 也能用。
PAPERS = {
    "white": "white_plain",
    "lined": "lined_note",
    "grid": "grid_paper",
    "redgrid": "red_grid",
    "cream": "cream_notebook",
}
_BG_EXTS = (".png", ".jpg", ".jpeg")


def is_file_paper(name):
    """--paper 传进来的到底是个"内置纸面名"还是"一个图片文件路径"。

    判据是**它能不能当路径打开**，不是"名字里有没有点/斜杠" ——
    用户可能用 `--paper 我拍的纸.png`（无路径）、也可能
    `--paper D:\\纸\\扫描.png`；反过来内置名里没有点，所以不会误判。
    扩展名限死在 _BG_EXTS 里：传 `--paper page.txt` 不该被当成背景图。
    """
    return (os.path.splitext(str(name))[1].lower() in _BG_EXTS
            and os.path.isfile(str(name)))


def bg_path(name):
    """按纸面名找背景图，返回路径；找不到返回 None。

    两种来源，先查内置、再当文件路径：
      1. **内置纸面**：backgrounds/<stem>.png 或 .jpg —— 让用户放哪种格式都能用；
      2. **自己导入的背景**：--paper 直接给一张图片的路径。

    ⚠ 为什么要支持第 2 种：内置 5 张纸面拍的都是"整幅干净纸"，而用户手上
    可能有更合心意的纸（比如自己拍的、带淡格线的）。但**背景图必须是纸面**
    —— 见 `paper_bg_notice()` 里的说明，图里若有桌面/手掌/其他物件，
    排版会把字写到纸面之外，成品就成了"字贴在桌子上"。
    """
    if is_file_paper(name):
        return str(name)
    stem = PAPERS.get(name)
    if stem is None:
        return None
    bdir = os.path.join(res_dir(), "backgrounds")
    for ext in _BG_EXTS:
        p = os.path.join(bdir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


# ⛔ 自定义背景的"进门须知"。用户 2026-09-18 明确要求：
#   「让他能够自己添加背景，但是必须说明添加的背景只能是纸面，不能带有其他东西，
#     不然就会出现'字体贴到桌子上'这种情况。你不用去添加导入检验，不要做这个东西，
#     你就把这个提示词设计的好一点，告诉他需要添加什么样的背景、
#     如果添加其他的可能会出现什么问题。」
# ⇒ **刻意不做自动校验**（不判纸面占比、不拦非纸面图）。理由：机器判"这是不是纸面"
#   必然有误判，而误判的代价是"用户明明有可用的图却不让用"。改成把话说透：
#   让人自己看图判断。下面是给用户看的原文，doctor 和报错提示共用同一份。
def paper_bg_notice():
    """自定义背景该用什么样的图、用错了会怎样。返回给用户看的行列表。

    ⚠ 这里列的两种后果都是**实测出来**的，不是推测：造一张"左侧纸、右侧木桌面"
    的图跑 render，成品里正文只占了左侧纸面那一块、右半边整页空着 ——
    纸面轮廓把桌面那部分正确地排除掉了，但**排版宽度也跟着窄了**，
    而且程序那句"没能量出纸面范围"的警告**不会出现**（轮廓是量出来了的），
    用户拿不到任何提示。所以只能在这里把话说在前头。
    """
    return [
        "自己导入背景（--paper 直接给一张图片的路径）：",
        "  handglyph.cmd render 内容.txt --paper \"D:\\我的纸.png\" -o out.png",
        "",
        "  ✅ 该用什么样的图：**整幅就是一张纸**，纸面铺满整个画面。",
        "     · 平铺正对拍/扫，别斜拍、别透视变形（斜的纸面轮廓会量偏，字会跳行）；",
        "     · 光线均匀，别一半亮一半暗、别留重阴影（阴影会被当成'非纸面'排除）；",
        "     · 长边建议 ≥1500px。太小会把字挤在一起，太大只是慢一点。",
        "     · 纸上有淡淡的格线/横线没关系（那正是纸面的一部分）。",
        "",
        "  ⛔ 不要用带桌面、手掌、书本边缘、笔、水杯等**纸面之外的东西**的照片。",
        "     排版是按'纸面轮廓'做的：程序先找出画面里的纸面区域，再往这块区域内摆字。",
        "     图里混进了纸外之物，那块会被判成非纸面、排除在可用范围外 —— 两种后果：",
        "",
        "       · **字只写在一角，大片空白**：可用宽度跟着变窄，正文挤在纸面那一块里，",
        "         右边的桌面整片空着。**程序不会为此报警**（轮廓量得出来，它以为一切正常），",
        "         你只会看到一张怪图。",
        "       · **字跑到纸外、贴到桌子上**：若纸面轮廓压根量不出来（画面太暗、纸面被",
        "         遮去大半），排版会退回一个保守宽度硬排，不再知道纸的边界在哪 ——",
        "         于是字一排排写到画面之外，成了'字贴在桌子上'。",
        "",
        "  💡 一句话判据：**把图打开，问自己'如果往这上面写字，每一处都能写吗'。**",
        "     只要有一处不是纸，就把它裁掉、或换一张 —— 程序不会替你拦，得你自己看。",
    ]



def default_lib():
    """字库默认位置：exe 旁边或当前目录下的 library。"""
    for base in (os.path.dirname(os.path.abspath(sys.argv[0])), os.getcwd()):
        p = os.path.join(base, "library")
        if os.path.exists(os.path.join(p, "manifest.json")):
            return p
    return os.path.join(os.getcwd(), "library")


def prune(lib_path, ratio=0.85, floor=0.35, dry=False):
    """淘汰差实例：不设数量上限，只删质量分明显低的。

    保留条件（双闸门，都必须满足）：
      1. 相对闸门：score >= ratio × 该字自身最优分   —— 删掉"同一个字里明显更差的那几版"
      2. 绝对闸门：score >= max(floor, 全库最优分分布的 q10) —— 删掉"整库范围内就烂的"
      3. 该字最优实例永远保留（保证没有字被清空）

    注：原签名里有个 k=3.0，函数体从未读它 —— 淘汰判据用的是 ratio/floor，
    不需要离群倍数（那个是 quality 的判据）。删掉以免 CLI 上多一个"填了也没用"的参数。
    """
    man = load_manifest(lib_path)
    per = {}
    for ch, items in man["chars"].items():
        rows = []
        for rel in items:
            p = safe_rel(lib_path, rel)
            if p is None or not os.path.exists(p):
                continue
            a = load_alpha(p)
            rows.append((score_alpha(a), rel))
        if rows:
            per[ch] = sorted(rows, key=lambda t: -t[0])
    if not per:
        return None
    bests = np.array([r[0][0] for r in per.values()], dtype=np.float32)
    q10 = float(np.quantile(bests, 0.10))
    hard = max(floor, q10)
    removed, stats = [], []
    for ch, rows in per.items():
        best = rows[0][0]
        thr = max(hard, ratio * best)
        keep = [rel for s, rel in rows if s >= thr] or [rows[0][1]]
        for s, rel in rows:
            if rel in keep:
                continue
            fp = safe_rel(lib_path, rel)     # 只删字库目录内的文件
            if not dry and fp and os.path.exists(fp):
                os.remove(fp)
            removed.append((ch, os.path.basename(rel), round(s, 3)))
        stats.append((ch, len(rows), len(keep), round(best, 3), round(thr, 3)))
        man["chars"][ch] = keep
    if not dry:
        with open(os.path.join(lib_path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
        invalidate_metrics(lib_path)
    return {"removed": removed, "stats": stats, "q10": q10, "floor": floor, "ratio": ratio, "dry": dry}


def _free_rel(dst, taken, limit=999):
    """给一个已被占用的目标路径找一个不冲突的替代：`00.png` → `01.png`、`02.png`…

    taken 是已占用的绝对路径集合（试算时目标还没落地，只能靠内存记账）。
    返回一个既不在 taken 里、磁盘上也不存在的路径；都满了就退到 `名_999.png`。
    """
    d = os.path.dirname(dst)
    base = os.path.splitext(os.path.basename(dst))[0]
    ext = os.path.splitext(dst)[1] or ".png"
    try:
        start = int(base)
    except ValueError:
        start = None
    if start is not None:
        for i in range(start + 1, limit + 1):
            cand = os.path.join(d, "%02d%s" % (i, ext))
            if cand not in taken and not os.path.exists(cand):
                return cand
    for i in range(limit + 1, limit + 10000):
        cand = os.path.join(d, "%s_%d%s" % (base, i, ext))
        if cand not in taken and not os.path.exists(cand):
            return cand
    return dst


def atlas_fix(lib_path, spec_path, dry=False):
    """改名逃生门：按一个"序号=字"的文本文件，重建 manifest 的字符映射。

    为什么需要：build 的自动命名只看"切出几个块、顺序对不对"，一旦切分或顺序
    出了问题，整批字的名字就全错位了 —— 而修法只有"重拍重跑"一条路，很折磨人。
    有了这个命令，用户对着 atlas.png 看一遍，把正确的对应关系写成一个文本文件
    （每行 `序号=字`），就能就地改名，把灾难降级成日常操作。

    文本格式（两种都认）：
        3=确
        7=能
    或一行一个连续区段：
        3-6=确能控制

    改名的实现是**搬目录**：字形按 ord(字) 建目录，改名就是换目录名。
    """
    man = load_manifest(lib_path)
    if not os.path.isfile(spec_path):
        print("找不到映射文件：%s" % spec_path)
        print("  请先跑一次 build 生成 atlas.png，人工核对切分序号，再按下面格式写：")
        print("    3=确")
        print("    7=能")
        print("  或：3-6=确能控制")
        return None
    mapping = {}
    with open(spec_path, encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "=" not in ln:
                print("跳过看不懂的行：%s" % ln)
                continue
            key, val = ln.split("=", 1)
            key, val = key.strip(), val.strip()
            if not val:
                continue
            if "-" in key:
                a, b = key.split("-", 1)
                try:
                    a, b = int(a), int(b)
                except ValueError:
                    print("跳过看不懂的序号范围：%s" % key)
                    continue
                for i, ch in enumerate(val):
                    mapping[a + i] = ch
            else:
                try:
                    mapping[int(key)] = val[0]
                except ValueError:
                    print("跳过看不懂的序号：%s" % key)
    if not mapping:
        print("映射文件里没有有效的 `序号=字` 行，未改动。")
        return None

    # 收集现有实例并确定它们的"序号"。
    # 优先用实例档案里的 seq（全局唯一，就是 atlas.png 上的编号）；
    # 老字库没有 instances 时才退回文件名 —— 但文件名在不同字符目录下会重名
    # （每个字的第一张都叫 00.png），这时序号的语义是"该字下的第几张"，
    # 只能用 `字符/序号` 形式的映射，所以下面的 mapping 支持两种键。
    inst_map = man.get("instances") or {}
    all_rel = []
    for ch, rels in man.get("chars", {}).items():
        all_rel.extend(rels)
    if not all_rel:
        print("字库里没有字形实例，无法改名。")
        return None

    # ⛔ 老字库（没有 instances）**只能**用 `字符/序号` 形式的映射，不能吃纯数字键。
    # 原因：没有 instances 时序号只能退回"文件名"，而文件名在每个字符目录下
    # 都从 00.png 开始 —— 纯数字键 `0=确` 会命中**所有**目录下的 00.png，
    # 把甲/00.png、乙/00.png、丙/00.png 一起改名成"确"。这是跨字符的批量误改，
    # 而用户以为自己只改了一个字。宁可报错让他改写映射，也不能默认接受。
    if not inst_map:
        numeric_keys = sorted(k for k in mapping if isinstance(k, int))
        if numeric_keys:
            print("这个字库缺少实例档案（manifest 里没有 instances），不能用纯数字序号改名。")
            print("")
            print("  原因：没有实例档案时，序号只能从文件名推 —— 而文件名在每个字符目录下")
            print("  都从 00.png 开始，都是 0。你的映射里有 %d 个纯数字键（如 %s=…），"
                  % (len(numeric_keys), numeric_keys[0]))
            print("  它会把每个字符目录下的那个文件**一起**改名，改错的不止一个。")
            print("")
            print("  请把映射键改成 `字符/序号` 形式，例如把 `0=确` 改成：")
            print("          甲/0=确")
            print("          乙/0=确")
            print("")
            print("  想确认每个文件属于哪个字，先跑 handglyph atlas --lib %s 看图鉴。" % lib_path)
            return None

    moved, missing = [], []
    new_chars = {}
    # 已确定的目标相对路径集合，用来在内存里避免撞名（试算时文件还没落地）
    _taken = set()
    for rel in all_rel:
        base = os.path.splitext(os.path.basename(rel))[0]
        rec = inst_map.get(rel) or {}
        idx = rec.get("seq")
        if idx is None:
            try:
                idx = int(base)
            except ValueError:
                continue
        d0 = os.path.dirname(os.path.join(lib_path, rel))
        if not os.path.isdir(d0):
            continue
        stem = os.path.basename(d0)
        try:
            cur = chr(int(stem)) if stem.isdigit() else stem
        except ValueError:
            cur = stem
        # 先按全局序号找，再按 `字符/序号` 找（兼容老字库）
        ch = mapping.get(idx)
        if ch is None:
            ch = mapping.get("%s/%s" % (cur, base))
        if ch is None:
            missing.append(rel)
            continue
        new_d = os.path.join(lib_path, "glyphs", str(ord(ch)) if len(ch) == 1 else ch)
        new_rel = os.path.relpath(os.path.join(new_d, base + ".png"), lib_path).replace("\\", "/")
        if cur == ch:
            new_chars.setdefault(ch, []).append(new_rel)
            continue
        if dry:
            # 试算时目标目录还没建，用内存里的占用表判断会不会撞名
            if new_rel in _taken:
                new_rel = _free_rel(new_rel, _taken)
            _taken.add(new_rel)
            moved.append((idx, cur, ch))
            new_chars.setdefault(ch, []).append(new_rel)
            continue
        os.makedirs(new_d, exist_ok=True)
        src = safe_rel(lib_path, rel)          # 只搬字库目录内的文件
        dst = os.path.join(lib_path, new_rel)
        if src is None or not os.path.isfile(src):
            missing.append(rel)
            continue
        if os.path.isfile(dst):
            # 目标已被占用：不放弃，换一个不冲突的编号 —— 否则改名会静默不落地。
            # （不同来源的同名字形在一个字下并存是正常的：多写几遍正是字库该有的样子。）
            dst = _free_rel(dst, _taken)
            new_rel = os.path.relpath(dst, lib_path).replace("\\", "/")
        os.replace(src, dst)
        _taken.add(new_rel)
        rec2 = inst_map.pop(rel, None)
        if rec2 is not None:
            inst_map[new_rel] = rec2
        moved.append((idx, cur, ch))
        new_chars.setdefault(ch, []).append(new_rel)

    if dry:
        print("试算：%d 个实例会被改名。" % len(moved))
        for idx, cur, ch in moved[:20]:
            print("  #%d  %s → %s" % (idx, cur, ch))
        return {"moved": len(moved), "missing": len(missing), "dry": True}

    man["chars"] = new_chars
    with open(os.path.join(lib_path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    invalidate_metrics(lib_path)
    print("已改名 %d 个实例，字库现有 %d 个字符。" % (len(moved), len(new_chars)))
    if missing:
        print("  有 %d 个实例在映射表里没有对应序号，保持原样。" % len(missing))
        print("  这些实例仍挂在旧字符下，可用 quality / atlas 再核对一次。")
    print("  建议接着跑：handglyph clean --lib %s && handglyph quality --lib %s" % (lib_path, lib_path))
    return {"moved": len(moved), "missing": len(missing), "dry": False}


def extract(image, out, lang="chi_sim+eng", backend="auto"):
    """从图片提取文本，生成页面描述骨架。本机 OCR 为可选后端。

    OCR 只是"偷懒输入"的便利功能，不是必需环节 —— 没有它也能用（手打文字即可）。
    所以这里刻意不把任何 OCR 库列为硬依赖，缺了就退回手填骨架。

    backend：
      auto      —— 有 pytesseract 就用（tesseract 需另装程序本体 + 语言包）
      tesseract —— 强制用 pytesseract
      paddle    —— 用 PaddleOCR（识别中文更准，但装起来是大工程：paddlepaddle 数百 MB）
    """
    lines = []
    if backend in ("auto", "paddle"):
        try:
            lines = _ocr_paddle(image)
        except ImportError:
            if backend == "paddle":
                lines = ["# 未安装 PaddleOCR（pip install paddleocr paddlepaddle）。",
                         "# 也可以改用 --backend tesseract，或直接把文字逐行填在下面。",
                         ""]
        except Exception as e:
            if backend == "paddle":
                lines = ["# PaddleOCR 运行失败（%s: %s）" % (type(e).__name__, str(e)[:80]),
                         "# 或直接把要写的文字逐行填在下面，然后删掉这两行注释。",
                         ""]
    if not lines and backend in ("auto", "tesseract"):
        try:
            import pytesseract
            txt = pytesseract.image_to_string(open_rgb(image), lang=lang)
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            if not lines:
                lines = ["# OCR 没识别出文字（图片可能太糊或没有文字）。",
                         "# 把要写的文字逐行填在下面，然后删掉这两行注释。",
                         ""]
        except ImportError:
            lines = ["# 未安装 OCR 组件，跳过自动提取（这不影响其他功能）。",
                     "# 如需自动提取：pip install pytesseract，并另装 tesseract-ocr 程序本体；",
                     "# 中文还需额外装 chi_sim 语言包，否则识别不出汉字。",
                     "# 或者直接把要写的文字逐行填在下面，然后删掉这几行注释。",
                     ""]
        except Exception as e:
            lines = ["# 未能自动提取文本（%s: %s）" % (type(e).__name__, str(e)[:80]),
                     "# 常见原因：未安装 tesseract 程序本体，或缺 chi_sim 中文语言包。",
                     "# 或直接把要写的文字逐行填在下面，然后删掉这几行注释。",
                     ""]
    if not lines:
        lines = ["# 未选择可用的 OCR 后端。",
                 "# 可选：--backend tesseract / --backend paddle，或直接把文字填在下面。",
                 ""]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out, len([x for x in lines if not x.startswith("#")])


def _ocr_paddle(image):
    """PaddleOCR 后端（可选）。装在 `pip install handglyph[ocr-pp]` 之后可用。"""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    res = ocr.ocr(np.asarray(open_rgb(image)), cls=True)
    out = []
    for page in (res or []):
        for item in (page or []):
            try:
                txt = item[1][0]
            except Exception:
                continue
            if txt and txt.strip():
                out.append(txt.strip())
    return out


def doctor():
    """环境自检：把"为什么在我电脑上跑不起来"一次问清楚。

    检查项：Python 版本、三个核心依赖、中文字体、背景图是否齐、OCR 是否可用。
    """
    lines = []
    ok = "  [ok]"
    no = "  [!!]"
    lines.append("handglyph 环境自检")
    lines.append("")
    lines.append("Python 版本：%s" % sys.version.split()[0])
    if sys.version_info < (3, 9):
        lines.append("  [!!] 版本过低，需要 3.9 及以上。去 https://www.python.org/downloads/ 装新版。")
    else:
        lines.append(ok + " 满足要求（>=3.9）")
    lines.append("")

    lines.append("核心组件：")
    for name, imp in (("pillow", "PIL"), ("numpy", "numpy"), ("scipy", "scipy")):
        try:
            mod = __import__(imp)
            ver = getattr(mod, "__version__", "?")
            lines.append(ok + " %-8s %s" % (name, ver))
        except ImportError:
            lines.append(no + " %-8s 未安装 —— 执行：pip install %s" % (name, name))
    lines.append("")
    lines.append("可选组件：")
    try:
        import pytesseract  # noqa: F401
        try:
            v = pytesseract.get_tesseract_version()
            lines.append(ok + " tesseract %s（extract 可用）" % v)
        except Exception:
            lines.append(no + " 装了 pytesseract 但找不到 tesseract 程序本体。")
            lines.append("      extract 会跳过；要自动提取请另装 tesseract-ocr。")
    except ImportError:
        lines.append("  [--] pytesseract 未安装（不影响使用，只影响 extract 自动提取）")
    lines.append("")

    lines.append("中文字体：")
    fp = cjk_font_path()
    if fp:
        lines.append(ok + " %s" % fp)
    else:
        lines.append(no + " 没找到中文字体。成品笔迹不受影响，但补字单上的说明文字会显示不出来。")
        lines.append("      解决：装个中文字体，或设置环境变量 HANDGLYPH_FONT 指向字体文件。")
    lines.append("")

    lines.append("背景纸面：")
    bdir = os.path.join(res_dir(), "backgrounds")
    if os.path.isdir(bdir):
        for name, stem in PAPERS.items():
            p = bg_path(name)
            if p:
                lines.append(ok + " %-8s %s" % (name, os.path.basename(p)))
            else:
                lines.append(no + " %-8s 缺失（%s.png / .jpg 都找不到）" % (name, stem))
    else:
        lines.append(no + " 背景目录不存在：%s" % bdir)
        lines.append("      说明：背景图不放仓库里（体积大），需要自己放进去。")
    lines.append("")
    for ln in paper_bg_notice():
        lines.append("  " + ln)
    lines.append("")
    lines.append("工作目录：%s" % os.getcwd())
    libp = "library"
    if os.path.isdir(libp):
        m = os.path.join(libp, "manifest.json")
        if os.path.isfile(m):
            try:
                man = read_json(m)
                lines.append(ok + " 已有字库：%d 个不同字符" % len(man.get("chars", {})))
                inst = man.get("instances") or {}
                if inst:
                    srcs = {}
                    for v in inst.values():
                        s = v.get("src", "?")
                        srcs[s] = srcs.get(s, 0) + 1
                    top = sorted(srcs.items(), key=lambda kv: -kv[1])[:3]
                    lines.append("      实例档案：%d 条，来源 %d 张字版（%s）"
                                 % (len(inst), len(srcs),
                                    "、".join("%s×%d" % (k, v) for k, v in top)))
            except Exception:
                lines.append(no + " 字库 manifest.json 读取失败")
        else:
            lines.append(no + " library 目录在，但缺 manifest.json")
    else:
        lines.append("  [--] 还没有字库（首次使用请先 build）")
    lines.append("")
    lines.append("使用提示：")
    lines.append("  · 手机拍的字版进程序前会自动按 EXIF 标记转正（竖拍照片不会再躺倒）。")
    lines.append("  · 补字单务必用 build <照片> --form-sheet --expect \"...\" --merge --replace 入库；")
    lines.append("    不带 --form-sheet 会把表单上的打印说明当成笔迹吃进字库。")
    return "\n".join(lines)


def selftest():
    """内置回归测试。返回 (结果行列表, 失败项数)。

    每个用例对应开发中真实踩过的坑。改阈值、改流程前先跑一遍，
    比手工渲染几页看图快得多，也能挡住旧问题复发。
    """
    import contextlib
    import hashlib
    import io as _io
    import math
    import re
    import shutil
    import tempfile

    res = []
    fails = [0]

    def check(name, ok, detail=""):
        if not ok:
            fails[0] += 1
        res.append("  [%s] %s%s" % ("通过" if ok else "失败", name,
                                    ("   " + detail) if detail else ""))

    @contextlib.contextmanager
    def quiet():
        """静音被测函数的正常输出，只留测试结论。"""
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield

    tmp = tempfile.mkdtemp(prefix="handglyph_selftest_")
    res.append("handglyph 自检（临时目录 %s）" % tmp)
    res.append("")

    # --- 1. vscore 向量化实现必须与逐像素实现数值等价 ---
    def ref_vscore(a, thr):
        b = a > 0.5
        tot = 0
        for col in b.T:
            run = 0
            for v in col:
                if v:
                    run += 1
                else:
                    if run >= thr:
                        tot += run
                    run = 0
            if run >= thr:
                tot += run
        return float(tot)

    rng = np.random.default_rng(1)
    same = True
    for _ in range(20):
        m = (rng.random((30, 20)) < rng.uniform(0.2, 0.9)).astype(np.float32)
        for thr in (2, 7, 15):
            if abs(ref_vscore(m, thr) - vscore(m, thr)) > 1e-6:
                same = False
    check("vscore 向量化与逐像素实现等价", same)

    # --- 1b. 模块级常量的定义位置必须早于"把它们当默认值用"的地方 ---
    #
    # 这条不是理论洁癖：本轮把散落的 108 收成 CELL 常量时，我把它放在
    # FORM_X0 旁边（约 1520 行），而 build() 的签名 `cell=CELL` 在 1252 行 ——
    # Python 的默认参数在 def 执行时求值，于是 import 阶段直接 NameError，
    # 整个程序一行都跑不起来。这种错只要有人再挪一次常量就可能复发，
    # 而它的表现是"程序完全无法启动"，值得一条固定断言。
    #
    # 判据：拆出模块级赋值的行号，扫所有函数签名（**含跨行的续行**）里引用的
    # 常量，定义行号必须更小。
    #
    # 踩过的坑：只扫 `^def ` 那一行会漏 —— build() 的签名就是跨行的，
    # `cell=CELL` 落在续行上，于是这条断言"通过"了，程序却 import 就崩。
    _self_src = open(os.path.abspath(__file__), encoding="utf-8").read().splitlines()
    _mod_names, _mod_line = set(), {}
    for _ln, _line in enumerate(_self_src, 1):
        _m = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=", _line)
        if _m:
            _mod_names.add(_m.group(1))
            _mod_line.setdefault(_m.group(1), _ln)
    _bad_order = []
    _i = 0
    while _i < len(_self_src):
        if re.match(r"^def\s+\w+", _self_src[_i]):
            _depth, _started, _j = 0, False, _i
            while _j < len(_self_src):
                _depth += _self_src[_j].count("(") - _self_src[_j].count(")")
                if "(" in _self_src[_j]:
                    _started = True
                if _started and _depth <= 0:
                    break
                _j += 1
            _sig = "\n".join(_self_src[_i:_j + 1])
            for _r in set(re.findall(r"=\s*([A-Z_][A-Z0-9_]*)\b", _sig)):
                if _r in _mod_names and _mod_line[_r] > _i + 1:
                    _bad_order.append("%s(行%d 引用) 定义在 行%d"
                                      % (_r, _i + 1, _mod_line[_r]))
            _i = _j + 1
        else:
            _i += 1
    check("模块常量定义早于函数默认值引用", not _bad_order,
          "；".join(_bad_order[:3]) if _bad_order else "（%d 个常量）" % len(_mod_names))

    # --- 2. 编号要跳过空洞，否则 prune 后 merge 会覆盖幸存的旧实例 ---
    idxdir = os.path.join(tmp, "idx")
    os.makedirs(idxdir)
    for n in ("00", "01", "02"):
        with open(os.path.join(idxdir, n + ".png"), "wb") as f:
            f.write(b"x")
    os.remove(os.path.join(idxdir, "00.png"))
    nxt = next_index(idxdir)
    check("编号跳过空洞（prune 后 merge 不覆盖旧实例）", nxt == 3, "得到 %d，应为 3" % nxt)

    # --- 3. 竖画倾角召回：合成已知倾角的竖线，看能否报出接近的角度 ---
    #
    # 判据按角度分档，因为小角度的**相对难度本来就更高**：
    # 一条 3° 的斜线在 70px 画布上只横向偏 2.6px，而 vscore 的角度分辨率
    # 就是由这个偏移量决定的。实测小角度档的误差稳定落在 3.0（恰好压线），
    # 余量为零 —— 这不是"判据太松"，而是这条断言的固有精度。
    # 分档比统一放宽更有信息量：大角度必须严格（它们是真正要扶正的情况）。
    ok_lean, lean_txt = True, []
    for deg in (3, -5, 8, -11):
        im = Image.new("L", (70, 70), 0)
        dr = ImageDraw.Draw(im)
        half, dx = 25.0, 25.0 * math.tan(math.radians(deg))
        dr.line([(35 - dx, 35 - half), (35 + dx, 35 + half)], fill=255, width=4)
        a = np.asarray(im).astype(np.float32) / 255.0
        o, _conf = lean_report(a)
        lean_txt.append("%d°→%.0f°" % (deg, o))
        # 小角度档（|deg| <= 5）容忍 3.5°；大角度档要求 2.0°。
        limit = 3.5 if abs(deg) <= 5 else 2.0
        if abs(abs(o) - abs(deg)) > limit:
            ok_lean = False
            lean_txt[-1] += "(超限 %s)" % limit
    check("竖画倾角召回（±3~11° 竖线：小角≤3.5°、大角≤2.0°）", ok_lean, "  ".join(lean_txt))

    # --- 4~7. 依赖中文字体：合成字版 -> build -> render 全链路 ---
    fpath = cjk_font_path()
    if not fpath:
        res.append("  [跳过] 本机没有中文字体，无法合成字版做全链路测试")
    else:
        chars = "今天天气不错"
        sheet = os.path.join(tmp, "sheet.png")
        im = Image.new("RGB", (1500, 800), (250, 250, 246))
        dr = ImageDraw.Draw(im)
        ft = ImageFont.truetype(fpath, 110)
        for i, ch in enumerate(chars):
            dr.text((70 + (i % 5) * 280, 160 + (i // 5) * 300), ch, font=ft, fill=(30, 30, 36))
        im.save(sheet)

        libdir = os.path.join(tmp, "lib")
        with quiet():
            r = build(sheet, libdir, expect=chars)
        check("build 切出字数与 --expect 一致", r["glyphs"] == len(chars),
              "切出 %d / 期望 %d" % (r["glyphs"], len(chars)))
        got = set(read_json(os.path.join(libdir, "manifest.json"))["chars"])
        check("build 命名与 --expect 一致", got == set(chars), "得到 %s" % "".join(sorted(got)))

        bgp = bg_path("white")
        if not bgp:
            res.append("  [跳过] 找不到白色纸面背景，跳过渲染测试")
        else:
            spec = os.path.join(tmp, "s.txt")
            with open(spec, "w", encoding="utf-8") as f:
                f.write("今天\n# 这是注释：今天今天\n不错\n")
            o1, o2 = os.path.join(tmp, "o1.png"), os.path.join(tmp, "o2.png")
            with quiet():
                render(spec, libdir, bgp, o1, seed=20260101)
                render(spec, libdir, bgp, o2, seed=20260101)
            h1 = hashlib.md5(open(o1, "rb").read()).hexdigest()
            h2 = hashlib.md5(open(o2, "rb").read()).hexdigest()
            check("同 seed 渲染两次逐字节一致", h1 == h2)

            atxt = open(os.path.splitext(o1)[0] + ".audit.txt", encoding="utf-8").read()
            mo = re.search(r"共 (\d+) 字", atxt)
            n = int(mo.group(1)) if mo else -1
            check("注释行不计入渲染（正文 4 字）", n == 4, "实际 %d 字" % n)

            lspec = os.path.join(tmp, "long.txt")
            with open(lspec, "w", encoding="utf-8") as f:
                f.write("\n".join(["今天天气"] * 60) + "\n")
            o3 = os.path.join(tmp, "o3.png")
            with quiet():
                render(lspec, libdir, bgp, o3)
            a3 = open(os.path.splitext(o3)[0] + ".audit.txt", encoding="utf-8").read()
            check("超页截断写进审计报告", "内容被截断" in a3)

            # --- 8. 字数对账：应渲染 == 实渲染，且写进审计 ---
            a1 = open(os.path.splitext(o1)[0] + ".audit.txt", encoding="utf-8").read()
            m = re.search(r"应渲染 (\d+) 字（其中程序画点 (\d+) 字），实渲染 (\d+) 字", a1)
            ok_acc = bool(m) and m.group(1) == m.group(3)
            check("字数对账行存在且应渲==实渲", ok_acc,
                  ("应%s/实%s" % (m.group(1), m.group(3))) if m else "未找到对账行")

            # --- 9. 折行：一行超长内容必须折成多行，而不是整体缩成一行 ---
            wspec = os.path.join(tmp, "wrap.txt")
            with open(wspec, "w", encoding="utf-8") as f:
                f.write("今天天气不错" * 12 + "\n")
            ow = os.path.join(tmp, "ow.png")
            with quiet():
                render(wspec, libdir, bgp, ow)
            gimg = np.asarray(Image.open(ow).convert("L"))
            rows_ink = np.where((gimg < 200).any(axis=1))[0]
            bands = 1
            for y0, y1 in zip(rows_ink[:-1], rows_ink[1:]):
                if y1 - y0 > 3:
                    bands += 1
            check("超长行折成多行（不再整行压缩）", bands >= 2, "墨带 %d 条" % bands)

            # --- 10. 缩进指令 ">" 不是要写的字 ---
            ispec = os.path.join(tmp, "ind.txt")
            with open(ispec, "w", encoding="utf-8") as f:
                f.write("> 今天\n")
            oi = os.path.join(tmp, "oi.png")
            with quiet():
                render(ispec, libdir, bgp, oi)
            ai = open(os.path.splitext(oi)[0] + ".audit.txt", encoding="utf-8").read()
            mi = re.search(r"共 (\d+) 字", ai)
            check("缩进指令 > 不计入字数", bool(mi) and int(mi.group(1)) == 2,
                  "实际 %s 字（应为 2）" % (mi.group(1) if mi else "?"))

    # --- 11. EXIF 方向：带旋转标记的图必须被转正 ---
    exifp = os.path.join(tmp, "exif.jpg")
    src = Image.new("RGB", (400, 200), (255, 255, 255))
    ImageDraw.Draw(src).rectangle([10, 10, 60, 180], fill=(0, 0, 0))
    ex = src.getexif()
    ex[274] = 6                     # Orientation=6：顺时针转 90°
    src.save(exifp, exif=ex)
    raw = Image.open(exifp).size
    fixed = open_rgb(exifp).size
    check("读图应用 EXIF 旋转标记", raw == (400, 200) and fixed == (200, 400),
          "原 %s → 转正后 %s" % (raw, fixed))

    # --- 12. BOM：Windows 记事本存的 UTF-8 不能把首行搞坏 ---
    bp = os.path.join(tmp, "bom.txt")
    with open(bp, "w", encoding="utf-8-sig") as f:
        f.write("* 标题\n正文\n")
    check("读取 BOM 文本时吃掉 \\ufeff", read_text(bp).startswith("*"),
          "首字符 %r" % read_text(bp)[0])

    # --- 13. 补字单：格子感知切分必须只认得手写字，不认打印内容 ---
    fp2 = cjk_font_path()
    if fp2:
        fchars = "与门天气"
        sheet2 = Image.new("RGB", (1654, 2339), (252, 252, 250))
        d2 = ImageDraw.Draw(sheet2)
        f30, f62, f76 = cjk_font(30), cjk_font(62), cjk_font(76)
        # 顶部说明文字：故意用老的深灰 70，考验墨深门槛而不是靠颜色躲
        d2.text((90, 90), "照抄下面每个字，写在对应格子里，写得好不好直接决定成图效果",
                font=f30, fill=(70, 70, 70))
        for k in range(len(fchars)):
            cx = FORM_X0 + (k % 12) * CELL
            cy = FORM_Y0 + (k // 12) * CELL
            d2.rectangle([cx, cy, cx + CELL, cy + CELL], outline=(150, 150, 150))
            bb = d2.textbbox((0, 0), fchars[k], font=f62)
            d2.text((cx + (CELL - (bb[2] - bb[0])) / 2 - bb[0],
                     cy + (CELL - (bb[3] - bb[1])) / 2 - bb[1]),
                    fchars[k], font=f62, fill=(175, 175, 175))
        # 只"手写"第 0、2 格
        for k in (0, 2):
            cx = FORM_X0 + (k % 12) * CELL
            cy = FORM_Y0 + (k // 12) * CELL
            bb = d2.textbbox((0, 0), fchars[k], font=f76)
            d2.text((cx + (CELL - (bb[2] - bb[0])) / 2 - bb[0],
                     cy + (CELL - (bb[3] - bb[1])) / 2 - bb[1]),
                    fchars[k], font=f76, fill=(30, 32, 36))
        fimg = os.path.join(tmp, "form.png")
        sheet2.save(fimg)

        flib = os.path.join(tmp, "flib")
        with quiet():
            build(fimg, flib, expect=fchars, form_sheet=True)
        got_f = set(read_json(os.path.join(flib, "manifest.json"))["chars"])
        check("补字单：只收手写字，打印内容不入库",
              got_f == {"与", "天"}, "得到 %s（应为 与天）" % ("".join(sorted(got_f)) or "空"))

        # 空白补字单：一个字没写 → 必须 0 入库，而不是吃进说明文字
        blank2 = Image.new("RGB", (1654, 2339), (252, 252, 250))
        db2 = ImageDraw.Draw(blank2)
        db2.text((90, 90), "照抄下面每个字，写在对应格子里，写得好不好直接决定成图效果",
                 font=f30, fill=(70, 70, 70))
        for k in range(len(fchars)):
            cx = FORM_X0 + (k % 12) * CELL
            cy = FORM_Y0 + (k // 12) * CELL
            db2.rectangle([cx, cy, cx + CELL, cy + CELL], outline=(150, 150, 150))
            bb = db2.textbbox((0, 0), fchars[k], font=f62)
            db2.text((cx + (CELL - (bb[2] - bb[0])) / 2 - bb[0],
                      cy + (CELL - (bb[3] - bb[1])) / 2 - bb[1]),
                     fchars[k], font=f62, fill=(175, 175, 175))
        bimg = os.path.join(tmp, "blank.png")
        blank2.save(bimg)
        blib = os.path.join(tmp, "blib")
        with quiet():
            rb = build(bimg, blib, expect=fchars, form_sheet=True)
        check("空白补字单：干净拒绝、不入库任何字形", rb["glyphs"] == 0,
              "入库 %d 个（应为 0）" % rb["glyphs"])

        # 周期合理性：说明文字行距本身也是强周期，自相关会把它当成格距
        # （实测 40~50 px）。若不否决，窗口会被压成 108x50 把字拦腰切断，
        # 而残片能通过后面所有检查被静默命名入库 —— 输出看着像"完全成功"。
        # 这里造一个三行说明文字的表单复现该场景。
        adv = Image.new("RGB", (1654, 2339), (252, 252, 250))
        da = ImageDraw.Draw(adv)
        for i, txt in enumerate([
            "照抄下面每个字，写在对应格子里，写得好不好直接决定成图效果",
            "每字写进格子、占七成大小；不要压在格线上",
            "写不好可以重写，重写时把字写足、不要写瘦",
        ]):
            da.text((90, 90 + i * 40), txt, font=f30, fill=(165, 165, 165))
        for k in range(len(fchars)):
            cx = FORM_X0 + (k % 12) * CELL
            cy = FORM_Y0 + (k // 12) * CELL
            da.rectangle([cx, cy, cx + CELL, cy + CELL], outline=(150, 150, 150))
            bb = da.textbbox((0, 0), fchars[k], font=f62)
            da.text((cx + (CELL - (bb[2] - bb[0])) / 2 - bb[0],
                     cy + (CELL - (bb[3] - bb[1])) / 2 - bb[1]),
                    fchars[k], font=f62, fill=(200, 200, 200))
        for k in (0, 1):
            cx = FORM_X0 + (k % 12) * CELL
            cy = FORM_Y0 + (k // 12) * CELL
            bb = da.textbbox((0, 0), fchars[k], font=f76)
            da.text((cx + (CELL - (bb[2] - bb[0])) / 2 - bb[0],
                     cy + (CELL - (bb[3] - bb[1])) / 2 - bb[1]),
                    fchars[k], font=f76, fill=(30, 32, 36))
        aimg = os.path.join(tmp, "adv.png")
        adv.save(aimg)
        ag = np.asarray(Image.open(aimg).convert("L")).astype(np.float32)
        acp, arp = detect_grid(ag, cell=CELL)
        p_lo, p_hi = CELL * GRID_PERIOD_LO, CELL * GRID_PERIOD_HI
        check("补字单：带外周期被否决（回退已知格宽）",
              all(p is None or p_lo <= p <= p_hi for p in (acp, arp)),
              "cp=%s rp=%s（合理带 %.0f~%.0f）" % (acp, arp, p_lo, p_hi))

        alib = os.path.join(tmp, "alib")
        with quiet():
            build(aimg, alib, expect=fchars, form_sheet=True)
        a_min = None
        for _rels in read_json(os.path.join(alib, "manifest.json"))["chars"].values():
            for _r in _rels:
                _a = np.asarray(Image.open(os.path.join(alib, _r)))
                _al = _a[..., 3].astype(np.float32) / 255.0
                _rows = np.where((_al > 0.5).sum(axis=1) > 0)[0]
                _h = int(_rows[-1] - _rows[0] + 1) if _rows.size else 0
                a_min = _h if a_min is None else min(a_min, _h)
        check("补字单：入库字形不是被切断的残片",
              a_min is not None and a_min >= FORM_FRAG_SPAN * CELL,
              "最矮字形实墨高 %s（下限 %.0f）" % (a_min, FORM_FRAG_SPAN * CELL))

        # 残片闸门：即便周期判错（强制喂一个 108x50 的矮窗口），扁到不足
        # 格高 35% 的墨迹也必须被丢弃，而不是静默入库。
        _orig_dg = detect_grid
        globals()["detect_grid"] = lambda *a, **k: (None, 50)
        try:
            flap = os.path.join(tmp, "fraglib")
            with quiet():
                rfl = build(aimg, flap, expect=fchars, form_sheet=True)
        finally:
            globals()["detect_grid"] = _orig_dg
        check("补字单：残片闸门拦住被切断的墨迹", rfl["glyphs"] == 0,
              "入库 %d 个（应为 0，全是残片）" % rfl["glyphs"])

    # atlas --fix：改名必须落地。目标字符目录下已有同名文件时不能静默跳过
    # （每个字的第一张都叫 00.png，改名时撞名是常态，不是异常）。
    t2 = os.path.join(tmp, "atlasfix")
    os.makedirs(t2, exist_ok=True)
    a_im = Image.new("RGB", (400, 300), "white")
    ad = ImageDraw.Draw(a_im)
    ad.text((60, 60), "甲", font=cjk_font(96), fill=(20, 20, 20))
    a_s = os.path.join(t2, "a.png")
    a_im.save(a_s)
    a_lib = os.path.join(t2, "lib")
    with quiet():
        build(a_s, a_lib, expect="甲")
    g_dir = os.path.join(a_lib, "glyphs")
    src_png = os.path.join(g_dir, str(ord("甲")), "00.png")
    dst_dir = os.path.join(g_dir, str(ord("乙")))
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.isfile(src_png):
        shutil.copy2(src_png, os.path.join(dst_dir, "00.png"))   # 制造撞名
    spec_p = os.path.join(t2, "m.txt")
    with open(spec_p, "w", encoding="utf-8") as fh:
        fh.write("0=乙\n")
    with quiet():
        atlas_fix(a_lib, spec_p)
    am = load_manifest(a_lib)
    a_chars = set(am.get("chars", {}))
    a_paths = [r for rs in am["chars"].values() for r in rs]
    a_ok = (a_chars == {"乙"}
            and all(os.path.isfile(os.path.join(a_lib, r)) for r in a_paths))
    check("atlas --fix 撞名也能改名（跳号命名）", a_ok,
          "字符集 %s" % "".join(sorted(a_chars)))

    # ================= 以下为"护栏"专项：每条对应一个已修掉的静默失败 =================
    #
    # 共同点：这些错误原先**都不报错**，只是悄悄产出坏结果。所以断言不能只看
    # 退出码，必须同时检查"该拦的拦住了" + "拦的理由说出来了" + "旧数据还在"。

    # --- 14. form --kind symbols 不能再抛 NameError ---
    # 原先 SYMBOLS 常量根本不存在，这条命令必然崩。崩了很容易看出来，
    # 但它藏在 argparse 的分支里，没人跑到就没人发现 —— 所以固定测一条。
    ok_sym, sym_txt = True, ""
    try:
        with quiet():
            sp = make_form(list(SYMBOLS), os.path.join(tmp, "sym.png"))
        ok_sym = bool(sp) and os.path.isfile(sp)
        sym_txt = "%d 个字符" % len(SYMBOLS)
    except Exception as e:
        ok_sym, sym_txt = False, "%s: %s" % (type(e).__name__, e)
    check("form --kind symbols 不抛异常且产图", ok_sym, sym_txt)

    # --- 14b. form --copies 的格位顺序与 --expect 串一致 ---
    # 这条盯的是"两处各写一遍展开逻辑"这类错误：make_form 里怎么重复、
    # form_expect 里就该怎么重复，一旦不一致，入库时每个字都被安到错的名字上，
    # 而字数是对得上的 —— 全流程一声不吭。所以断言必须直接比对**顺序**，
    # 不能只比长度。顺带覆盖 copies=1 时不能把字重复（这是最容易被写坏的一档）。
    fx_ok = form_expect("甲乙", 1) == "甲乙" and form_expect("甲乙", 3) == "甲甲甲乙乙乙"
    fx_ok = fx_ok and form_expect("甲乙", 0) == "甲乙"   # 0 与负数按 1 处理
    fx_ok = fx_ok and form_expect("甲乙", -5) == "甲乙"
    pg_ok, pg_txt = True, "copies=1 单页"
    try:
        with quiet():
            p1 = make_form("甲乙", os.path.join(tmp, "cp1.png"), cols=4)
        with quiet():
            p3 = make_form("甲乙", os.path.join(tmp, "cp3.png"), cols=4, copies=3)
        pg_ok = bool(p1) and bool(p3) and os.path.isfile(p1) and os.path.isfile(p3)
        pg_txt = "copies=1 / copies=3 都出图"
    except Exception as e:
        pg_ok, pg_txt = False, "%s: %s" % (type(e).__name__, e)
    check("form --copies 的格位顺序与 --expect 串一致",
          fx_ok and pg_ok,
          "copies1=%r copies3=%r %s" % (form_expect("甲乙", 1),
                                        form_expect("甲乙", 3), pg_txt))

    # --- 14c. 自动排版：装箱不越界、且"能塞就塞" ---
    # 自动排版是本工具里唯一会"改变内容分布"的功能，装错了不会报错 ——
    # 只是内容被挪到别的页、或者一页塞进了放不下的行。所以两条都要钉住：
    #   ① 任何一页的累计高度不能超过页高（不然会溢出到画面外）
    #   ② 贪心：前一页在"再加一行就会超"之前不该提前收工
    it = [("a", 400), ("b", 400), ("c", 400), ("d", 400)]
    pg = plan_pages(it, None, None, 900, True)
    sum_ok = all(sum(it[k][1] for k in p) <= 900 for p in pg)
    flat = [k for p in pg for k in p]
    order_ok = flat == list(range(len(it)))          # 顺序不能乱、不能丢行
    greedy_ok = len(pg[0]) == 2                       # 400+400 放得下、再加就超
    check("自动排版：装箱不越界、顺序不乱、能塞就塞",
          sum_ok and order_ok and greedy_ok,
          "页数=%d 每页=%s 累计=%s"
          % (len(pg), [len(p) for p in pg],
             [sum(it[k][1] for k in p) for p in pg]))

    # --- 14d. 长行折行的纵向跨度要真的算进去 ---
    # measure_line_span 若不把折行累加，排版会以为"这行只占一行高"，
    # 于是把超高的行塞进页尾 —— 渲染时才发现放不下，行就落到画面外了。
    # 用 no_wrap=False 测：这时长行靠**折行**消化，跨度必须跟着涨。
    rows_f = [(y, 40, 1200) for y in range(0, 4000, 4)]
    bot_f = [(x, 3900) for x in range(0, 1300, 8)]
    fake_lib = {}
    for _c, _w in (("甲", 34), ("乙", 30), ("丙", 26)):
        _arr = np.zeros((40, _w), np.float32)
        _arr[4:36, 2:_w - 2] = 1.0
        fake_lib[_c] = [_arr]
    long_toks = _tokens("甲乙丙" * 40)
    one_toks = _tokens("甲")
    _n1, end1 = measure_line_span(long_toks, 40, rows_f, bot_f, 90, 60, 400,
                                  False, indent=0.0, lib=fake_lib)
    _n2, end2 = measure_line_span(one_toks, 40, rows_f, bot_f, 90, 60, 400,
                                  False, indent=0.0, lib=fake_lib)
    check("自动排版：量出的行高随内容增长（长行不会只算一行）",
          end1 > end2 + 100 and end2 <= 90 + 40 * 3,
          "短行结束于 %d，长行结束于 %d" % (end2, end1))

    # --- 14d2. 「量」与「画」必须同一把尺（跨模块口径一致性）---
    #
    # ⛔ 为什么必须跨模块比：原来 measure_line_span 用一张固定倍率表、
    # _render_line 用"实测字形宽 + 随机字距"，而**估算表里没有字距这一项** ——
    # 汉字实际推进 1.09~1.26h、估算只有 0.98h，量出来的行数系统性偏少
    # （每 5 行少算 1 行），分页偏乐观，最后靠"截断"收场。
    # 14c/14d 两条都是**自洽性检查**（拿喂进去的跨度反过来验装箱），
    # 测不出这个偏差 —— 必须让两边对同一行文字各算一遍再比。
    _cvs2 = np.zeros((200, 600, 3), np.float32)
    _rng2 = np.random.default_rng(11)
    _pk2 = Picker(fake_lib, rng=_rng2)
    _one_toks2 = list(_tokens("甲"))
    _cvs2, _rr2, _used2, _nch2, _xend2, _cut2 = _render_line(
        _cvs2, _one_toks2, fake_lib, 50.0, 90 + 40, 40, 400, _rng2, _pk2, 90,
        drift_amp=0.0, no_wrap=True)
    _adv_act = float(_xend2) - 50.0
    _wr2 = glyph_wh_ratio(fake_lib, "甲")
    _kind2, _tw_pred, _ = token_metrics("甲", 40, _wr2, False)
    _adv_lo = _tw_pred + 40 * 0.22 * 0.85      # 字距随机区间下界
    _adv_hi = _tw_pred + 40 * 0.22 * 1.2       # 上界
    # ⚠ 这里**故意不比对"折行行数"**：行数是取整量，每行差一个字就会在长文里
    # 累积成 1~2 行差异，拿它做判据会假失败。跨模块比对的正确对象是
    # **单字推进量** —— 它连续、可预测，而且是"同一把尺"的直接体现。
    check("自动排版：量行高与贴字同一把尺（单字推进量落在预测区间内）",
          _kind2 == "glyph" and _adv_lo - 1.5 <= _adv_act <= _adv_hi + 1.5,
          "实际推进 %.1f px，按 token_metrics 预测 %.1f~%.1f（字距随机区间）"
          % (_adv_act, _adv_lo, _adv_hi))

    # --- 14e. 空白段能在一张真成品上被找出来 ---
    # blank_bands 是"哪里有大片空白"的唯一判据，判反了自动排版就没有意义。
    # 拿一张"上半有墨、下半全空"的合成图验证：必须报出下半那一大段。
    blank_ok, blank_txt = False, ""
    try:
        bgw = Image.new("RGB", (400, 900), (250, 250, 250))
        arr = np.asarray(bgw).astype(np.float32)
        arr[100:300, 60:340] = 25.0          # 上半造一片墨
        rows_b = [(y, 40, 380) for y in range(0, 900, 4)]
        bot_b = [(x, 880) for x in range(0, 400, 8)]
        bd = blank_bands(arr.astype(np.uint8), rows_b, bot_b, min_h=200)
        # 期望：墨迹在 y=100..300，那 y>=300 的那一大段空白必须被报出来。
        tall = [b for b in bd if b["h"] >= 400 and b["y0"] >= 300]
        blank_ok = len(tall) >= 1
        blank_txt = ("找到 %d 段，最大 %dpx 起点 y=%d"
                     % (len(bd), tall[0]["h"] if tall else 0,
                        tall[0]["y0"] if tall else -1))
    except Exception as e:
        blank_txt = "%s: %s" % (type(e).__name__, e)
    check("自动排版：成品里的成片空白能被定位", blank_ok, blank_txt)

    # --- 15. build 到已有字库、且未带 --merge/--rebuild → 必须拒绝 ---
    # 原先会整体覆盖 manifest.json：旧字形文件还在磁盘上，但映射没了，
    # 全部变成孤儿 —— 输出一行警告都没有，用户以为只是"又加了几个字"。
    if fpath:
        rlib = os.path.join(tmp, "refuse")
        with quiet():
            build(sheet, rlib, expect=chars)
        n_before = len(load_manifest(rlib).get("chars") or {})
        with quiet():
            r_ref = build(sheet, rlib, expect="别的字")
        n_after = len(load_manifest(rlib).get("chars") or {})
        check("build 未带 --merge 时拒绝覆盖旧字库",
              r_ref.get("refused") is True and n_after == n_before,
              "refused=%s 字数 %d→%d" % (r_ref.get("refused"), n_before, n_after))

        # --- 16. --rebuild 是明确意愿，必须放行 ---
        with quiet():
            r_re = build(sheet, rlib, expect="别的字", rebuild=True)
        check("build --rebuild 明确放行重建",
              not r_re.get("refused") and r_re["glyphs"] > 0,
              "入库 %d 个" % r_re["glyphs"])

        # --- 17. 并入/改名/清理之后 metrics.json 必须失效 ---
        # metrics 是全库分组的中位数与 MAD。不失效 → 新并入的字被旧基准判成
        # 离群，prune 按旧分布删掉好字。这条断言直接看文件在不在。
        mlib = os.path.join(tmp, "met")
        with quiet():
            build(sheet, mlib, expect=chars)
        with quiet():
            calibrate(mlib)                       # 生成metrics
        mpath = os.path.join(mlib, "metrics.json")
        had = os.path.isfile(mpath)
        with quiet():
            build(sheet, mlib, expect=chars, merge=True)
        gone = not os.path.isfile(mpath)
        check("并入字形后 metrics.json 已失效", had and gone,
              "校准后有=%s，merge 后仍在=%s" % (had, not gone))

    # --- 18. plaus() 必须放过单笔画字（一 / 丨 / - ）---
    # 原先统一按宽高比闸门（0.25~2.6）判，这三个字全被判"形状可疑"并拒收。
    # 一整类合法的字被静默排除，而它们恰恰是最常用的字。
    single_txt, ok_single = [], True
    for ch, hw in (("一", (8, 55)), ("丨", (55, 8)), ("-", (6, 30))):
        hh, ww = hw
        arr = np.zeros((hh, ww), np.float32)
        if hh > ww:
            arr[:, max(0, ww // 2 - 1):ww // 2 + 1] = 1.0
        else:
            arr[max(0, hh // 2 - 1):hh // 2 + 1, :] = 1.0
        why = plaus(arr, ch=ch)
        single_txt.append("%s→%s" % (ch, why or "可信"))
        if why:
            ok_single = False
    check("plaus() 放过单笔画字（一/丨/-）", ok_single, "  ".join(single_txt))

    # --- 19. 落点在画布外的字必须报"未渲染"，不能悄悄消失在 recs 里 ---
    # 原先 blend() 静默 return，调用侧以为画上了 → 字数对账还报"相符"。
    # 直接测 blend() 的返回值协议：完全越界必须 (False, 1.0)。
    cvs = np.zeros((60, 60, 3), np.float32)
    rect = np.ones((10, 10), np.float32)
    oob = [blend(cvs, rect, -5, 0), blend(cvs, rect, 0, -5),
           blend(cvs, rect, 60, 0), blend(cvs, rect, 0, 60)]
    ok_oob = all(v[0] is False and v[1] == 1.0 for v in oob)
    ok_in = blend(cvs, rect, 10, 10)
    check("blend() 越界返回未落下、界内返回已落下",
          ok_oob and ok_in[0] is True and ok_in[1] == 0.0,
          "越界 %s / 界内 %s" % ([v[0] for v in oob], ok_in))

    # --- 20. 老字库（无 instances）不接受纯数字序号改名 ---
    # 序号语义在"有实例档案"和"没有"两种字库里完全不同：前者是全局顺序，
    # 后者是"该字下的第几张"。纯数字键在老字库上会跨字符改名。
    old_lib = os.path.join(tmp, "oldlib")
    os.makedirs(os.path.join(old_lib, "glyphs"), exist_ok=True)
    for _ch in "甲乙":
        _d = os.path.join(old_lib, "glyphs", str(ord(_ch)))
        os.makedirs(_d, exist_ok=True)
        arr_g = np.zeros((40, 40, 4), np.uint8)
        arr_g[8:32, 8:32] = [40, 40, 46, 255]
        Image.fromarray(arr_g, "RGBA").save(os.path.join(_d, "00.png"))
    with open(os.path.join(old_lib, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"chars": {"甲": ["glyphs/%d/00.png" % ord("甲")],
                             "乙": ["glyphs/%d/00.png" % ord("乙")]}}, fh)
    old_map = os.path.join(tmp, "old.txt")
    with open(old_map, "w", encoding="utf-8") as fh:
        fh.write("0=丙\n")
    with quiet():
        r_old = atlas_fix(old_lib, old_map)
    old_chars = set(load_manifest(old_lib).get("chars") or {})
    check("老字库（无 instances）拒绝纯数字键改名",
          r_old is None and old_chars == {"甲", "乙"},
          "返回 %s，字符集 %s" % (r_old, "".join(sorted(old_chars))))

    # --- 21. 缺 --expect 的补字单必须退出码非 0 ---
    # 这是唯一一条必须走 CLI 的断言：闸门装在 build() 的入口校验里。
    # 走 library 层是测不到的 —— 而用户恰恰是从命令行进来的。
    #
    # 注意：这里必须自己造一张补字单，不能复用上面 fimg ——
    # 那个变量在 `if fp2:` 分支里，字体缺失时根本没定义。
    # 而且这条断言测的是"缺 --expect 时被拦"，被拦住了就不会去读图，
    # 所以喂一张空白图就够，不需要造真格子。
    noexp_sheet = os.path.join(tmp, "noexp.png")
    Image.new("RGB", (300, 200), "white").save(noexp_sheet)
    rc_nox = None
    try:
        with quiet():
            rc_nox = run(["build", noexp_sheet, "-o", os.path.join(tmp, "noxp"),
                          "--form-sheet"])
    except BaseException as e:                  # SystemExit 也算"拦住了"
        rc_nox = "%s" % type(e).__name__
    ok_nox = rc_nox is not None and rc_nox != 0
    check("补字单缺 --expect 退出码非 0", ok_nox, "退出码 %s" % rc_nox)

    # --- 21b. 补字单上一个字都没读到（本次零入库）必须退出码非 0 ---
    # 这是"静默失败"的第三种形态，也是上面 P0-4 那条护栏管不到的：
    # 护栏只在**旧库非空**时拒绝覆盖；而"空白补字单 + 空目标目录"会让
    # build 走 --form-sheet 分支，在"没有一个格子里找到手写墨迹"处提前返回，
    # 早期实现这里 rc 是 0，于是 `handglyph build ... && 下一步` 照常往下走。
    #
    # 判据是退出码：JSON 里的 empty 字段只是给调用方看的，命令行用户看的是 rc。
    # 这里必须给 --expect，否则会被上一条闸门先拦住，测不到本条的路径。
    empty_sheet = os.path.join(tmp, "empty_form.png")
    Image.new("RGB", (600, 400), "white").save(empty_sheet)
    rc_emp = None
    try:
        with quiet():
            rc_emp = run(["build", empty_sheet, "-o", os.path.join(tmp, "emplib"),
                          "--form-sheet", "--expect", "甲乙丙"])
    except BaseException as e:
        rc_emp = "%s" % type(e).__name__
    ok_emp = rc_emp is not None and rc_emp != 0
    check("空白补字单（本次零入库）退出码非 0", ok_emp, "退出码 %s" % rc_emp)

    # --- 21c. 墨量过低的字形不能直接贴版（会被放大成黑块） ---
    # 实测来源：字库里 `·` 的字形是 18×11、只有 47 个墨点像素的一片残笔，
    # 贴到 40px 高的版面上被横向拉到 65px → 成品上就是一坨实心黑。
    #
    # 这条测的是**阈值函数本身**，不走整页渲染：渲染是随机的（vary/抖动），
    # 用成品图反推"某个字有没有变黑"既不稳也说不清失败原因。
    # 判据取两端：明显过小的必须被判低墨量，正常字形必须不被误伤 ——
    # 只测一端的话，把阈值写成 inf 也能"通过"。
    tiny = np.zeros((11, 18), np.float32)
    tiny[4:6, 6:12] = 1.0                      # 47 个墨点，形状无关，只看数量
    big = np.zeros((55, 46), np.float32)
    big[5:50, 6:40] = 1.0
    n_tiny, n_big = _too_thin(tiny), _too_thin(big)
    ok_thin = (n_tiny == int((tiny > 0.5).sum()) and n_tiny < GLYPH_MIN_INK
               and n_big is None)
    check("墨量过低的字形被判出、正常字形不误伤",
          ok_thin, "残留字形墨点 %s（阈 %d）／正常字形 %s"
          % (n_tiny, GLYPH_MIN_INK, "放行" if n_big is None else n_big))
    # shape 报告必须能报出这类字形。原实现只调 squeeze_report，
    # 而 `·` 是符号、走不进汉字判据 —— 于是坏字形一路静默通过。
    rep_thin = thin_report("·", tiny)
    rep_norm = thin_report("A", big)
    check("shape 的墨量判据能报出符号类坏字形",
          bool(rep_thin) and rep_thin.get("ink") == n_tiny and rep_norm is None,
          "· → %s ／ A → %s" % (rep_thin, rep_norm))

    # --- 22. 画布重构不能改变墨迹密度 ---
    #
    # 这轮把渲染热路径从"每字整页 float32 往返"改成"整页只在开头结尾各转一次"。
    # 性能改动的最大风险是**笔迹变了**，而肉眼看几页图很难看出来。所以设一条
    # 直接量墨迹的断言：相对差超过 1% 就说明重构动了渲染语义。
    #
    # 判据是**新旧两条路径互比**（同字形、同落点、同背景），而不是拿实渲图和
    # 某个绝对阈值比 —— 后者会把"散文内容变了"误判成"渲染变了"。
    ref_canvas = np.asarray(Image.new("RGB", (600, 400), (250, 250, 246)),
                            dtype=np.float32)
    one = np.zeros((40, 40), np.float32)
    one[10:30, 12:28] = 1.0
    for _i in range(40):
        # 旧写法：每贴一个字就把整页 float32 → uint8 → Image → float32 搬两趟
        old_u8 = np.clip(ref_canvas, 0, 255).astype(np.uint8)
        back = np.asarray(Image.fromarray(old_u8), dtype=np.float32)
        blend(back, one, 40 + _i * 12, 60 + (_i % 5) * 12)
        ref_canvas = back
    # 新写法：直接在 float32 画布上落墨，全程不物化
    new_canvas = np.asarray(Image.new("RGB", (600, 400), (250, 250, 246)),
                            dtype=np.float32)
    for _i in range(40):
        blend(new_canvas, one, 40 + _i * 12, 60 + (_i % 5) * 12)

    d_old = float((ref_canvas.mean(axis=2) < 200).mean())
    d_new = float((new_canvas.mean(axis=2) < 200).mean())
    rel = abs(d_new - d_old) / max(1e-9, d_old)
    check("画布重构前后墨迹密度一致（相对差 <1%）", rel < 0.01,
          "旧 %.4f%% → 新 %.4f%%（相对差 %.3f%%）" % (d_old * 100, d_new * 100, rel * 100))

    # 再加一条：走完整 render() 也不能崩、且确实产出了墨迹。
    # 上面那条只量了 blend()，这条兜住"画布重构把 render 的整体流程搞坏"。
    if fpath and bg_path("white"):
        dlib = os.path.join(tmp, "dens")
        with quiet():
            build(sheet, dlib, expect=chars)
        dspec = os.path.join(tmp, "dens.txt")
        with open(dspec, "w", encoding="utf-8") as fh:
            fh.write("今天天气不错\n" * 3)
        dp = os.path.join(tmp, "dens.png")
        with quiet():
            render(dspec, dlib, bg_path("white"), dp, seed=7)
        dgray = np.asarray(Image.open(dp).convert("L"))
        dens = float((dgray < 200).mean())
        check("重构后 render 仍能产出墨迹", 0.0005 < dens < 0.5,
              "整页墨迹占比 %.3f%%" % (dens * 100))

    # --- 22b. 抖动幅度必须停在"微妙"档，不能悄悄涨回去 ---
    #
    # 为什么值得一条断言：这组数值没有客观正确值，只能靠"观感"判断，
    # 而观感判断很容易在下次改别的东西时被顺手调大 —— 而且调大之后
    # 图还是"能看"，不会有人立刻发现，只是字迹悄悄变得像描粗的艺术字。
    # 用户 2026-09-17 的原话：「你这个参数太明显了，就是这个粗细变化幅度太大了，
    # 你应该把它调得非常微妙微小」。
    #
    # 判据取**上限**而不是精确值：允许往下调（更微妙总是安全的），
    # 只拦"涨回去"。三个量分别是粗细整体增益、旋转上限、整体缩放抖动。
    ok_subtle = (STROKE_BIAS <= 0.08 and ROTATE <= 0.60
                 and STROKE_LO >= 0.84 and STROKE_HI >= 0.86)
    check("抖动幅度停在微妙档（粗细/旋转不被调大）", ok_subtle,
          "增量 %.2f 旋转 %.2f° 墨量 %.2f/%.2f"
          % (STROKE_BIAS, ROTATE, STROKE_LO, STROKE_HI))

    # 粗细扰动的**实测**强度。这里量的是"连写一整行"的墨量离散度，
    # 不是"两遍的差"—— 因为用户看到的正是"一页字里有的粗有的细"，
    # 那是**同一 seed 内的分布**，而两遍互比只能量出"整体偏浓还是偏淡"。
    # （2026-09-17 教训：一开始只测了两遍互比，结果真凶 elongate 的方向效应
    #   完全没被这条断言看见 —— 两遍平均下来差不多，但行内忽粗忽细。）
    # 上限 32% 是"粗看齐整、细看有自然起伏"的经验界；实测约 28%。
    if chars:
        # 用真字形、不用合成矩形：合成矩形的"软边"比例不真实，
        # 而重描那一步恰恰只作用在软边上，用矩形会量出偏小的值。
        g_lib = os.path.join(tmp, "strokelib")
        with quiet():
            build(sheet, g_lib, expect=chars)
        gsrc = None
        gl = load_library(g_lib)
        for ch0 in chars:
            if ch0 in gl and gl[ch0]:
                # load_library 已经把 alpha 归一化成 float32 了，这里不用再转。
                gsrc = np.asarray(gl[ch0][0], dtype=np.float32)
                break
        if gsrc is not None and gsrc.shape[0] >= 12:
            # 连写 40 次，量与量的离散度
            rng_s = np.random.default_rng(11)
            inks = np.array([float(vary(gsrc, rng_s).sum()) for _ in range(40)])
            spread = float((inks.max() - inks.min()) / inks.mean())
            check("粗细扰动实测低于 32%（行内不显胖瘦）", spread < 0.32,
                  "连写 40 次：均值 %.0f 标差 %.0f 极差 %.1f%%"
                  % (inks.mean(), inks.std(), spread * 100))

    # --- 22c. 墨量归一必须真的把"同字多实例"的浓淡拉齐 ---
    #
    # 为什么值得一条断言：这个功能**第一次实现是完全无效的（1.00x，毫无变化）**，
    # 而且不报错、图也照出。原因是我拿 `stroke_weight`（笔画粗细）当标尺，
    # 而 ink_normalize 调的是 alpha 深浅 —— 度量看不见自己的调整，
    # 于是"测出来没改善"和"实际也没改善"混在一起，分不清是功能没用还是度量错了。
    # 这条断言直接量**调整前后的墨量**，量的是功能真正改变的那个量。
    if chars:
        n_lib = os.path.join(tmp, "normlib")
        with quiet():
            build(sheet, n_lib, expect=chars)
        nl = load_library(n_lib)
        # 造两个墨量差很大的"实例"：把同一个字形整体减淡一半当第二实例
        probe = None
        for ch0 in chars:
            if ch0 in nl and nl[ch0] and nl[ch0][0].shape[0] >= 20:
                probe = nl[ch0][0]
                break
        if probe is not None and probe.sum() > 200:
            fake = [probe, np.clip(probe * 0.5, 0, 1)]
            tgt = page_ink_target({"X": fake}, {"X"})
            before = np.array([float(x.sum()) for x in fake])
            after = np.array([float(ink_normalize(x, tgt).sum()) for x in fake])
            cv0 = before.std() / before.mean()
            cv1 = after.std() / after.mean()
            check("墨量归一确实拉齐了同字多实例的浓淡", cv1 < cv0 * 0.75,
                  "变异系数 %.1f%% -> %.1f%%（标尺 %.0f）"
                  % (cv0 * 100, cv1 * 100, tgt))
            # 归一化绝不能改轮廓：非零像素数必须一字不差
            n_b = int((fake[1] > 0.0).sum())
            n_a = int((ink_normalize(fake[1], tgt) > 0.0).sum())
            check("墨量归一只改浓淡、不动轮廓",
                  n_a == n_b,
                  "改动轮廓像素 %d -> %d" % (n_b, n_a))

    # --- 22d. 笔画宽度归一已于 2026-09-18 整块删除，这里只留「删除的依据」 ---
    #
    # ⛔ 这一节原先有 5 条断言（方向正确、不动外框、贴版半径朝目标移动、目标等于
    #    现状时不动、标尺折算唯一）。**其中前三条是假绿**：它们用一根人造的大竖条
    #    （源半径 4.5）当样本，而真实字库里的笔画半径多在 1~2px。换成真样本后
    #    立刻暴露 —— 见常量区 INK_NORM_* 上方的长注释、以及
    #    gen_out/v07_in/_sn_check.txt 的实测数据。
    #
    #    删掉的是「修正」这件事本身，不是断言写错了。所以这里**不再断言它能修**，
    #    改成断言**「删除是彻底的」+「它本来为什么做不到」**这两件事实。
    #    这样将来有人想把这功能加回来，会先在这里被拦下。
    if chars:
        # ① 彻底删除：模块里不该再残留任何入口（函数名、常量名、CLI 参数）。
        _leftover = [n for n in ("stroke_normalize", "page_stroke_target",
                                 "STROKE_NORM_ON", "STROKE_NORM_GAIN",
                                 "STROKE_NORM_DELTA_CAP", "STROKE_NORM_THIN_R")
                     if n in globals()]
        check("笔画宽度归一已彻底移除（无残留函数/常量入口）",
              not _leftover,
              "残留：%s" % (", ".join(_leftover) if _leftover else "无"))

        # ② 为什么会删：细笔画「变细」这件事**物理上退无可退**。
        #    1px 竖条做一次 binary_erosion 就整条消失 —— 不是代码不努力，
        #    是已经只剩 1px 了。这条断言把"不可修"钉死，免得又有人不信邪重写一遍。
        _one = np.zeros((36, 9), np.float32)
        _one[1:35, 4:5] = 1.0                       # 1px 宽竖条
        _st = ndimage.generate_binary_structure(2, 1)
        _ero = ndimage.binary_erosion(_one > 0.5, structure=_st)
        check("笔画宽度归一不该复活：1px 笔画做一次腐蚀即整条消失",
              int(_ero.sum()) == 0 and int((_one > 0.5).sum()) == 34,
              "原墨 %d 点 → 腐蚀后 %d 点（退无可退，这就是删它的原因）"
              % (int((_one > 0.5).sum()), int(_ero.sum())))

        # ③ 量尺在细笔画上没有分辨率 —— 这是当初写出假断言的直接原因。
        #    2px 条与 3px 条的 stroke_radius **都是 1.0000**。所以「半径纹丝不动」
        #    里头有很大一部分是**量尺没变化**，不是墨没变化。用它做 <1px 的判断
        #    必然得出错误结论。留这条，是为了让度量本身的局限写在明面上。
        _b2 = np.zeros((40, 12), np.float32)
        _b2[1:39, 5:7] = 1.0                        # 2px 宽
        _b3 = np.zeros((40, 12), np.float32)
        _b3[1:39, 5:8] = 1.0                        # 3px 宽
        _r2, _r3 = stroke_radius(_b2), stroke_radius(_b3)
        check("stroke_radius 在 2px/3px 笔画上无分辨率（故不可用作细笔画判据）",
              abs(_r2 - _r3) < 1e-6,
              "2px 条量出 %.4f、3px 条量出 %.4f（同值 → 量尺看不出差别）"
              % (_r2, _r3))


    # --- 22e. 底部印刷格线必须被切掉（用户报"每个 A 下面都多一根小横线"）---
    #
    # 为什么值得一条断言：这是用户**看图直接抓到的**缺陷，而且老版 clean()
    # 的"按连通域删横线"对它**完全无效**（横线跟字形粘连时抓不到）——
    # 属于"功能看着有、实际不覆盖这个 case"的隐蔽漏洞。
    # 这条断言同时守两头：粘连横线要切掉、真手写底横不能误切。
    if chars:
        # ① 粘连横线：一个三角（模拟 A 的两条腿）+ 一条铺满宽度的底线，
        #    底线与三角右下角**故意重叠**，制造"单一连通域"这个坑。
        glued = np.zeros((60, 46), np.float32)
        for i in range(40):                       # 三角两腰
            glued[8 + i, max(0, 10 - i // 2):min(46, 11 + i // 2)] = 1.0
        glued[6:10, 10:36] = 1.0                  # 顶部横
        glued[56:58, :] = 1.0                     # 印刷线：铺满 46px 宽
        glued[40:56, 22:25] = 1.0                 # 把线跟主体连起来（制造粘连）
        n_before = int((glued > _CUT_RULE_THR).sum())
        rows_cut = _cut_bottom_rule(glued)
        n_after = int((glued > _CUT_RULE_THR).sum())
        check("底部粘连的印刷格线被切掉（老版连通域判据抓不到）",
              rows_cut > 0 and n_after < n_before,
              "切掉 %d 行，墨点 %d -> %d" % (rows_cut, n_before, n_after))

        # ② 真手写底横不能被误切：画一个"日"字（底横只占主体宽度，不溢出）
        real = np.zeros((60, 40), np.float32)
        real[8:12, 8:32] = 1.0                    # 顶横
        real[54:58, 8:32] = 1.0                   # 底横（宽度 = 主体宽，溢出 0）
        real[8:58, 8:11] = 1.0                    # 左竖
        real[8:58, 29:32] = 1.0                   # 右竖
        n_real_b = int((real > _CUT_RULE_THR).sum())
        cut_real = _cut_bottom_rule(real)
        n_real_a = int((real > _CUT_RULE_THR).sum())
        check("真手写底横不被误切（溢出量为 0 时放过）",
              cut_real == 0 and n_real_a == n_real_b,
              "切掉 %d 行，墨点 %d -> %d" % (cut_real, n_real_b, n_real_a))

    # --- 22f. 字库分批：批内齐、批间不齐要能被抓到（用户 2026-09-17 要求）---
    #
    # 为什么值得一条断言：用户补字是**分次写**的，不同次落笔轻重天然不同；
    # 如果系统对"新批比旧批粗了一圈"毫无知觉，用户就会得到一页
    # "一半字胖、一半字瘦"的成品，而且**没有任何提示**告诉他问题出在字库上。
    #
    # 这条断言守三件事：
    #   ① 量得准  —— 一批字形被人为加粗/减细后，stroke_med 必须真的跟着变；
    #   ② 抓得到  —— 偏离超过容差必须判"超限"，并给出重写清单；
    #   ③ 不误报  —— 批间一致时（含只差一点点的自然抖动）必须放过。
    # 只测 ② 的话，一个"永远返回超限"的实现也能通过 —— 那是误报，不是功能。
    def _fake_glyph(h, w_frac):
        """合成一个"笔画粗细可控"的字形：笔画占字高的 w_frac。

        ⚠ 宽度必须按**字高**取比例，不能按"字宽"。_stroke_of_alpha 折算的
        是"字高 40px 时这个字形的笔画半径"，所以 60px 高的块和 180px 高的块
        只有宽度同比（0.30 × 字高）才能算出同一个折算半径；
        如果都取固定像素宽（比如 18px），两者原始半径一样，折算后
        反而差 3 倍 —— 那条断言就变成了在测我的合成函数写错了。
        """
        g = np.zeros((h, h), np.float32)
        w = max(2, int(round(h * w_frac)))
        x0 = h // 2 - w // 2
        g[int(h * 0.10):int(h * 0.90), x0:x0 + w] = 1.0
        return g

    # 折算的前提是"笔画宽度与字高成比例"。60px 高的块半径约 8.5、
    # 180px 高的块约 27，折算到 40px 字高后都应落在 6 附近。
    r_ref = _stroke_of_alpha(_fake_glyph(60, 0.30))
    r_scaled_h = _stroke_of_alpha(_fake_glyph(180, 0.30))
    ok_ref = (r_ref is not None and r_scaled_h is not None
              and abs(r_ref - r_scaled_h) / max(r_ref, 1e-9) < 0.05)
    check("笔画半径按字高折算（大字形不被误判为粗）", ok_ref,
          "60px 字高折算 %.2f ／ 180px 字高折算 %.2f" % (r_ref or -1, r_scaled_h or -1))

    # 批间比对的核心算法：造"两批一样粗"和"第二批明显粗"两种情形。
    #
    # ⚠ 容差必须从 compare_batches 自己的判据来取（它用的是"相对基准的偏差"），
    # 不能拿 BATCH_STROKE_TOL 的直觉值去试。第一版我用 2.0 → 2.9 造"粗 45%"，
    # 结果算法报 18.4% 而不是 45%（基准取的是两批的**中位**，2.9 被算成
    # "相对 2.45 偏高 18.4%"）—— 断言差点因为我把口径想错而红。
    # 这里直接构造一个"远大于容差"的偏离：2.0 与 4.0 → 基准 3.0、偏差 ±33%。
    same = {"batches": [{"id": "b1", "stroke_med": 2.00},
                        {"id": "b2", "stroke_med": 2.02}]}
    differ = {"batches": [{"id": "b1", "stroke_med": 2.00},
                          {"id": "b2", "stroke_med": 4.00}]}
    drift_same = compare_batches(same)
    drift_diff = compare_batches(differ)
    # 基准取两批中位，所以"一样粗"时两批偏差都接近 0。
    d_same = max(abs(d) for _i, d, _m, _c in drift_same) if drift_same else 9.9
    d_diff = max(abs(d) for _i, d, _m, _c in drift_diff) if drift_diff else 0.0
    ok_same = bool(drift_same) and d_same < BATCH_STROKE_TOL
    ok_diff = bool(drift_diff) and d_diff > BATCH_STROKE_TOL
    check("批间粗细：一致时不报、明显不齐时报（且超限判据真的能触发）",
          ok_same and ok_diff,
          "一致 %.1f%%（放行）／ 不齐 %.1f%%（超限），容差 %.0f%%"
          % (d_same * 100, d_diff * 100, BATCH_STROKE_TOL * 100))

    # 单批 / 无分批信息时不能崩，也不能谎报"有差异"。
    one = compare_batches({"batches": [{"id": "b1", "stroke_med": 2.0}]})
    none = compare_batches({})
    check("批不足 2 个时不比对（老库无批次信息也不能崩）",
          one == [] and none == [],
          "单批 %d 组 ／ 无批次 %d 组" % (len(one), len(none)))

    # 端到端：真的走一遍 merge，验证 (a) 开新批次 (b) 实例带上 batch 字段
    # (c) 同一份样本并入两次，第二次的 median 必须与第一次同源、落在容差内。
    # 这一步是"文档里承诺的字段真的存在"的唯一保证 —— 只测纯函数的话，
    # _store_glyphs 里漏写 batches 也照样全绿。
    if chars and fpath:
        b1_lib = os.path.join(tmp, "batlib")
        with quiet():
            build(sheet, b1_lib)
        m1 = load_manifest(b1_lib)
        bs1 = m1.get("batches") or []
        # 第二次并入（--merge 必须开新批次，不能覆盖 b1）
        with quiet():
            build(sheet, b1_lib, expect=chars, merge=True)
        m2 = load_manifest(b1_lib)
        bs2 = m2.get("batches") or []
        ids = [b.get("id") for b in bs2]
        has_batch_field = all(isinstance(v, dict) and v.get("batch")
                              for v in (m2.get("instances") or {}).values())
        meds = [b.get("stroke_med") for b in bs2 if b.get("stroke_med")]
        # 同源并入：两批中位数应接近（容差放宽到 2 倍，只拦"完全量错"）
        ok_med = bool(meds) and max(meds) / max(min(meds), 1e-9) < 2.0
        check("merge 会开新批次、实例带批次号、同源的字粗细一致",
              len(bs1) == 1 and len(bs2) >= 2 and ids[0] == "b1"
              and len(set(ids)) == len(ids) and has_batch_field and ok_med,
              "批次 %s ／ 实例带号 %s ／ 中位 %.2f~%.2f"
              % (ids, "是" if has_batch_field else "否",
                 min(meds) if meds else -1, max(meds) if meds else -1))

        # rebuild 必须把批次清空重来（否则"推倒重建"会留下幽灵批次，
        # 下次比对时凭空多出一批不存在的字）。
        with quiet():
            build(sheet, b1_lib, rebuild=True)
        m3 = load_manifest(b1_lib)
        bs3 = m3.get("batches") or []
        check("rebuild 后批次清空重建（不留幽灵批次）",
              len(bs3) == 1 and bs3[0].get("id") == "b1",
              "重建后批次 %s" % [b.get("id") for b in bs3])

    # 报告文本层：超限时**必须**出现"重写"字样与重写方法，
    # 因为这是用户拿到后唯一能照着做的动作。只返回 bool 的话，
    # CLI 会打印一份"有问题但不说怎么办"的报告。
    lines_over, over = batch_report(os.path.join(tmp, "batlib") if chars else tmp,
                                    tol=0.01)
    joined = "\n".join(lines_over)
    check("批间超限的报告里给出重写清单与方法",
          isinstance(lines_over, list) and (over is False or "重写" in joined),
          "超限 %s ／ 含重写字样 %s" % (over, "是" if "重写" in joined else "否"))

    # --- 22g. 质检基准必须按字类分组（否则整块字母被误报成"糊"）---
    #
    # 为什么值得一条断言：这是**用户看不到的度量 bug**，症状只是报告里
    # 多出十几个"不合格"，图本身没问题。而"报告说有 22 处不合格"会直接
    # 误导用户去重写根本没坏的字。实测证据：同一页 407 字里，
    # 拉丁组 237 字有 17 字越过 dens 判据，标点组 163 字一个都没有 ——
    # 两组的差异不是字形质量，是 LATIN=0.95 让拉丁字母的渲染尺度不同。
    # 这条断言守两头：拉丁/符号/汉字三组各自建基准，且"混在一起"会被判出来。
    #
    # ⚠ 造数据要**模拟真实的类间差异**：拉丁组整体给更低的 sharp、更高的 dens，
    # 汉字组给正常值。然后验证：分组口径下全通过，单一基准口径下拉丁组被误报。
    # ⚠ 造数据要**贴实测**，而且要造出「组内离群」，不能造「整组平移」。
    #
    # 实测（同一页 407 字）：dens 全体中位 0.224，拉丁组中位 0.239，
    # 标点组中位 0.192 —— 三组的中位**本身就不同**（拉丁 > 汉字 > 标点）。
    # 误报的来源正是"三组中位不同，却共用一条基准线"：
    # 拉丁组里那些**略高于本组中位**的字，一旦越过"全体中位 × 1.75"就被错杀。
    #
    # 所以对照数据要能把这个机制复现出来：
    #   · 符号组压低（0.190）→ 把全体中位往下拉
    #   · 拉丁组正常水平设在 0.25（本组自比完全正常，跟标的却偏高）
    #   · 另设 2 个真正的组内离群（0.75）→ 分组后仍必须被抓到
    # 断言分两条：① 正常的拉丁字不再被误报 ② 组内真离群仍被抓到。
    # 只测 ① 会漏掉"分组把判据彻底废掉"；只测 ② 会漏掉"分组没生效"。
    def _mk(ch, dens, sharp):
        return {"ch": ch, "dens": dens, "sharp": sharp, "dark": 1.0}

    fake = ([_mk(c, 0.22, 0.62) for c in "今天天气不错我们"]        # 汉字 8 个
            # 拉丁 12 个：10 个在**本组内**完全正常（0.36），2 个是真离群（0.75）
            # ⚠ 0.36 这个值是按"会越过全体基准（0.333）、但落在拉丁组自己
            #   的容忍带内（1.75 × 拉丁组中位）"挑的 —— 这正是实测里
            #   那 17 个被错杀的拉丁字的处境。
            + [_mk(c, 0.36, 0.58) for c in "ABCDEFGHIJ"]
            + [_mk("K", 0.75, 0.58), _mk("L", 0.74, 0.58)]
            # 符号 40 个：压低到 0.19，把**全体中位**拉到 0.19 附近
            + [_mk("=+()→·-"[(i % 7)], 0.19 - 0.001 * (i % 3), 0.60)
               for i in range(40)])

    rep_g, _n = audit(fake)
    has_three = rep_g.count("组基准") == 3
    nbad_g = int(rep_g.rsplit("不合格 ", 1)[1].split()[0]) if "不合格 " in rep_g else -1
    # ① 只剩 K/L 两个组内真离群，10 个正常拉丁字不再被误报
    ok_group = has_three and nbad_g == 2 and "K" in rep_g and "L" in rep_g
    check("质检基准按字类分组（正常拉丁字不再被误报）",
          ok_group,
          "基准组数 %d ／ 不合格 %d 处（应为 2：K L）"
          % (rep_g.count("组基准"), nbad_g))
    # ② 组内离群必须仍然被抓到
    caught = ("K" in rep_g and "L" in rep_g)
    check("分组后组内真离群仍被抓到（判据没被废掉）", caught,
          "报告里出现异常字：%s" % ("K L" if caught else "无"))

    # 反向对照：**退回单一基准**时，正常拉丁字也会被误报 —— 说明分组确实在干活。
    ds_all = np.array([r["dens"] for r in fake])
    sh_all = np.array([r["sharp"] for r in fake])
    d1, s1 = float(np.median(ds_all)), float(np.median(sh_all))
    n_old = sum(1 for r in fake
                if r["dens"] > 1.75 * d1 or r["sharp"] < 0.55 * s1)
    check("（对照）单一基准会把正常拉丁字一起误报——证明分组不是摆设",
          n_old > nbad_g,
          "旧口径误报 %d 处 > 新口径 %d 处（全体中位 %.3f，判据 %.3f）"
          % (n_old, nbad_g, d1, 1.75 * d1))

    # `_audit_class` 的分类边界：`⊕` `→` `·` 这类非 ASCII 符号
    # 必须归到 punct，不能按 isascii() 混进汉字组当基准。
    cls = {c: _audit_class(c) for c in "天A1=⊕→·（）"}
    ok_cls = (cls["天"] == "cjk" and cls["A"] == "latin" and cls["1"] == "latin"
              and cls["="] == "punct" and cls["⊕"] == "punct"
              and cls["→"] == "punct" and cls["·"] == "punct")
    check("字类边界：非 ASCII 符号归符号组、不混进汉字组", ok_cls,
          " ".join("%s=%s" % (k, v) for k, v in cls.items()))

    # 自定义背景：--paper 既能是内置名、也能是图片路径，两者不能互相误判。
    # 只测一侧的话，"任何输入都当文件路径"或"任何输入都查内置"都能过。
    real_img = os.path.join(tmp, "my_paper.png")
    Image.new("RGB", (300, 400), (250, 249, 245)).save(real_img)
    ok_bg = (
        bg_path("white") is not None                      # 内置名仍走 backgrounds/
        and bg_path(real_img) == real_img                 # 真图片路径原样返回
        and bg_path("nosuchpaper") is None                # 未知名回 None，不抛异常
        and not is_file_paper("white")                    # 内置名不被当路径
        and is_file_paper(real_img)                       # 真图片被认出来
        and is_file_paper(real_img.upper() + ".PNG") is False  # 不存在的路径不认
        and not is_file_paper("page.txt")                 # 扩展名不在白名单里不认
    )
    check("自定义背景：内置名与图片路径互不误判", ok_bg,
          "white=%s / 自建图=%s / 未知名=%s / txt=%s"
          % (bg_path("white") is not None, bg_path(real_img) == real_img,
             bg_path("nosuchpaper") is None, is_file_paper("page.txt")))

    # 提示语必须真的把"不能带纸面之外的东西"说清楚 —— 这是本功能唯一的护栏
    # （刻意不做自动校验）。少了它，用户拿一张带桌面的照片进来就会出"字贴桌上"。
    nt = "\n".join(paper_bg_notice())
    ok_nt = ("字贴在桌子上" in nt and "纸面之外" in nt
             and "--paper" in nt and ".png" in nt)
    check("自定义背景提示语：说清了该用什么图、用错会怎样", ok_nt,
          "含'字贴在桌子上'=%s 含'纸面之外'=%s" % ("字贴在桌子上" in nt, "纸面之外" in nt))

    # ========== 第七批护栏（v0.7）：这批错都"结果看着正常"，只能量结果 ==========
    #
    # 共同点：函数被测透了，**接进管线才失效** ——
    # 归一被抵消、量尺与画尺不一致、截断标志写完没人读……
    # 所以断言一律落在**管线末端**：成品像素、退出码、报告文字、命中率。

    # --- 23. 图鉴必须是"浅底深字" ---
    # 原先 write_atlas_from_lib 把 alpha 直接当灰度画（墨 = 1.0 → 255 = 白），
    # 且 paste 不传 mask（整块矩形替换到白纸上）—— 每个格位成了一块黑方块 + 白笔画，
    # 与 build 时的图鉴（纸色底、深色笔画）对比度完全相反；而图鉴唯一的用途
    # 就是看笔画粗细/发虚，反相之后这个判断就失真了。
    # 判据直接量**字形框内**的暗像素占比：正确时框内几乎全白，反相时反过来。
    at_lib = os.path.join(tmp, "atlaslib")
    at_dir = os.path.join(at_lib, "glyphs", str(ord("甲")))
    os.makedirs(at_dir, exist_ok=True)
    at_a = np.zeros((40, 40), np.float32)
    at_a[5:35, 19:21] = 1.0                     # 竖
    at_a[19:21, 5:35] = 1.0                     # 横（"十"字，墨量接近真实汉字）
    at_rgba = np.dstack([np.full((40, 40, 3), ink_rgb(), np.uint8),
                         (at_a * 255).astype(np.uint8)])
    Image.fromarray(at_rgba, "RGBA").save(os.path.join(at_dir, "00.png"))
    with open(os.path.join(at_lib, "manifest.json"), "w", encoding="utf-8") as _f:
        json.dump({"chars": {"甲": ["glyphs/%d/00.png" % ord("甲")]}}, _f)
    at_png, _at_n = write_atlas_from_lib(at_lib)
    at_dark = -1.0
    if at_png:
        at_g = np.asarray(Image.open(at_png).convert("L"))
        # 复算 paste 的落点（规则与 write_atlas_from_lib 一致）：第 0 格、居中、y 偏移 22
        _sc = min((CELL - 26) / 40.0, (CELL - 26) / 40.0)
        _bw = max(1, int(40 * _sc))
        _bx, _by = (CELL - _bw) // 2, 22
        _box = at_g[_by:_by + _bw, _bx:_bx + _bw]
        if _box.size:
            at_dark = float((_box < 64).mean())
    check("图鉴是浅底深字（字形框内暗像素 < 50%）",
          at_png is not None and 0 <= at_dark < 0.50,
          "框内暗像素 %.1f%%（反相时约 93%%）" % (at_dark * 100))

    # --- 24. 倾角缓存的键必须锚在"源字形身份"上 ---
    # 原来键 = (tol, cap, gain, 形状, 和, 平方和)，而 upright() 拿到的是 vary() 里
    # **刚做过随机形变**的数组 —— 每次指纹都不同，命中率恒为 0，等于没缓存；
    # 而且结果完全正确，从任何输出上都看不出来。这条断言直接量命中率。
    _UPRIGHT_CACHE.clear()
    CACHE_HIT[0] = 0
    CACHE_MISS[0] = 0
    _cache_rng = np.random.default_rng(3)
    _stroke = np.zeros((40, 40), np.float32)
    _stroke[4:36, 18:22] = 1.0                  # 一根竖画，足以让 lean_report 有判定
    _pk = Picker({"丨": [_stroke]}, rng=_cache_rng)
    for _i in range(20):
        _gid, _src = _pk.take("丨", float(_i * 30), 0.0)
        vary(_src, _cache_rng, gid=_gid)
    _ctot = CACHE_HIT[0] + CACHE_MISS[0]
    check("倾角缓存命中率 > 80%（键锚在源字形身份）",
          _ctot > 0 and CACHE_HIT[0] / float(_ctot) > 0.8,
          "命中 %d/%d" % (CACHE_HIT[0], _ctot))

    # --- 25. 自由字版零入库也必须退出码非 0 ---
    # rc=4 原来只挂在 build_form_sheet 的 empty 标志上，自由字版路径从不设它 ——
    # 拍了一张空白纸照样 rc=0，脚本里 `handglyph build ... && 下一步` 照常往下走。
    blank_free = os.path.join(tmp, "blank_free.png")
    Image.new("RGB", (600, 400), (252, 252, 250)).save(blank_free)
    rc_free = None
    try:
        with quiet():
            rc_free = run(["build", blank_free, "-o", os.path.join(tmp, "freelib"),
                           "--expect", "甲乙"])
    except BaseException as e:
        rc_free = "%s" % type(e).__name__
    check("自由字版零入库退出码非 0", rc_free is not None and rc_free != 0,
          "退出码 %s" % rc_free)

    # --- 26. --flow 装不下时必须报出来（端到端：退出码 + 报告文字）---
    # 原来 render_flow 里 truncated 只被置真然后 break，之后再没人读它：
    # 那一页剩余的行既不在本页、也不在任何后续页，而报告还写着"已压到阈值内"。
    # 这里造一张小纸 + 把 plan_pages 打桩成"全塞一页"，必然超页，验证它会被报出来。
    flow_ok, flow_txt = False, "环境不足（缺中文字体或白纸背景）"
    if fpath and bg_path("white"):
        small_bg = os.path.join(tmp, "small_paper.png")
        Image.new("RGB", (400, 320), (252, 252, 250)).save(small_bg)
        flspec = os.path.join(tmp, "flow.txt")
        with open(flspec, "w", encoding="utf-8") as f:
            f.write("\n".join(["今天天气不错"] * 6) + "\n")
        _orig_plan = plan_pages
        globals()["plan_pages"] = (lambda items, rows, bot, page_h, no_wrap:
                                   [list(range(len(items)))])
        try:
            with quiet():
                rc_flow = run(["render", flspec, "--lib", libdir, "--paper", small_bg,
                               "--flow", "-o", os.path.join(tmp, "flow_out.png")])
        except BaseException as e:
            rc_flow = "%s" % type(e).__name__
        finally:
            globals()["plan_pages"] = _orig_plan
        _fa = os.path.join(tmp, "flow_out.audit.txt")
        _fatxt = open(_fa, encoding="utf-8").read() if os.path.isfile(_fa) else ""
        flow_ok = (rc_flow == 5) and ("内容被截断" in _fatxt)
        flow_txt = "rc=%s，报告含'内容被截断'=%s" % (rc_flow, "内容被截断" in _fatxt)
    check("--flow 装不下时：rc=5 且报告写明截断", flow_ok, flow_txt)

    # --- 27. 被纸面边界裁掉过半的字形要进审计报告 ---
    # rec["clipped"] 原来只写不读：BLEND_CLIP_WARN 注释写着"必须程序说"，
    # 报告里一个字都没有。
    _clip_txt, _ = audit([{"ch": "甲", "dens": 0.2, "sharp": 0.7, "clipped": 0.66}],
                         expected=1)
    check("审计报告会点名被裁掉过半的字形",
          "裁" in _clip_txt and "66" in _clip_txt,
          "含裁切行：%s" % ("是" if "裁掉过半" in _clip_txt else "否"))

    # --- 28. 墨量过低被跳过的字形要在审计里标"跳过"并点名 ---
    _sk_txt, _ = audit([{"ch": "甲", "dens": 0.0, "sharp": 0.0, "skipped": "墨量过低"}],
                       expected=1)
    check("墨量过低的字形：审计标「跳过」并点名",
          "跳过" in _sk_txt and "墨量过低" in _sk_txt,
          "含'跳过'=%s 含原因=%s" % ("跳过" in _sk_txt, "墨量过低" in _sk_txt))

    # --- 29. metrics.json 损坏时要重算，而不是报"字库为空" ---
    bm_lib = os.path.join(tmp, "badmetric")
    os.makedirs(os.path.join(bm_lib, "glyphs"), exist_ok=True)
    bm_dir = os.path.join(bm_lib, "glyphs", str(ord("甲")))
    os.makedirs(bm_dir, exist_ok=True)
    bm_arr = np.zeros((40, 40, 4), np.uint8)
    bm_arr[8:32, 8:32] = [40, 40, 46, 255]
    Image.fromarray(bm_arr, "RGBA").save(os.path.join(bm_dir, "00.png"))
    with open(os.path.join(bm_lib, "manifest.json"), "w", encoding="utf-8") as _f:
        json.dump({"chars": {"甲": ["glyphs/%d/00.png" % ord("甲")]}}, _f)
    with open(os.path.join(bm_lib, "metrics.json"), "w", encoding="utf-8") as _f:
        _f.write("{ 这不是合法 JSON")
    with quiet():
        bm_txt, _ = lib_quality(bm_lib)
    bm_recovered = isinstance(read_json(os.path.join(bm_lib, "metrics.json")), dict)
    check("metrics.json 损坏时重算（不报“字库为空”）",
          "字库为空" not in bm_txt and bm_recovered,
          "报告首行：%s" % bm_txt.splitlines()[0][:40])

    # --- 30. 补字单容量：列超限 / 字太多（含 --copies 展开）都必须被拒 ---
    _cap_png = os.path.join(tmp, "cap.png")
    with quiet():
        _cap_bad1 = make_form(["甲"], _cap_png, cols=99)
        _cap_bad2 = make_form(["甲"] * 200, _cap_png, cols=12, copies=3)
        _cap_ok = make_form(["甲"], _cap_png, cols=12)
    check("补字单容量：列超限 / 字太多（含 copies）都被拒、单字单格正常",
          _cap_bad1 is None and _cap_bad2 is None
          and bool(_cap_ok) and os.path.isfile(_cap_png),
          "99 列 → %s / 200 字×3 遍 → %s / 12 列 → %s"
          % (_cap_bad1, _cap_bad2, bool(_cap_ok)))

    # --- 31. 单笔画字符集必须真的含全角引号 ---
    # 该集合原来靠"相邻字符串字面量拼接"才没报语法错，代价是全角引号 “” 压根
    # 没进集合（实测 38 个字符，应为 40），而且再加一个 ASCII 双引号就会 SyntaxError。
    _mq = [c for c in ("“", "”") if c not in SINGLE_STROKE_CHARS]
    check("单笔画字符集含全角引号（不再靠隐式拼接）",
          not _mq and len(SINGLE_STROKE_CHARS) >= 40,
          "缺 %s，集合 %d 个字符" % ("".join(_mq) or "无", len(SINGLE_STROKE_CHARS)))

    # --- 32. --stroke 必须幂等（同进程调两次不能累积）---
    # 原来那几行是从"当前全局值"派生的，第二次调用就变成 k²、第三次 k³。
    idem_ok, idem_txt = False, "环境不足（缺中文字体或白纸背景）"
    if fpath and bg_path("white"):
        _ispec = os.path.join(tmp, "idem.txt")
        with open(_ispec, "w", encoding="utf-8") as f:
            f.write("今天\n")
        for _i in range(2):
            with quiet():
                run(["render", _ispec, "--lib", libdir, "--paper", bg_path("white"),
                     "--stroke", "0.5", "-o", os.path.join(tmp, "idem%d.png" % _i)])
        idem_ok = abs(STROKE_BIAS - _STROKE_BIAS0 * 0.5) < 1e-9
        idem_txt = "STROKE_BIAS = %.4f（期望 %.4f）" % (STROKE_BIAS, _STROKE_BIAS0 * 0.5)
        # 还原默认档，别影响后面的断言
        globals()["STROKE_BIAS"] = _STROKE_BIAS0
        globals()["STROKE_LO"] = _STROKE_LO0
        globals()["STROKE_HI"] = _STROKE_HI0
        globals()["STROKE_GATE"] = _STROKE_GATE0
    check("--stroke 幂等：同进程调两次不累积", idem_ok, idem_txt)

    shutil.rmtree(tmp, ignore_errors=True)
    res.append("")
    if fails[0]:
        res.append("结果：%d 项失败，请检查上面的失败项。" % fails[0])
    else:
        res.append("结果：全部通过。")
    return res, fails[0]


def main(argv=None):
    global ROTATE, STRETCH, LEAN_TOL, LEAN_CAP, LEAN_GAIN, LATIN, BASE_DRIFT, NO_WRAP
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="handglyph", description="用你自己的手写样本生成手写风图片")
    # --version：不依赖任何子命令就能查版本号。
    # ⚠ v0.7 的修改说明里写了"顺带补了 --version"，但实测**并没有补上** ——
    #   `handglyph.py --version` 当时返回 rc=2「arguments are required: cmd」，
    #   因为 `required=True` 的子命令解析发生在任何可选参数之前。现在补在这里。
    ap.add_argument("--version", action="version",
                    version="handglyph %s" % __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="字版照片 -> 字形库")
    b.add_argument("sample")
    b.add_argument("-o", "--out", default="library")
    b.add_argument("--expect", help="字版上的字符顺序，用于自动命名")
    b.add_argument("--merge", action="store_true", help="并入已有字库（保留旧实例）")
    b.add_argument("--replace", action="store_true", help="并入并替换这些字的旧实例（重写某几个字时用）")
    b.add_argument("--rebuild", action="store_true",
                   help="推倒重建：目标已有字库时清空旧映射（危险，旧字形会变成孤儿文件）")
    b.add_argument("--debug", action="store_true", help="输出切分诊断图与丢弃原因")
    b.add_argument("--form-sheet", action="store_true",
                   help="按补字单处理：先检格子，格位顺序即字符顺序。用它才能过滤掉表单上的打印内容")
    b.add_argument("--cols", type=int, default=12, help="补字单列数（配合 --form-sheet，默认 12）")

    c = sub.add_parser("coverage", help="检查缺字")
    c.add_argument("--lib", default="library")
    c.add_argument("-t", "--text", required=True)

    f = sub.add_parser("form", help="生成补字单")
    f.add_argument("chars", nargs="?", default="")
    f.add_argument("-o", "--out", default="form.png")
    f.add_argument("--kind", default="text", choices=["text", "symbols"],
                   help="text 汉字补字单 / symbols 字符版（数字、字母、符号）")
    f.add_argument("--cols", type=int, default=12,
                   help="列数（默认 12）。与 build --form-sheet --cols 必须同值，否则格位对不上")
    f.add_argument("--copies", type=int, default=1, metavar="N",
                   help="每个字连着排 N 格（默认 1）。写多遍才有多个字形实例可供轮转，"
                        "同一个字在同一页里出现两次就不会长得一模一样")

    q = sub.add_parser("quality", help="字库质检：哪些字需要重写（稳健离群，自动分组校准）")
    q.add_argument("--lib", default="library")
    q.add_argument("--k", type=float, default=3.0, help="离群倍数，越大越宽松")
    q.add_argument("--form", help="把需要重写的字直接生成补字单到该路径")

    cal = sub.add_parser("calibrate", help="用字库自身分布校准阈值")
    cal.add_argument("--lib", default="library")
    cal.add_argument("--k", type=float, default=3.0)

    pr = sub.add_parser("prune", help="淘汰字库里的差实例（不设数量上限）")
    pr.add_argument("--lib", default="library")
    pr.add_argument("--ratio", type=float, default=0.85, help="相对闸门：低于该字最优分的这个比例就删")
    pr.add_argument("--floor", type=float, default=0.35, help="绝对闸门下限（与全库 q10 取大）")
    pr.add_argument("--dry", action="store_true", help="只报告，不删除")

    cl = sub.add_parser("clean", help="对既有字库统一去格线、裁边")
    cl.add_argument("--lib", default="library")
    cl.add_argument("--dry", action="store_true")

    sh = sub.add_parser("shape", help="形状校验：排查被压扁/比例异常的字形")
    sh.add_argument("--lib", default="library")
    sh.add_argument("--limit", type=float, default=SHAPE_RATIO_LIMIT,
                    help="宽高比偏差倍数阈值（默认 %.1f）" % SHAPE_RATIO_LIMIT)

    at = sub.add_parser("atlas", help="改名逃生门：按 `序号=字` 文本重建字库里的字符映射")
    at.add_argument("--lib", default="library")
    at.add_argument("--fix", metavar="映射文件",
                    help="按该文件里的 `序号=字` 行就地改名（序号看 atlas.png 上的红字）")
    at.add_argument("--dry", action="store_true", help="与 --fix 同用：只试算不真改")

    ex = sub.add_parser("extract", help="从图片提取文本，生成页面描述骨架")
    ex.add_argument("image")
    ex.add_argument("-o", "--out", default="page.txt")
    ex.add_argument("--lang", default="chi_sim+eng")
    ex.add_argument("--backend", default="auto", choices=["auto", "tesseract", "paddle"],
                    help="OCR 后端：auto 有哪个用哪个 / tesseract / paddle（需另装，中文更准）")

    sub.add_parser("doctor", help="环境自检：依赖、字体、背景图、字库状态")
    sub.add_parser("selftest", help="自检：跑内置回归测试（改代码后先跑它）")

    # 分批与批间粗细比对。为什么要有这条命令：补字是**分次写**的，
    # 不同次写的字落笔轻重会不一样（同一批内是齐的）。这条命令把
    # "哪一批跟别的批不是一路粗细"算出来，并给出重写清单。
    bt = sub.add_parser("batch", help="字库分批：列出各批次，比对批间粗细，超限就提示重写")
    bt.add_argument("--lib", default="library")
    bt.add_argument("--tol", type=float, default=BATCH_STROKE_TOL,
                    help="批间粗细差的容忍带（默认 %.2f = %.0f%%）；超过就判为需重写"
                         % (BATCH_STROKE_TOL, BATCH_STROKE_TOL * 100))

    r = sub.add_parser("render", help="按纸面排版并合成")
    r.add_argument("spec")
    r.add_argument("--lib", default="library")
    # ⚠ 这里**故意不用 choices**：choices 会把"自定义背景路径"一并拦掉。
    # 校验下移到 bg_path() —— 内置名走 backgrounds/，否则当图片路径打开。
    r.add_argument("--paper", default="white", metavar="纸面",
                   help="纸面：内置 white 白纸 / lined 横线 / grid 方格 / redgrid 红格 / "
                        "cream 米黄笔记本；也可以直接给一张**整幅是纸**的图片路径"
                        "（如 --paper \"D:\\我的纸.png\"）。"
                        "⛔ 背景图里不能有桌面、手掌等纸面之外的东西 —— 见 doctor 的说明。")
    r.add_argument("-o", "--out", default="out.png")
    r.add_argument("--seed", type=int, default=20260914)
    r.add_argument("--rotate", type=float, default=ROTATE,
                   help="整字微旋转上限（度，默认 %.2f）。**调大它就明显了** —— "
                        "手写的倾角集中在 ±1.5° 内，超过 1.5 就能逐字看出歪。" % ROTATE)
    r.add_argument("--stroke", type=float, default=1.0, metavar="K",
                   help="笔画粗细波动的倍数（默认 1.0 = 微妙档）。"
                        "1.0 下单字墨量起伏约 ±3%%，肉眼只在并排比对时可见；"
                        "0 = 完全关掉粗细扰动（字库原样，只保留形变与旋转）。"
                        "大于 1.6 会开始像描粗的艺术字，不建议。")
    r.add_argument("--stretch", type=float, default=0.14, help="笔画加长比例上限（相对字高）")
    r.add_argument("--lean-tol", type=float, default=LEAN_TOL,
                   help="竖画允许多少度倾斜（超过即扶正，默认 1.2 度）；0=关闭")
    r.add_argument("--lean-cap", type=float, default=LEAN_CAP,
                   help="单次扶正的幅度上限（度，默认 3.5）；设 0 等于只报告不扶正")
    r.add_argument("--lean-gain", type=float, default=LEAN_GAIN,
                   help="扶正的置信门限（默认 1.0）：长竖画得分须比曲线中位高该比例才动手")
    r.add_argument("--latin", type=float, default=0.95,
                   help="字母与数字相对汉字的高度比（1.0 = 与汉字等高）")
    r.add_argument("--no-wrap", action="store_true",
                   help="长行不折行，改为整行等比缩小（旧行为；折行更自然但会多占几行）")
    r.add_argument("--drift", type=float, default=BASE_DRIFT,
                   help="行基线漂移幅度（像素，默认 %.1f）；0=关闭" % BASE_DRIFT)
    r.add_argument("--flow", action="store_true",
                   help="自动排版：把整篇内容按\"填满一页再进下一页\"重新分页。"
                        "内容短时不会一页尾巴留一大片空白，而是把后续内容提前填进来；"
                        "内容长时自动分成多页（文件名加 _p1/_p2…）。")
    r.add_argument("--flow-slack", type=float, default=FLOW_SLACK, metavar="PX",
                   help="页尾空白的容忍上限（像素，默认 %d）。"
                        "只有超过这个值才算\"大片空白\"，才值得回填。" % FLOW_SLACK)

    a = ap.parse_args(argv)
    if a.cmd == "build":
        r = build(a.sample, a.out, a.expect, merge=a.merge, debug=a.debug,
                  replace=a.replace, form_sheet=a.form_sheet, cols=a.cols,
                  rebuild=a.rebuild)
        print(json.dumps(r, ensure_ascii=False))
        if r.get("refused"):
            # 2 = 主动拒绝（护栏拦住），与"参数用错"同一类：都是"你没让我干成"。
            return 2
        # 4 = 跑完了但一个字都没入库。判据看**结果本身**（glyphs == 0），
        # 而不是只看某个分支里设的 empty 标志 —— 补字单路径会设它，自由字版
        # （拍了一张空白纸/过曝照片）从不设，只查标志位就会漏掉后者，
        # 于是"零产出"照样以 rc=0 返回，脚本里 `&& 下一步` 照常往下走。
        if r.get("empty") or int(r.get("glyphs", 0) or 0) == 0:
            # 不用 3：3 已被 run() 占用为"未预料的运行期错误"，语义不同 ——
            # 这里没有异常，只是本次调用没产出任何东西。
            # 注意：这只对**本次零入库**报警，不是"库里本来就没字"。
            return 4
    elif a.cmd == "coverage":
        lib = load_library(a.lib)
        need, miss = coverage(lib, a.text)
        if miss:
            order = sorted(miss, key=lambda k: -miss[k])
            print("缺 %d 个字：%s" % (len(miss), " ".join(order)))
            for ln in refill_notice(order, "字库里还没有这 %d 个字。" % len(miss)):
                print(ln)
            return 1
        print("字库覆盖完整（共 %d 个不同字符）" % len(need))
    elif a.cmd == "form":
        if a.kind == "symbols":
            # symbols 是"数字 + 字母 + 标点 + 数学符号"的整套清单，见 SYMBOLS 常量。
            # 传了位置参数就按用户给的字符来（那是更精确的诉求），
            # 没传才铺整套 —— 否则用户想看一个符号却打出一整张单子。
            chars = [c for c in (a.chars or SYMBOLS) if c not in " \t\"'"]
            print("字符版补字单：%d 个字符（数字 / 字母 / 标点 / 数学符号）。" % len(chars))
        else:
            chars = [c for c in a.chars if c not in " \t"]
        p = make_form(chars, a.out, cols=a.cols, copies=a.copies)
        if p is None:
            return 2
        print(p)
        if a.copies > 1:
            # 把 --expect 要照抄的串直接打印出来。手敲 3 遍很容易漏一个，
            # 而漏了不会报错 —— 字数对上就行，只是每个字都被安到错的名字上。
            print("")
            print("每个字写了 %d 遍。入库时 --expect 要照抄这一串（字符顺序与格位一致）：" % a.copies)
            print("  %s" % form_expect(chars, a.copies))
    elif a.cmd == "prune":
        r = prune(a.lib, ratio=a.ratio, floor=a.floor, dry=a.dry)
        if r is None:
            print("字库为空。")
        else:
            print("%s：删除 %d 个差实例（相对闸门 %.2f×该字最优，绝对闸门 max(%.2f, 全库 q10=%.3f)）"
                  % ("试算" if r["dry"] else "已执行", len(r["removed"]), r["ratio"], r["floor"], r["q10"]))
            for ch, n0, n1, best, thr in r["stats"][:12]:
                if n1 < n0:
                    print("  %s  实例 %d→%d  最优%.3f 阈值%.3f" % (ch, n0, n1, best, thr))
            print("  未设数量上限：只要过闸门就留下。")
    elif a.cmd == "clean":
        r = clean(a.lib, a.dry)
        print("%s：去格线 %d 处，切邻字 %d 处，重裁 %d 个字形，剔残片 %d 个"
              % ("试算" if a.dry else "已执行", r["lines"], r["split"], r["trimmed"], r["dropped"]))
        for ch, rel, prob in r["list"][:10]:
            print("  剔 %s/%s（%s）" % (ch, os.path.basename(rel), prob))
    elif a.cmd == "shape":
        shp_lines, _bad = shape_lines(a.lib, limit=a.limit)
        for ln in shp_lines:
            print(ln)
    elif a.cmd == "batch":
        b_lines, over = batch_report(a.lib, tol=a.tol)
        for ln in b_lines:
            print(ln)
        # 超限时返回 1：跟 coverage 报缺字同一类 —— 「跑通了，但有东西需要你处理」。
        # 不返回 0，是因为调用方（脚本/自动化）需要能区分"没事"和"该重写了"。
        if over:
            return 1
    elif a.cmd == "atlas":
        # 两条路都先刷新图鉴 —— 图鉴是这条路唯一的输入，没有它就没法核对序号。
        ap_png, n_drawn = write_atlas_from_lib(a.lib)
        if a.fix is None:
            if not ap_png:
                print("字库里没有可用字形，无法生成图鉴：%s" % a.lib)
                return 2
            print("已写出图鉴：%s（%d 个字形实例）" % (ap_png, n_drawn))
            print("")
            print("  下一步：打开这张图，核对每个序号（左上角红字）该是什么字。")
            print("  蓝色小字是它**现在**的名字 —— 名字不对就说明 build 时的自动命名错位了。")
            print("")
            print("  然后在文本文件里逐行写 `序号=字`，再跑：")
            print("")
            print("    handglyph atlas --fix 映射.txt --dry     # 先试算")
            print("    handglyph atlas --fix 映射.txt           # 确认后真改")
            print("")
            print("  映射格式（两种都认）：")
            print("          3=确")
            print("          7=能")
            print("          或写成区段：3-6=确能控制")
            print("")
            print("  序号看 atlas.png 左上角的红字（全局顺序，从 0 开始）。")
            print("  老字库若没有实例档案（manifest 里缺 instances），序号语义是")
            print("  「该字下的第几张」，这时请写成 `字符/序号` 形式，例如：")
            print("          确/0=确")
            return 0
        if a.fix is not None and not ap_png:
            print("字库里没有可用字形，无法改名：%s" % a.lib)
            return 2
        r = atlas_fix(a.lib, a.fix, dry=a.dry)
        if r is None:
            return 2
        if not a.dry and r.get("moved"):
            # 改名之后再刷一次图鉴：上面那次刷新是在**改名之前**做的，
            # 蓝字标的是旧名字 —— 用户改完回头打开 atlas.png 复核，
            # 看到旧名字会以为"没改成功"。
            ap2, n2 = write_atlas_from_lib(a.lib)
            if ap2:
                print("图鉴已按新名字刷新：%s（%d 个实例）" % (ap2, n2))
    elif a.cmd == "extract":
        p, n = extract(a.image, a.out, a.lang, backend=a.backend)
        print("已写出页面描述：%s（自动提取 %d 行）" % (p, n))
    elif a.cmd == "quality":
        txt, need = lib_quality(a.lib, a.k)
        print(txt, end="")
        with open(os.path.join(a.lib, "quality.txt"), "w", encoding="utf-8") as f:
            f.write(txt)
        if a.form:
            if need:
                print(make_form(need, a.form))
            else:
                print("没有需要重写的字，未生成补字单。")
    elif a.cmd == "calibrate":
        data = calibrate(a.lib, a.k)
        print("已校准：k=%.1f，分组 %s" % (data["k"], list(data["groups"])) if data else "字库为空。")
    elif a.cmd == "doctor":
        print(doctor())
    elif a.cmd == "selftest":
        res, nf = selftest()
        for ln in res:
            print(ln)
        return 1 if nf else 0
    elif a.cmd == "render":
        ROTATE, STRETCH, LEAN_TOL, LATIN = a.rotate, a.stretch, a.lean_tol, a.latin
        LEAN_CAP = a.lean_cap
        LEAN_GAIN = a.lean_gain
        BASE_DRIFT = a.drift
        NO_WRAP = a.no_wrap
        # 粗细波动按倍数缩放。--stroke 0 表示"完全不要粗细扰动"，
        # 默认档位（STROKE_* 常量）本身已经是微妙的，这里的倍数只是给人微调空间。
        k = max(0.0, float(a.stroke))
        # ⚠ 一律从 **_STROKE_*0 基准**派生，不要读当前全局值 ——
        # 读当前值会让同进程的第二次调用把 k 平方、第三次立方（非幂等）。
        globals()["STROKE_BIAS"] = _STROKE_BIAS0 * k
        globals()["STROKE_LO"] = 1.0 - (1.0 - _STROKE_LO0) * k
        globals()["STROKE_HI"] = 1.0 - (1.0 - _STROKE_HI0) * k
        # 闸门反向缩放：倍数越小，闸门越高（越少像素被动到），k=0 时闸门 1.0 谁都不碰。
        globals()["STROKE_GATE"] = min(1.0, _STROKE_GATE0 + (1.0 - k) * (1.0 - _STROKE_GATE0))
        bg = bg_path(a.paper)
        if not bg:
            # 两种失败要分开说：传的是路径 → 文件不在；传的是名字 → 内置纸面缺失。
            # （原来这里直接 PAPERS[a.paper]，自定义路径会 KeyError 崩掉。）
            if is_file_paper(a.paper) is False and os.path.splitext(str(a.paper))[1].lower() in _BG_EXTS:
                print("找不到这个背景文件：%s" % a.paper)
                print("  检查路径有没有写错、文件是不是被移走了。")
            elif a.paper in PAPERS:
                print("找不到纸面背景图：%s" % a.paper)
                print("  应在 backgrounds/ 目录下放 %s.png（或 .jpg）。" % PAPERS[a.paper])
                print("  先运行 doctor 查看缺哪些文件。")
            else:
                print("认不出这个纸面：%r" % a.paper)
                print("  可以填内置名：%s" % " / ".join(PAPERS))
                print("  也可以填一张图片的路径，但必须**整幅都是纸**。")
            print("")
            for ln in paper_bg_notice():
                print(ln)
            return 2
        if a.flow:
            # 全局开关（NO_WRAP / BASE_DRIFT 等）在 render 里读的是模块级变量，
            # 这里先把命令行值写进去，两条路径共用同一套观感参数。
            NO_WRAP = a.no_wrap or False
            globals()["NO_WRAP"] = a.no_wrap or False
            outs, trunc_pages = render_flow(a.spec, a.lib, bg, a.out, seed=a.seed,
                                            no_wrap=a.no_wrap, slack=a.flow_slack)
            # 5 = 跑完了但有内容没落地（超页被截断）。与 build 的 4 同一类：
            # "跑完了"不等于"内容都在"，脚本必须能分辨这两种情况。
            return 5 if trunc_pages else 0
        op, trunc = render(a.spec, a.lib, bg, a.out, seed=a.seed)
        print(op)
        return 5 if trunc else 0
    return 0


def run(argv=None):
    """main() 的外壳：把未预料的异常变成一句人话，而不是一屏 traceback。

    为什么需要：main() 内部有不少地方会碰到用户数据（照片损坏、manifest 半截、
    字体文件是个坏文件）。这些都在各函数里各判各的，但判不完 ——
    兜底一层，至少能让用户看到"出了什么事、可以怎么办"，而不是对着
    `Traceback (most recent call last)` 发呆。退出码约定：
      0 成功 / 1 自检失败 / 2 用法或环境错误 / 3 未预料的运行期错误
      4 跑完了但零产出（如 build 一个字都没入库）
      5 跑完了但有内容没落地（render/render --flow 超页被截断）
      130 用户 Ctrl-C
    """
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        print("已中断（Ctrl-C）。已写出的文件保持原样，可以重跑。", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as e:
        print("", file=sys.stderr)
        print("出错：%s: %s" % (type(e).__name__, str(e)[:300]), file=sys.stderr)
        print("", file=sys.stderr)
        print("这不是预期内的情况。请先跑一次 doctor 看环境是否完整：", file=sys.stderr)
        print("    handglyph doctor", file=sys.stderr)
        print("若环境没问题，把上面这行错误连同你执行的命令一起反馈。", file=sys.stderr)
        if os.environ.get("HANDGLYPH_DEBUG"):
            import traceback
            traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(run())
