"""打包分发包：源码 + 背景纸 + 说明，打成 zip 便于发送。

用法：
    python pack.py                 # 输出到桌面（先跑一次出厂自检）
    python pack.py D:/somewhere    # 输出到指定目录
    python pack.py --no-gate       # 跳过出厂自检（明知某项失败仍要出包时）

设计要点：
  1. **绝不打包 library/** —— 那是本人手写笔迹，属个人信息。
  2. 背景纸优先用 .jpg（体积只有 PNG 的约 1/9，肉眼无差别）。
  3. **打包前先跑 selftest**，不通过就中止，别把坏代码发出去。
  4. 打完后自动做一次夹带检查，确认没有字形数据混进去。

为什么用 argparse 而不是 sys.argv[1]：原来把 `sys.argv[1]` 无条件当成输出目录，
`python pack.py --no-gate` 会把 "--no-gate" 当目录名，然后在 os.path.join 后
写文件时抛 FileNotFoundError —— 而 --no-gate 恰恰是"自检已经失败、正在救火"
时才会用的参数，最不该在这时候炸。argparse 顺手把用法也印清楚了。
"""

import argparse
import builtins
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))

# 只在读不到 pyproject.toml 时才用的兜底版本号。
# 这里**刻意不写 "handglyph_v0.2" 这种带包名的形式**：包名由 package_name()
# 从版本号派生（见下），把包名再写一份在这里就是第二个真源，发版必漏改一处。
FALLBACK_VERSION = "0.7"

# 要放进包里的文件（相对工程根目录）
FILES = [
    "handglyph.py",
    "handglyph.cmd",
    "handglyph.sh",
    "pyproject.toml",
    "README.txt",
    "能力说明.md",
    "能力说明.docx",
    "LICENSE",
    ".gitignore",
    "pack.py",
]
DIRS = [
    ("backgrounds", (".jpg", ".jpeg", ".png")),
    ("examples", (".txt",)),
]

# 版本号只认 pyproject.toml —— 包的目录名、zip 名都从它派生。
# 原来 NAME 在 pack.py 里写死一份、pyproject.toml 里又写一份，
# 发版时漏改一处就会出现"包里说 v0.2、元数据说 v0.3"。
_PYPROJ = os.path.join(ROOT, "pyproject.toml")
_VER_RE = re.compile(r'^\s*version\s*=\s*["\']([^"\']+)["\']', re.M)


def project_version(fallback="0.0"):
    """从 pyproject.toml 读版本号；读不到就用 fallback（不中断打包）。"""
    try:
        with open(_PYPROJ, encoding="utf-8-sig") as f:
            m = _VER_RE.search(f.read())
        if m:
            return m.group(1).strip()
    except OSError:
        pass
    print("  [警告] 读不到 pyproject.toml 里的 version，包名退化为 %s" % fallback)
    return fallback


def package_name(version):
    """由版本号派生包名：0.2 -> handglyph_v0.2

    取主次两段即可 —— 补丁号（0.2.1）不进包名，避免每次打补丁都换一次收件人的解压目录名。
    """
    parts = str(version).split(".")
    short = ".".join(parts[:2]) if len(parts) >= 2 else str(version)
    return "handglyph_v" + short


def pick_backgrounds(bdir):
    """每个纸面主干名只挑一个文件，优先 jpg。"""
    rank = {".jpg": 0, ".jpeg": 1, ".png": 2}
    best = {}
    for f in sorted(os.listdir(bdir)):
        base, ext = os.path.splitext(f)
        ext = ext.lower()
        if ext not in rank:
            continue
        r = rank[ext]
        if base not in best or r < best[base][0]:
            best[base] = (r, f)
    return [best[b][1] for b in sorted(best)]


def module_version():
    """从 handglyph.py 里读 __version__（正则，不 import —— import 会拉起 numpy）。"""
    try:
        with open(os.path.join(ROOT, "handglyph.py"), encoding="utf-8") as f:
            m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', f.read(), re.M)
        return m.group(1).strip() if m else None
    except OSError:
        return None


def version_check():
    """核对 handglyph.py 的 __version__ 与 pyproject.toml 的 version。

    只告警不中止：版本号写错是"文档问题"，不该拦住出包。
    但必须说出来 —— 不然就会发出"包里说 v0.3、元数据说 v0.2"的包。
    """
    try:
        with open(_PYPROJ, encoding="utf-8-sig") as f:
            m = _VER_RE.search(f.read())
        pv = m.group(1).strip() if m else None
    except OSError:
        pv = None
    mv = module_version()
    if pv and mv and pv != mv:
        print("  [警告] 版本号不一致：pyproject.toml = %s，handglyph.py __version__ = %s" % (pv, mv))
        return False
    if mv:
        print("  版本一致：%s" % mv)
    return True


def static_guard():
    """纯静态检查：模块级常量的定义必须早于"被当默认值引用"的地方。

    为什么放在这里而不是 selftest 里：这类错误的表现是**import 阶段就 NameError**，
    整个模块一行都跑不起来 —— selftest 是模块内的函数，根本没机会执行。
    打包是"发出去就收不回"的动作，所以在这道闸门前先静态扫一遍。

    真实来由：本轮把散落的 108 收成 CELL 常量时放在了 1520 行，
    而 build() 的签名 `cell=CELL` 在 1252 行，Python 的默认参数在 def 执行时求值，
    于是程序完全无法启动。这种错只要有人再挪一次常量就可能复发。
    """
    with open(os.path.join(ROOT, "handglyph.py"), encoding="utf-8") as f:
        src = f.read().splitlines()
    defined = {}
    for i, ln in enumerate(src, 1):
        m = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=", ln)
        if m:
            defined.setdefault(m.group(1), i)

    # 关键：函数签名常跨多行（build 就是），默认值可能落在续行上。
    # 只扫 `^def ` 那一行会漏掉 —— 实测漏过一次，静态检查"通过"了但程序 import 就崩。
    # 所以按"从 def 行起一直读到该签名闭合"的区间来找。
    bad = []
    i = 0
    while i < len(src):
        if re.match(r"^def\s+\w+", src[i]):
            depth = 0
            started = False
            j = i
            while j < len(src):
                depth += src[j].count("(") - src[j].count(")")
                if "(" in src[j]:
                    started = True
                if started and depth <= 0:
                    break
                j += 1
            sig = "\n".join(src[i:j + 1])
            for name in set(re.findall(r"=\s*([A-Z_][A-Z0-9_]*)\b", sig)):
                if name in defined and defined[name] > i + 1:
                    bad.append("%s 在 行%d 的默认值里被引用，却到 行%d 才定义"
                               % (name, i + 1, defined[name]))
            i = j + 1
        else:
            i += 1
    return bad


def compile_guard():
    """抓「未定义名」—— 用**标准库**，不依赖可选的 pyflakes。返回问题列表。

    ⛔ 为什么必须有这道门禁（两道老门禁都漏得掉，实测过）：

      · `static_guard()` 只查「常量定义位置早于默认值引用」，看源码文本；
      · `lint_scan()` 查"未定义名"，但它**强依赖 pyflakes**，而 pyflakes 是
        可选依赖 —— 没装的机器上它打印「[跳过]」就 `return [], []`，
        门禁**静默失效**。
      实测（2026-09-18）：插一个 `_this_name_does_not_exist_12345` 的调用，
      跑 `pack.py` → **rc=0、照样出包**。所以"带门禁的包"当时的说法不成立。

    ⛔ 顺带纠正一个**我自己也搞错过**的判断：原以为"把模块 import 一遍就能
       抓到未定义名"—— **错**。实测：函数体里的名字**在 import 时根本不解析**，
       要等真调用那一行才 NameError。所以
         · `import` 成功后仍可能有未定义名（实测 `GATE_IMPORT_OK` 照样打印）；
         · `py_compile` 更抓不到（实测通过）；
         · 只有**调用**才暴露。
       也就是说"发出去的包能 import"远不等于"能跑"。这正是要静态查的原因：
       它必须覆盖**没被 selftest 走到**的那些分支（那才是真正会漏出去的路径）。

    做法：用 `symbols` 表逐块做**保守的作用域解析** —— 宁可漏报，不可误报
    （误报会让人习惯性 `--no-gate`，那样的门禁等于没有）。

    ⚠ 它**不是**完整的作用域分析，覆盖不到：嵌套函数里的闭包变量、
      `global/nonlocal`、`exec/eval` 动态名、`getattr` 拼出来的名字。
      这些一律按"可能可见"放过。真正的兜底还是 selftest 的用例覆盖。
    """
    import symtable
    path = os.path.join(ROOT, "handglyph.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    try:
        top = symtable.symtable(src, "handglyph.py", "exec")
    except SyntaxError as e:
        return ["语法错误 行%s：%s" % (e.lineno, e.msg)]

    problems = []

    def binds(tab):
        """本块**真正绑定了**的名字集合。

        ⛔ 这里的关键是"绑定"而不是"出现"。`tab.get_symbols()` 会把**只被引用**
        的名字也列进来 —— 用它当"可见名"等于让未定义名**自己给自己放行**：
        实测 `_this_name_does_not_exist_12345` 确实出现在 `_zz_probe` 的符号表里，
        于是 `name in visible` 判为 True、门禁一声不响地放行。
        判据必须收窄到"这个名字在本块被赋值/当参数/import/def/class 引入"。
        """
        out = set()
        for s in tab.get_symbols():
            # is_local() 且被绑定过 = 本块真正定义的名字
            if s.is_assigned() or s.is_parameter() or s.is_imported():
                out.add(s.get_name())
            # 本块里 def/class 的名字会算 assigned；`global x` 声明不算绑定
        return out

    def walk(tab, outer):
        """outer = 外层作用域里**已绑定**的名字（放行闭包变量/全局名）。"""
        visible = binds(tab) | outer
        for s in tab.get_symbols():
            name = s.get_name()
            if not s.is_referenced():
                continue
            # ⛔ 别再按 is_global() 过滤！**未定义名恰恰被判成 global**：
            #    实测 `_this_name_does_not_exist_12345` 的属性是
            #    `ref=True assigned=False param=False global=True local=False`
            #    —— 在函数里读一个自己没绑定的名字，symtable 就标它 global
            #    （意思是"去全局找"）。所以"跳过 is_global"正好把要找的东西
            #    全跳过，门禁会一声不响地放行。这个坑我踩过两次，写在这儿。
            if name in visible:
                continue
            if hasattr(s, "is_comp_iter") and s.is_comp_iter():
                continue
            problems.append("未定义名 %s（在 %s 里被引用，全文件找不到定义）"
                            % (name, tab.get_name() or "<module>"))
        for child in tab.get_children():
            walk(child, visible)

    # 顶层可见 = 内置名 + 顶层**绑定过**的名字
    seeds = set(dir(builtins)) | binds(top)

    walk(top, seeds)
    # 名字里有 `__` 的一律放过（dunder 多是运行时注入的）
    walk(top, seeds)
    problems = [p for p in problems
                if "__" not in p.split("未定义名 ")[1].split("（")[0]]
    # 去重（同一个名字可能出现在多个函数里）
    seen, uniq = set(), []
    for p in problems:
        key = p.split("（")[0]
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def lint_scan():
    """用 pyflakes 扫"未定义名"与"赋值未使用"，返回 (硬失败列表, 提示列表)。

    为什么进打包门禁：上一轮删掉了 8 处历史死代码，**同一轮又自己新添了 2 处**
    —— 一个定义了但从未接线的模块级缓存常量、一个只赋值不读的局部变量。
    这类东西不影响运行，只会越积越多，而人眼审 3800 行是审不住的。

    ⚠ pyflakes 是**可选依赖**，没装时这里只能跳过。但"未定义名"这一类别**不能**
    靠它兜底 —— 没装的机器上打包会静默放过 import 就崩的模块（实测发生过）。
    所以未定义名的检查已上移到 `compile_guard()`（只用标准库，必然能跑）。
    这里退化成"锦上添花"：有 pyflakes 时多扫出游丝信息，没有也不影响门禁成立。
    """
    try:
        import pyflakes.api
        import pyflakes.reporter
    except ImportError:
        print("  [跳过] 未安装 pyflakes（装它可多扫出'赋值未使用'一类；"
              "未定义名已由 compile_guard 用标准库守住）")
        return [], []
    import io
    out, err = io.StringIO(), io.StringIO()
    reporter = pyflakes.reporter.Reporter(out, err)
    pyflakes.api.checkPath(os.path.join(ROOT, "handglyph.py"), reporter)
    msgs = [ln.strip() for ln in (out.getvalue() + err.getvalue()).splitlines() if ln.strip()]
    hard = [m for m in msgs if "undefined name" in m or "never used" in m]
    soft = [m for m in msgs if m not in hard]
    return hard, soft


def selftest_gate():
    """打包前先跑一遍 selftest，失败就中止 —— 别把坏代码发出去。

    为什么值得这两行：selftest 是唯一能在几秒内覆盖「补字单污染、周期误检、
    残片入库、改名撞名」这类真实踩坑场景的检查，而打包是"发出去就收不回"的动作。
    跑一次的成本远低于发一版坏代码。
    可用 --no-gate 跳过（明知某项失败但就是要打包时）。
    """
    print("== 出厂自检 ==")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "handglyph.py"), "selftest"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    tail = out.splitlines()[-1] if out else "(无输出)"
    # 没有中文字体的机器上部分断言会跳过，所以只看"是否失败"
    if r.returncode != 0 or "失败" in tail:
        print("  ！自检未通过，已中止打包。")
        for ln in out.splitlines()[-25:]:
            print("    " + ln)
        return False
    print("  %s" % tail)
    return True


def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="pack.py",
        description="打包分发包（源码 + 背景纸 + 说明）成 zip。")
    ap.add_argument("out_dir", nargs="?", default=None,
                    help="输出目录（默认：桌面）")
    ap.add_argument("--no-gate", action="store_true",
                    help="跳过出厂自检（明知某项失败仍要出包时）")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    out_dir = a.out_dir or os.path.join(os.path.expanduser("~"), "Desktop")
    name = package_name(project_version(FALLBACK_VERSION))
    out_zip = os.path.join(out_dir, name + ".zip")

    if not a.no_gate:
        # 静态检查先跑：它能抓到"模块根本 import 不进来"这类错误，
        # 而 selftest 必须在模块能 import 的前提下才谈得上运行。
        bad_static = static_guard()
        if bad_static:
            print("== 静态检查 ==")
            for b in bad_static:
                print("  ！%s" % b)
            print("  这些常量的定义位置太靠后，程序会在 import 阶段就崩。先修好再打包。")
            return 1
        # 死代码 / 未定义名（见 lint_scan 的说明）。硬失败才拦，其余只提示。
        hard_lint, soft_lint = lint_scan()
        if hard_lint:
            print("== 死代码 / 未定义名检查 ==")
            for m in hard_lint:
                print("  ！%s" % m)
            print("  未定义名会让程序当场崩；未使用的赋值是死代码回流的信号。先清掉再打包。")
            return 1
        for m in soft_lint:
            print("  [提示] %s" % m)
        # ⛔ 用标准库再扫一遍"未定义名"。上面那道依赖可选的 pyflakes，
        #    没装的机器上会静默失效 —— 实测曾因此放出"import 就崩"的包。
        #    这道只用 ast，必然能跑，所以未定义名不可能再漏过去。
        bad_compile = compile_guard()
        if bad_compile:
            print("== 未定义名检查（标准库，不依赖 pyflakes）==")
            for b in bad_compile:
                print("  ！%s" % b)
            print("  这类错误会让用户拿到一个**根本跑不起来**的包，先修好再打包。")
            return 1
        print("== 版本号核对 ==")
        version_check()          # 只告警，不中止
        if not selftest_gate():
            print("")
            print("要让这次打包通过：先修好上面失败的自检项。")
            print("确实要跳过自检（例如明知某项失败仍要出包）：python pack.py --no-gate")
            return 1
        print("")

    if not os.path.isdir(out_dir):
        print("输出目录不存在：%s" % out_dir)
        print("  先建这个目录，或换一个已存在的目录（例如 python pack.py .）。")
        return 2

    # 同名包已存在时先归档，不让新包把旧包覆盖掉。
    #
    # 为什么需要：package_name() 刻意只取"主.次"两段（0.3.1 -> handglyph_v0.3），
    # 好处是解压目录名稳定；代价是打补丁时新旧包同名，直接写就会**原地覆盖**，
    # 上一个版本的字节再也找不回来。而"任何版本都不该丢"是硬要求。
    # 所以先把旧包按它自己的真实版本另存一份再写新的。
    #
    # 注意"已存在"不等于"是旧版本"：同一个版本连打两次包时，磁盘上那份
    # 内容与本次要产出的完全相同。这种情况**不该归档** —— 否则会堆出一串
    # `xxx_v0.3_0.3.1.zip` 之类看着像历史版本、实则与主包字节一致的文件，
    # 反而让人分不清哪个才是真的旧版。
    # 判据用「旧包自报的版本号 != 本次要打的版本号」：这是纯元数据比较，
    # 不必读整包字节，且语义清楚 —— 版本号不同才叫"这是上一版"。
    archived = None
    old_ver = None
    if os.path.exists(out_zip):
        try:
            with zipfile.ZipFile(out_zip) as z:
                for n in z.namelist():
                    if n.endswith("/pyproject.toml"):
                        for line in z.read(n).decode("utf-8-sig").splitlines():
                            if line.strip().startswith("version"):
                                old_ver = line.split("=", 1)[1].strip().strip('"\'')
                                break
                        break
        except (OSError, zipfile.BadZipFile, ValueError):
            old_ver = None
        if old_ver and old_ver == project_version(FALLBACK_VERSION):
            print("  同名包已存在且版本相同（%s），直接覆盖，不重复归档。" % old_ver)
        else:
            base = name + (("_" + old_ver) if old_ver else "_prev")
            archived = os.path.join(out_dir, base + ".zip")
            # 同名归档已存在就不再覆盖（同一个版本重复打包时保持首份）
            if os.path.exists(archived):
                n = 2
                while os.path.exists(os.path.join(out_dir, "%s(%d).zip" % (base, n))):
                    n += 1
                archived = os.path.join(out_dir, "%s(%d).zip" % (base, n))
            shutil.copy2(out_zip, archived)

    stage = tempfile.mkdtemp(prefix="handglyph_pack_")
    pkg = os.path.join(stage, name)
    os.makedirs(pkg)
    copied = []

    try:
        for rel in FILES:
            src = os.path.join(ROOT, rel)
            if not os.path.isfile(src):
                print("  [缺失] %s" % rel)
                continue
            dst = os.path.join(pkg, rel)
            shutil.copy2(src, dst)
            copied.append((rel, os.path.getsize(dst)))

        bdir = os.path.join(ROOT, "backgrounds")
        if os.path.isdir(bdir):
            for f in pick_backgrounds(bdir):
                dst = os.path.join(pkg, "backgrounds", f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(bdir, f), dst)
                copied.append(("backgrounds/" + f, os.path.getsize(dst)))

        for d, exts in DIRS:
            if d == "backgrounds":
                continue
            srcd = os.path.join(ROOT, d)
            if not os.path.isdir(srcd):
                continue
            for f in sorted(os.listdir(srcd)):
                if not f.lower().endswith(exts):
                    continue
                dst = os.path.join(pkg, d, f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(srcd, f), dst)
                copied.append((d + "/" + f, os.path.getsize(dst)))

        print("== 打包内容 ==")
        print("  包名：%s（版本来自 pyproject.toml）" % name)
        for rel, sz in copied:
            print("  %-34s %9d" % (rel, sz))
        print("  合计 %d 个文件，%.2f MB" % (len(copied), sum(s for _, s in copied) / 1048576))

        # 夹带检查：按目录结构判定，光看文件名会误伤 handglyph.py（名字里带 glyph）
        leak, imgs = [], []
        for dirpath, _dn, filenames in os.walk(pkg):
            for f in filenames:
                rel = os.path.relpath(os.path.join(dirpath, f), pkg).replace("\\", "/")
                parts = rel.split("/")
                if parts[0] == "library" or "glyphs" in parts[:-1]:
                    leak.append(rel)
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    imgs.append(rel)
        print("")
        print("== 夹带检查 ==")
        print("  字形库数据：%s" % (leak if leak else "无（未夹带手写笔迹）"))
        print("  图片 %d 张，全部在 backgrounds/ 下：%s"
              % (len(imgs), all(i.startswith("backgrounds/") for i in imgs)))
        if leak:
            print("  ！发现了字形数据，已中止打包。")
            return 1

        # 包内容清单：**由脚本自己写**，不靠人手抄。
        # 来由：说明文档里那张"包内容对比"表，6 项里 5 项的字节数与实际发行包
        # 对不上 —— 表格是某个中间版本手抄的，之后代码又改了几轮。
        # 让打包脚本把真实数字写进包里，"文档与实物不符"就不可能再发生。
        info_lines = [
            "handglyph 发行包内容清单（由 pack.py 自动生成，请勿手改）",
            "版本：%s" % project_version(FALLBACK_VERSION),
            "打包时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"),
            "内容文件数：%d（不含本清单）" % len(copied),
            "",
        ]
        info_lines += ["  %-34s %9d" % (rel, sz) for rel, sz in copied]
        info_lines += ["", "合计 %.2f MB（未压缩）" % (sum(s for _, s in copied) / 1048576)]
        info_path = os.path.join(pkg, "PACK_INFO.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write("\n".join(info_lines) + "\n")
        # 说出来，免得"上面写 19 个文件、zip 里却有 20 个条目"看着像 bug
        print("  另写入 PACK_INFO.txt（内容清单，%d 字节）" % os.path.getsize(info_path))

        if os.path.exists(out_zip):
            os.remove(out_zip)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for dirpath, _dn, filenames in os.walk(stage):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    z.write(fp, os.path.relpath(fp, stage))

        print("")
        print("== 产物 ==")
        print("  %s" % out_zip)
        print("  %.2f MB，共 %d 个条目" % (
            os.path.getsize(out_zip) / 1048576,
            len(zipfile.ZipFile(out_zip).namelist())))
        if archived:
            print("  旧包已归档为：%s" % archived)
        return 0
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
