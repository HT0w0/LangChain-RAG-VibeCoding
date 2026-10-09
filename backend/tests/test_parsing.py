"""文档解析:格式分派与各个解析器。

**这一层出错的方式全都是安静的。** 解析器不抛异常、文档在界面上显示
「已就绪」、分块数也正常——只是内容不对:JS 和 CSS 混进了正文、
商品参数表整块丢了、公式单元格读成了空。这些没有一个会让别的东西变红,
只有问答时才会露出「资料里没有价格」,而那时没人会怀疑到解析头上。

`test_kb.py` 通过上传接口覆盖了 CSV / Markdown / PDF 三条路,但 **HTML 与
docx 一次都没被走到过**,而它们各自都有一个「漏掉就等于丢了最值钱的部分」
的分支(脚本样式没剥掉、docx 里的参数表没读)。这个文件补的就是这些,
外加 xlsx 那条公式警告——它盯的是一个会把整列价格读成空、而状态却是
「成功」的失败。

**走公开入口 `parse_document()`,不走 `_parse_html()` 这类私有函数。**
分派本身就是被测行为的一部分,绕过它等于少测一段。
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from app.core.exceptions import IngestError
from app.services.parsing import ParsedDocument, parse_document


def _parse(tmp_path: Path, data: bytes, name: str) -> ParsedDocument:
    """落盘 + 解析,像上传链路那样。

    **落盘文件名保留扩展名**(`store_upload` 的产物是 `{uuid4().hex}{原扩展名}`),
    这不是装饰:`openpyxl` 靠路径后缀判断格式,`load_workbook()` 拿到一个
    没有扩展名的路径会直接报「does not support file format」。第一版把
    扩展名丢了,xlsx 那条用例于是红在一个与它完全无关的地方。

    文件名只在这一处出现,所以路径后缀和 `filename=` 不可能对不上。
    """
    path = tmp_path / f"3f9a1c7e{Path(name).suffix.lower()}"
    path.write_bytes(data)
    return parse_document(path, filename=name)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <title>RX9070 显卡常见问题</title>
  <style>body { color: red; }</style>
  <script>var price = 4299; console.log('tracking-script');</script>
</head>
<body>
  <nav><a href="/">首页</a></nav>
  <header>站点头部</header>
  <h1>RX9070 显卡</h1>
  <p>官方指导价 4299 元。</p>

  <p>三年质保。</p>

  <aside>相关推荐</aside>
  <footer>版权所有</footer>
  <noscript>请开启 JavaScript</noscript>
</body>
</html>
""".encode()


class TestHtml:
    def test_the_chrome_of_the_page_never_reaches_the_index(self, tmp_path: Path) -> None:
        """**导航、页脚、页眉、侧栏、noscript 里的文字必须被剥掉。**

        它们是「正文」,但每个页面都一样:留着会让每一篇文档都长得差不多,
        检索时互相干扰,而用户在引用卡片里看到的是一段和问题无关的导航链接。
        剥掉之后文档状态仍然是「已就绪」——没有任何东西会红。

        **这一条守的是剥离列表里 nav/footer/header/aside/noscript 那几项。**
        脚本和样式也在列表里,但它们由 lxml 兜着(见下一条),所以拿掉
        列表里的 "script" 这条用例不会红——反向辨识时发现的。
        """
        parsed = _parse(tmp_path, _HTML, "FAQ.html")

        assert "首页" not in parsed.text, "nav 没被剥掉"
        assert "站点头部" not in parsed.text, "header 没被剥掉"
        assert "相关推荐" not in parsed.text, "aside 没被剥掉"
        assert "版权所有" not in parsed.text, "footer 没被剥掉"
        assert "请开启 JavaScript" not in parsed.text, "noscript 没被剥掉"
        # 真正该留下的还在——上面那些断言不能靠「把所有东西都删光」通过。
        assert "4299" in parsed.text
        assert "三年质保" in parsed.text

    def test_script_and_style_text_never_reaches_the_index(self, tmp_path: Path) -> None:
        """脚本与样式的**文字**也不能进正文。

        这份 HTML 里的 `<script>` 和 `<style>` 都带着可识别的标记
        (`tracking-script` / `color: red`),它们要是进了库,检索会把这些
        当成商品描述,引用卡片里则会显示一坨代码。

        **它守的是「结果」,不是剥离列表里那两项。** `lxml` 的 `get_text()`
        本来就不返回 script/style 的内容——把 `"script"` 从剥离列表里删掉,
        这条用例照样绿(反向辨识实测过)。所以它的价值在于:
        哪天解析器换成 `html.parser`(那个会把脚本文本吐出来),
        这条会立刻红。剥离列表里那两项是给那次更换留的保险。
        """
        parsed = _parse(tmp_path, _HTML, "FAQ.html")

        assert "tracking-script" not in parsed.text, "script 里的内容进了正文"
        assert "color: red" not in parsed.text, "style 里的内容进了正文"

    def test_blank_lines_are_collapsed(self, tmp_path: Path) -> None:
        """HTML 的 `get_text` 会产生大量空行。

        不压掉的话,分块器会切出很多**只含空白字符**的块——它们占着
        top-k 的名额,检索命中之后引用卡片里是一片空白。
        """
        parsed = _parse(tmp_path, _HTML, "FAQ.html")

        assert "\n\n" not in parsed.text
        assert all(line.strip() for line in parsed.text.splitlines())

    def test_title_and_encoding_land_in_meta(self, tmp_path: Path) -> None:
        """标题进 meta,而正常 UTF-8 文件**不该带任何编码警告**。

        标题是排查时唯一的「这是哪个页面」的线索——正文里往往没有它。
        `_parse_html` 会把它放进 meta。

        编码这里**不断言具体那个字符串**:候选表第一位是 `utf-8-sig`,
        而它解码无 BOM 的 UTF-8 完全正常,所以报出来永远是 `utf-8-sig`。
        钉住它等于钉住候选顺序——那是个没人依赖的实现细节,
        改一下顺序测试就红,而行为一点没变。要测的行为是「认出来了,
        没留警告」。
        """
        parsed = _parse(tmp_path, _HTML, "FAQ.html")

        assert parsed.meta["title"] == "RX9070 显卡常见问题"
        assert parsed.meta["encoding"].startswith("utf-8")
        assert parsed.warnings == []

    def test_an_unrecognised_encoding_is_flagged(self, tmp_path: Path) -> None:
        """认不出编码时**要留一句警告**。

        `latin-1` 能解码任何字节序列,所以它永不失败——但它几乎一定是错的。
        默默按 latin-1 读出来的是一整篇乱码,而乱码的失效方式是
        「检索不到东西」,不会报任何错。这条警告是用户唯一能看到的线索。

        `\\xff` 起头:UTF-8、gb18030、big5 都解不了它,必然落到兜底。
        """
        raw = b"\xff\xfe<p>\xe9\xff hello</p>"

        parsed = _parse(tmp_path, raw, "broken.html")

        assert parsed.meta["encoding"] == "latin-1"
        assert any("latin-1" in w for w in parsed.warnings), parsed.warnings


# ---------------------------------------------------------------------------
# docx
# ---------------------------------------------------------------------------


def _docx_bytes(*, with_table: bool = True, with_text: bool = True) -> bytes:
    import docx

    document = docx.Document()
    if with_text:
        document.add_heading("RX9070 显卡规格", level=1)
        document.add_paragraph("这款显卡采用 16GB 显存,三年质保。")

    if with_table:
        # 商品说明书里的参数表往往就在 docx 的表格里,
        # 而它恰恰是提问最常落在的地方。
        table = document.add_table(rows=3, cols=2)
        pairs = [("显存", "16GB"), ("功耗", "220W"), ("接口", "PCIe 4.0")]
        for row, (key, value) in zip(table.rows, pairs, strict=True):
            row.cells[0].text = key
            row.cells[1].text = value

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class TestDocx:
    def test_headings_become_markdown_markers(self, tmp_path: Path) -> None:
        """标题要转成 Markdown 的 `#`。

        转它是为了**复用分块层的结构感知**:`split_markdown()` 会把标题层级
        写成面包屑放进每块的元数据,于是「这条资料出自《RX9070 规格》的
        哪一节」在引用卡片上是可见的。如果 docx 的标题退化成普通段落,
        这条路就断了——而断掉的表现只是引用卡片少一行字,没人会注意。
        """
        parsed = _parse(tmp_path, _docx_bytes(), "规格.docx")

        assert "# RX9070 显卡规格" in parsed.text, parsed.text[:200]

    def test_tables_are_read(self, tmp_path: Path) -> None:
        """**docx 的表格必须被读进来。**

        这是这个解析器最值钱的一行:商品说明书的参数表常常整张放在表格里,
        漏掉它等于把「显存多大、功耗多少」这些最常被问的东西全丢了。
        而漏掉之后文档照样「已就绪」、分块数照样正常——只是答不出参数。

        断言的是**键和值都进了正文**,不是「调了 document.tables」。
        """
        parsed = _parse(tmp_path, _docx_bytes(), "规格.docx")

        assert parsed.meta["tables"] == 1
        for cell in ("显存", "16GB", "功耗", "220W", "PCIe 4.0"):
            assert cell in parsed.text, f"表格里的 {cell} 没被读出来"

    def test_a_docx_without_text_is_an_error(self, tmp_path: Path) -> None:
        """读不出任何文字的 docx 要**报错,而不是产出一篇空文档**。

        空文档入库的后果是:用户在列表里看到「已就绪」,提问时永远答不出来,
        而没有任何地方提示「这篇其实什么都没解析到」。
        """
        raw = _docx_bytes(with_table=False, with_text=False)

        with pytest.raises(IngestError):
            _parse(tmp_path, raw, "空的.docx")


# ---------------------------------------------------------------------------
# xlsx
# ---------------------------------------------------------------------------


def _xlsx_with_formula() -> bytes:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "报价"
    sheet.append(["商品名称", "单价", "数量"])
    # 写成公式,而不是算好的值。openpyxl 只写公式本身,**不写缓存结果**
    # ——这正是 `_count_formulas` 要警告的那个情形。
    sheet.append(["RX9070 显卡", "=1000*4.299", 12])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


class TestXlsx:
    def test_formula_cells_raise_a_warning(self, tmp_path: Path) -> None:
        """**含公式的工作簿要给用户一句警告。**

        `data_only=True` 读到的是公式的**缓存计算结果**,而缓存只在
        Excel/LibreOffice 保存过该文件时才存在。程序生成或从别的系统导出的
        xlsx 通常没有缓存,此时公式单元格读成 **None**——价格列整列变空,
        而文档状态是「成功」。这种失败最坏:用户以为入库了,只是模型
        从此不提价格。

        没有这条警告,用户唯一的线索是「怎么问都问不出价格」。
        """
        parsed = _parse(tmp_path, _xlsx_with_formula(), "报价.xlsx")

        assert any("公式" in w for w in parsed.warnings), parsed.warnings
        assert parsed.meta["sheets"] == ["报价"]
        # 不是公式那一列的数据照样要读出来——警告不该连累正常内容。
        assert any(row.get("商品名称") == "RX9070 显卡" for row in parsed.rows)
