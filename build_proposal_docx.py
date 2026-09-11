from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import re
import xml.etree.ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)
q = lambda tag: f"{{{W}}}{tag}"


class Node:
    def __init__(self, tag="root", attrs=None, parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs or []), parent
        self.children = []


class Parser(HTMLParser):
    void = {"meta", "br", "hr", "img", "link"}

    def __init__(self):
        super().__init__()
        self.root = Node()
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs, self.cur)
        self.cur.children.append(n)
        if tag not in self.void:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, attrs, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is not self.root:
            self.cur = n.parent

    def handle_data(self, data):
        if data:
            self.cur.children.append(data)


def find(node, tag):
    if isinstance(node, Node):
        if node.tag == tag:
            yield node
        for child in node.children:
            yield from find(child, tag)


def plain(node):
    if isinstance(node, str):
        return node
    return "".join(plain(c) for c in node.children)


def add_run(p, text, bold=False, italic=False, size=21, color=None):
    text = re.sub(r"\s+", " ", text)
    if not text:
        return
    r = ET.SubElement(p, q("r"))
    rp = ET.SubElement(r, q("rPr"))
    if bold:
        ET.SubElement(rp, q("b"))
    if italic:
        ET.SubElement(rp, q("i"))
    ET.SubElement(rp, q("sz"), {q("val"): str(size)})
    ET.SubElement(rp, q("szCs"), {q("val"): str(size)})
    if color:
        ET.SubElement(rp, q("color"), {q("val"): color})
    t = ET.SubElement(r, q("t"), {"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
    t.text = text


def inline(p, node, bold=False, italic=False, size=21, color=None):
    if isinstance(node, str):
        add_run(p, node, bold, italic, size, color)
        return
    nb = bold or node.tag in {"strong", "b"}
    ni = italic or node.tag in {"em", "i"}
    if node.tag == "br":
        r = ET.SubElement(p, q("r")); ET.SubElement(r, q("br")); return
    for c in node.children:
        inline(p, c, nb, ni, size, color)


def paragraph(node, kind="p", prefix=""):
    p = ET.Element(q("p"))
    pp = ET.SubElement(p, q("pPr"))
    spacing = {"p": (0, 110), "h1": (0, 140), "h2": (230, 70), "h3": (130, 40), "li": (0, 55)}[kind]
    ET.SubElement(pp, q("spacing"), {q("before"): str(spacing[0]), q("after"): str(spacing[1]), q("line"): "276", q("lineRule"): "auto"})
    if kind == "h1":
        ET.SubElement(pp, q("jc"), {q("val"): "center"})
    if kind == "li":
        ET.SubElement(pp, q("ind"), {q("left"): "360", q("hanging"): "180"})
    if kind == "h1": size, bold, color = 36, True, "17365D"
    elif kind == "h2": size, bold, color = 28, True, "17365D"
    elif kind == "h3": size, bold, color = 23, True, "365F91"
    else: size, bold, color = 21, False, None
    if prefix:
        add_run(p, prefix, bold=False, size=size)
    inline(p, node, bold=bold, size=size, color=color)
    return p


def shade(cell, fill):
    cp = cell.find(q("tcPr"))
    if cp is None:
        cp = ET.SubElement(cell, q("tcPr"))
    ET.SubElement(cp, q("shd"), {q("fill"): fill, q("val"): "clear"})


def make_table(node):
    tbl = ET.Element(q("tbl"))
    pr = ET.SubElement(tbl, q("tblPr"))
    ET.SubElement(pr, q("tblW"), {q("w"): "0", q("type"): "auto"})
    borders = ET.SubElement(pr, q("tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        ET.SubElement(borders, q(edge), {q("val"): "single", q("sz"): "6", q("color"): "AEB9C3"})
    rows = list(find(node, "tr"))
    for ri, row in enumerate(rows):
        tr = ET.SubElement(tbl, q("tr"))
        cells = [c for c in row.children if isinstance(c, Node) and c.tag in {"td", "th"}]
        for cell_node in cells:
            tc = ET.SubElement(tr, q("tc")); ET.SubElement(tc, q("tcPr"))
            is_head = cell_node.tag == "th"
            if is_head: shade(tc, "365F91")
            elif ri % 2: shade(tc, "F2F6FA")
            p = ET.SubElement(tc, q("p")); pp = ET.SubElement(p, q("pPr"))
            ET.SubElement(pp, q("spacing"), {q("after"): "30"})
            inline(p, cell_node, bold=is_head, size=19, color="FFFFFF" if is_head else None)
    return tbl


def main():
    root_dir = Path(__file__).resolve().parent
    html = (root_dir / "H2R_CoAssembly_Project_Proposal.html").read_text(encoding="utf-8")
    parser = Parser(); parser.feed(html)
    body_html = next(find(parser.root, "body"))
    template = Path("/home/skim3674/Downloads/Project Proposal Template.docx")
    output = root_dir / "H2R_CoAssembly_Project_Proposal.docx"

    with ZipFile(template) as zin:
        document = ET.fromstring(zin.read("word/document.xml"))
        body = document.find(q("body"))
        sect = deepcopy(body.find(q("sectPr")))
        for c in list(body): body.remove(c)
        for child in body_html.children:
            if not isinstance(child, Node): continue
            if child.tag in {"h1", "h2", "h3", "p"}:
                body.append(paragraph(child, child.tag))
            elif child.tag in {"ul", "ol"}:
                n = 1
                for li in [x for x in child.children if isinstance(x, Node) and x.tag == "li"]:
                    prefix = f"{n}. " if child.tag == "ol" else "• "
                    body.append(paragraph(li, "li", prefix)); n += 1
            elif child.tag == "table":
                body.append(make_table(child))
            elif child.tag == "div":
                classes = child.attrs.get("class", "")
                if "subtitle" in classes:
                    p = paragraph(child, "p"); p.find(q("pPr")).append(ET.Element(q("jc"), {q("val"): "center"})); body.append(p)
                elif "refs" in classes:
                    for pnode in [x for x in child.children if isinstance(x, Node) and x.tag == "p"]:
                        body.append(paragraph(pnode, "p"))
                elif "note" in classes:
                    body.append(paragraph(child, "p"))
        if sect is not None: body.append(sect)
        xml = ET.tostring(document, encoding="utf-8", xml_declaration=True)
        with ZipFile(output, "w", ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                zout.writestr(item, xml if item.filename == "word/document.xml" else zin.read(item.filename))
    print(output)


if __name__ == "__main__":
    main()
