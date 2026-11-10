"""Convert a downloaded HTML file to plain text (stdlib only).
Usage: python -I html2txt.py <in.html> <out.txt>
Prints the number of characters written."""
import sys
from html.parser import HTMLParser

SKIP = {"script", "style", "noscript", "svg"}
BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section",
         "article", "pre", "dt", "dd", "td", "th", "blockquote"}


class P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
        if tag in BLOCK:
            self.out.append("\n")
        if tag in ("td", "th"):
            self.out.append(" | ")
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v and v.startswith("http"):
                    self.out.append(" ")
                    self._href = v

    def handle_endtag(self, tag):
        if tag in SKIP and self.skip:
            self.skip -= 1
        if tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


raw = open(sys.argv[1], "rb").read()
for enc in ("utf-8", "cp1252", "latin-1"):
    try:
        html = raw.decode(enc)
        break
    except UnicodeDecodeError:
        continue
p = P()
p.feed(html)
text = "".join(p.out)
lines = [" ".join(l.split()) for l in text.splitlines()]
clean = []
for l in lines:
    if l or (clean and clean[-1]):
        clean.append(l)
res = "\n".join(clean)
open(sys.argv[2], "w", encoding="utf-8", newline="\n").write(res)
print(len(res), "chars,", res.count("\n"), "lines")
