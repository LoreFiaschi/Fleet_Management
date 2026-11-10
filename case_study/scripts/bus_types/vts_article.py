"""Print the text of one article of the Fedlex AkomaNtoso XML (VTS).
Usage: python -E vts_article.py file.xml art_7"""
import sys
import re
import xml.etree.ElementTree as ET

tree = ET.parse(sys.argv[1])
root = tree.getroot()
want = sys.argv[2]
for el in root.iter():
    if el.tag.endswith("article") and el.attrib.get("eId") == want:
        text = " ".join(t.strip() for t in el.itertext() if t.strip())
        text = re.sub(r"\s+", " ", text)
        print(text)
        break
else:
    ids = [el.attrib.get("eId") for el in root.iter() if el.tag.endswith("article")]
    print("not found; first ids:", ids[:20])
