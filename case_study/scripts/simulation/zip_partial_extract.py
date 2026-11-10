"""Extract chosen members from the first bytes of a zip (local headers + deflate data).
Usage: python -I zip_partial_extract.py <head.bin> <out_dir> name:csize:offset [...]"""
import os, struct, sys, zlib
data = open(sys.argv[1], "rb").read()
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
for spec in sys.argv[3:]:
    name, csize, off = spec.split(":")
    csize, off = int(csize), int(off)
    sig, _, _, method, _, _, _, _, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", data[off:off + 30])
    assert sig == 0x04034B50, name
    start = off + 30 + nlen + xlen
    raw = data[start:start + csize]
    assert len(raw) == csize, "member %s is not complete in the chunk" % name
    buf = zlib.decompressobj(-15).decompress(raw) if method == 8 else raw
    open(os.path.join(out, name), "wb").write(buf)
    print(name, len(buf), "bytes,", buf.count(b"\n"), "lines")
