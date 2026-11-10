"""List the members of a zip from a downloaded tail chunk (central directory) or a whole zip.
Usage: python -I zip_listing.py <tail.bin> <total_size_of_remote_zip>
       python -I zip_listing.py <whole.zip>
Prints name, compressed size, uncompressed size, local header offset."""
import struct
import sys
import zipfile

if len(sys.argv) == 2:
    with zipfile.ZipFile(sys.argv[1]) as z:
        for i in z.infolist():
            print(i.filename, i.compress_size, i.file_size, i.header_offset, sep="\t")
    sys.exit()

data = open(sys.argv[1], "rb").read()
total = int(sys.argv[2])
base = total - len(data)  # remote offset of data[0]
eocd = data.rfind(b"PK\x05\x06")
assert eocd >= 0, "no end-of-central-directory record in the tail"
(_, _, _, _, n_entries, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", data[eocd:eocd + 22])
# zip64?
z64 = data.rfind(b"PK\x06\x06")
if z64 >= 0:
    n_entries, cd_size, cd_off = struct.unpack("<QQQ", data[z64 + 32:z64 + 56])
pos = cd_off - base
assert pos >= 0, "tail too short: need %d more bytes" % (-pos)
for _ in range(n_entries):
    sig, _, _, _, _, _, _, crc, csize, usize, nlen, xlen, clen, _, _, _, off = struct.unpack(
        "<IHHHHHHIIIHHHHHII", data[pos:pos + 46])
    assert sig == 0x02014B50
    name = data[pos + 46:pos + 46 + nlen].decode("utf-8", "replace")
    extra = data[pos + 46 + nlen:pos + 46 + nlen + xlen]
    # zip64 extra field
    if 0xFFFFFFFF in (csize, usize, off):
        q = 0
        while q < len(extra):
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals = list(struct.unpack("<%dQ" % (hlen // 8), extra[q + 4:q + 4 + hlen]))
                if usize == 0xFFFFFFFF:
                    usize = vals.pop(0)
                if csize == 0xFFFFFFFF:
                    csize = vals.pop(0)
                if off == 0xFFFFFFFF:
                    off = vals.pop(0)
            q += 4 + hlen
    print(name, csize, usize, off, sep="\t")
    pos += 46 + nlen + xlen + clen
