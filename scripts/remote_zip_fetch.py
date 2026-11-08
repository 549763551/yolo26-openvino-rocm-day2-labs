"""Minimal remote-ZIP reader for extracting a few files from a huge HTTP zip.

Strategy (proxy-friendly, few requests):
  1. one ranged GET for the zip tail -> parse EOCD / Zip64 records
  2. one ranged GET for the whole central directory (~tens of MB)
  3. one ranged GET per requested file

Only standard library + curl are used.
"""
from __future__ import annotations

import re
import struct
import subprocess
import zlib
from pathlib import Path

CHUNK = 1024


def _curl_range(url: str, start: int, end: int | None, dest: Path) -> bytes:
    rng = f"{start}-{end if end is not None else ''}"
    r = subprocess.run(
        ["curl", "-L", "-sS", "--retry", "5", "--retry-all-errors",
         "--connect-timeout", "30", "--max-time", "1800", "-r", rng, url],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"range fetch failed ({rng}): {r.stderr.decode()[:200]}")
    return r.stdout


def parse_central_directory(tail: bytes):
    """Return (cd_offset, cd_size) from EOCD/Zip64 records in the tail bytes."""
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("EOCD not found in tail")
    if i + 22 > len(tail):
        raise RuntimeError("truncated EOCD")
    cd_size, cd_offset = struct.unpack("<II", tail[i + 12:i + 20])
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        if j < 0:
            raise RuntimeError("Zip64 EOCD required but missing")
        (sig, _sz) = struct.unpack("<IQ", tail[j:j + 12])
        cd_size, cd_offset = struct.unpack("<QQ", tail[j + 40:j + 56])
    return cd_offset, cd_size


def parse_cd_entries(cd: bytes) -> dict[str, dict]:
    """central directory bytes -> {filename: {offset, comp_size, method}}."""
    entries = {}
    pos = 0
    n = len(cd)
    while pos + 46 <= n and cd[pos:pos + 4] == b"PK\x01\x02":
        (method, _flags) = struct.unpack("<HH", cd[pos + 10:pos + 14])
        comp_size, uncomp_size = struct.unpack("<II", cd[pos + 20:pos + 28])
        name_len, extra_len, comment_len = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        local_offset = struct.unpack("<I", cd[pos + 42:pos + 46])[0]
        name = cd[pos + 46:pos + 46 + name_len].decode("utf-8", "replace")
        extra = cd[pos + 46 + name_len:pos + 46 + name_len + extra_len]
        if 0xFFFFFFFF in (comp_size, uncomp_size, local_offset):
            e = 0
            while e + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[e:e + 4])
                body = extra[e + 4:e + 4 + hsz]
                if hid == 0x0001:
                    p = 0
                    if uncomp_size == 0xFFFFFFFF and p + 8 <= len(body):
                        p += 8
                    if comp_size == 0xFFFFFFFF and p + 8 <= len(body):
                        comp_size = struct.unpack("<Q", body[p:p + 8])[0]
                        p += 8
                    if local_offset == 0xFFFFFFFF and p + 8 <= len(body):
                        local_offset = struct.unpack("<Q", body[p:p + 8])[0]
                    break
                e += 4 + hsz
        entries[name] = {"offset": local_offset, "comp_size": comp_size,
                         "method": method}
        pos += 46 + name_len + extra_len + comment_len
    return entries


def fetch_files(url: str, filenames: list[str], out_dir: Path) -> list[str]:
    """Extract the given files from the remote zip into out_dir.

    Returns list of successfully written files; raises if a file is missing.
    """
    TAIL = 4 * 1024 * 1024
    head = _curl_range(url, 0, CHUNK - 1, out_dir / "_head.tmp")
    del head
    tail_size = None
    # need file size: use a tiny HEAD via curl -I
    r = subprocess.run(["curl", "-sIL", "--max-time", "60", url],
                       capture_output=True)
    m = re.findall(rb"[Cc]ontent-[Ll]ength: (\d+)", r.stdout)
    if not m:
        raise RuntimeError("cannot determine zip size")
    total = int(m[-1])
    tail_size = min(TAIL, total)
    tail = _curl_range(url, total - tail_size, None, out_dir / "_tail.tmp")
    cd_offset, cd_size = parse_central_directory(tail)
    print(f"  zip size {total/1e9:.2f} GB · central dir {cd_size/1e6:.1f} MB "
          f"@ {cd_offset}")
    cd = _curl_range(url, cd_offset, cd_offset + cd_size - 1,
                     out_dir / "_cd.tmp")
    entries = parse_cd_entries(cd)
    by_base = {Path(k).name: k for k in entries}
    print(f"  central directory: {len(entries)} entries")
    written = []
    for fn in filenames:
        ent = entries.get(fn) or entries.get(by_base.get(Path(fn).name, ""))
        if ent is None:
            raise RuntimeError(f"file not in zip: {fn}")
        hdr = _curl_range(url, ent["offset"], ent["offset"] + 512 - 1,
                          out_dir / "_lh.tmp")
        if hdr[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {fn}")
        name_len, extra_len = struct.unpack("<HH", hdr[26:30])
        data_start = ent["offset"] + 30 + name_len + extra_len
        end = data_start + ent["comp_size"] - 1
        blob = _curl_range(url, data_start, end, out_dir / "_data.tmp")
        if ent["method"] == 8:
            blob = zlib.decompress(blob, -15)
        (out_dir / fn).write_bytes(blob)
        written.append(fn)
    for tmp in out_dir.glob("_*.tmp"):
        tmp.unlink()
    return written
