"""
Công cụ chuẩn hóa văn bản + đối chiếu giá trị bảo vệ, dùng chung cho
input guardrail, output guardrail và egress (pipeline).

Ý tưởng chính: kẻ tấn công hay giấu nội dung bằng Unicode lạ, ký tự vô hình,
cách viết tách chữ, base64/hex/ROT13/đảo chuỗi. Thay vì viết regex cho từng
trường hợp, ta tạo ra nhiều "phiên bản đã giải mã" của câu rồi chạy cùng một
bộ luật trên tất cả các phiên bản đó.

Không dùng LLM trong file này: toàn bộ là luật xác định.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import html
import re
import unicodedata
from functools import lru_cache
from urllib.parse import unquote

from core.config import DEMO_SECRETS, load_protected_payload

# ---------------------------------------------------------------------------
# Chuẩn hóa
# ---------------------------------------------------------------------------

# Một số ký tự trông giống chữ Latin (Cyrillic/Greek) mà NFKC không đổi.
_CONFUSABLES = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
        "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "к": "k", "м": "m",
        "н": "h", "т": "t", "в": "b", "ѵ": "v", "ո": "n",
        "α": "a", "ε": "e", "ο": "o", "ρ": "p", "ι": "i", "κ": "k", "ν": "v",
        "τ": "t", "υ": "u", "χ": "x", "β": "b", "η": "n", "μ": "u",
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
        "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ѕ": "S",
        "ı": "i", "ſ": "s",
    }
)

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def strip_invisible(text: str) -> str:
    """Bỏ ký tự định dạng/vô hình (zero-width, bidi, soft hyphen, tag chars...)."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cf", "Co", "Cs"):
            continue
        if cat == "Cc" and ch not in "\n\t\r":
            continue
        out.append(ch)
    return "".join(out)


def _collapse_spaces(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text)


def normalize(text: str, *, collapse: bool = True) -> str:
    """NFKC + bỏ ký tự vô hình + đổi ký tự giả dạng Latin + gộp khoảng trắng."""
    text = unicodedata.normalize("NFKC", text or "")
    text = strip_invisible(text)
    text = text.translate(_CONFUSABLES)
    text = re.sub(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]", " ", text)
    return _collapse_spaces(text) if collapse else text


def fold_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt / dấu Latin: 'Bỏ qua hướng dẫn' -> 'Bo qua huong dan'."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def squash(text: str) -> str:
    """Chỉ giữ chữ + số, viết thường: so khớp bất chấp dấu câu / khoảng trắng."""
    return re.sub(r"[^a-z0-9]", "", fold_accents(normalize(text)).casefold())


def _join_spaced_letters(text: str) -> str:
    """'i g n o r e' / 'i.g.n.o.r.e' -> 'ignore' (chuỗi ≥4 ký tự đơn tách nhau)."""

    def _join(m: re.Match) -> str:
        return re.sub(r"[ .\-_*|/]", "", m.group(0))

    # Dấu tách là MỘT ký tự; hai khoảng trắng liền nhau được coi là ranh giới giữa các từ.
    return re.sub(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9][ .\-_*|/]){3,}[A-Za-z0-9](?![A-Za-z0-9])", _join, text)


def _words_to_digits(text: str) -> str:
    def _sub(m: re.Match) -> str:
        return _NUM_WORDS[m.group(0).lower()]

    return re.sub(r"\b(?:" + "|".join(_NUM_WORDS) + r")\b", _sub, text, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Giải mã các kiểu che giấu
# ---------------------------------------------------------------------------

_B64_RE = re.compile(r"[A-Za-z0-9+/_-]{8,}={0,2}")
_HEX_RE = re.compile(r"(?:\\x|0x)?[0-9a-fA-F]{2}(?:[\s:,\-]?(?:\\x|0x)?[0-9a-fA-F]{2}){5,}")
_BIN_RE = re.compile(r"(?:[01]{8}\s?){4,}")


def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    ok = sum(1 for c in s if c.isprintable() or c in "\n\t")
    return ok / len(s)


def _try_b64(token: str) -> str | None:
    tok = token.strip("=")
    for alt in (False, True):
        t = tok.replace("-", "+").replace("_", "/") if alt else tok
        pad = "=" * (-len(t) % 4)
        for decoder in (base64.b64decode, base64.b32decode):
            try:
                raw = decoder((t + pad).upper().encode() if decoder is base64.b32decode else (t + pad).encode())
            except (binascii.Error, ValueError):
                continue
            s = raw.decode("utf-8", errors="ignore")
            if len(s) >= 4 and _printable_ratio(s) > 0.85:
                return s
    return None


def _try_hex(token: str) -> str | None:
    cleaned = re.sub(r"\\x|0x|[\s:,\-]", "", token)
    if len(cleaned) % 2 or len(cleaned) < 8:
        return None
    try:
        s = bytes.fromhex(cleaned).decode("utf-8", errors="ignore")
    except ValueError:
        return None
    return s if len(s) >= 4 and _printable_ratio(s) > 0.85 else None


def _try_bin(token: str) -> str | None:
    bits = re.sub(r"\s", "", token)
    if len(bits) % 8:
        return None
    try:
        s = bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8)).decode("utf-8", errors="ignore")
    except ValueError:
        return None
    return s if len(s) >= 4 and _printable_ratio(s) > 0.85 else None


def decoded_fragments(text: str) -> list[str]:
    """Các đoạn văn bản được giải mã từ base64/base32/hex/binary/URL/HTML entity."""
    frags: list[str] = []
    for m in _B64_RE.finditer(text):
        d = _try_b64(m.group(0))
        if d:
            frags.append(d)
    for m in _HEX_RE.finditer(text):
        d = _try_hex(m.group(0))
        if d:
            frags.append(d)
    for m in _BIN_RE.finditer(text):
        d = _try_bin(m.group(0))
        if d:
            frags.append(d)
    if "%" in text:
        u = unquote(text)
        if u != text:
            frags.append(u)
    if "&" in text:
        h = html.unescape(text)
        if h != text:
            frags.append(h)
    return frags


def text_variants(text: str, *, depth: int = 2) -> list[str]:
    """Trả về danh sách phiên bản của ``text`` để chạy luật phát hiện.

    Bao gồm: bản chuẩn hóa, bỏ dấu, gỡ leetspeak, ghép chữ tách rời, các đoạn
    giải mã (base64/hex/...), ROT13 và chuỗi đảo ngược của chúng (giải mã lồng
    nhau tối đa ``depth`` tầng).
    """
    seen: set[str] = set()
    out: list[str] = []

    def add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    def expand(raw: str, level: int) -> None:
        base = normalize(raw)
        folded = fold_accents(base)
        add(base)
        add(folded)
        # ghép chữ tách rời trên bản CHƯA gộp khoảng trắng để còn thấy ranh giới từ
        joined = _collapse_spaces(_join_spaced_letters(fold_accents(normalize(raw, collapse=False))))
        add(joined)
        add(_words_to_digits(joined))
        add(folded.translate(_LEET))
        # ROT13 và đảo chuỗi (chỉ hữu ích khi ghép với luật cụm từ mạnh)
        add(codecs.decode(folded, "rot_13"))
        add(folded[::-1])
        add(" ".join(w[::-1] for w in folded.split()))
        if level < depth:
            for frag in decoded_fragments(base):
                expand(frag, level + 1)

    expand(text or "", 0)
    return out


# ---------------------------------------------------------------------------
# Giá trị được bảo vệ (secrets) — đọc từ data/protected/vinbank_secrets.json
# ---------------------------------------------------------------------------

def _load_secret_values() -> dict[str, list[str]]:
    """Gom các giá trị bảo vệ theo loại: password / api_key / internal_host."""
    groups: dict[str, list[str]] = {"password": [], "api_key": [], "internal_host": []}
    try:
        payload = load_protected_payload()
    except FileNotFoundError:
        payload = {}
    secrets = payload.get("secrets") or {}
    kind_of = {"admin_password": "password", "api_key": "api_key", "db_host": "internal_host"}
    for key, kind in kind_of.items():
        val = (secrets.get(key) or "").strip()
        if val:
            groups[kind].append(val)
            if kind == "internal_host" and ":" in val:
                groups[kind].append(val.split(":", 1)[0])
    for target in payload.get("leak_targets") or []:
        kind = kind_of.get(target.get("id"), "")
        if not kind:
            continue
        for s in [target.get("value", "")] + list(target.get("match_substrings") or []):
            s = (s or "").strip()
            if s and s not in groups[kind]:
                groups[kind].append(s)
    # Dự phòng nếu file thiếu: dùng DEMO_SECRETS đã nạp trong config
    if not any(groups.values()):
        groups["password"] = list(DEMO_SECRETS)
    return groups


@lru_cache(maxsize=1)
def secret_groups() -> dict[str, tuple[str, ...]]:
    raw = _load_secret_values()
    return {kind: tuple(vals) for kind, vals in raw.items()}


@lru_cache(maxsize=1)
def secret_needles() -> tuple[tuple[str, str], ...]:
    """[(kind, needle đã squash)] — bỏ needle quá ngắn để tránh trùng ngẫu nhiên."""
    needles: list[tuple[str, str]] = []
    for kind, vals in secret_groups().items():
        for v in vals:
            for sq in (squash(v), squash(v.translate(_LEET))):  # dạng gốc + dạng leetspeak
                if len(sq) >= 6 and (kind, sq) not in needles:
                    needles.append((kind, sq))
    return tuple(needles)


def find_protected_kinds(text: str) -> set[str]:
    """Loại giá trị bảo vệ xuất hiện trong ``text`` (kể cả bị mã hóa / tách ký tự).

    Dò trên mọi phiên bản của câu: gộp bỏ dấu câu, chữ số viết bằng chữ,
    base64/hex/ROT13/đảo ngược. Trả về tập {"password","api_key","internal_host"}.
    """
    found: set[str] = set()
    if not text:
        return found
    needles = secret_needles()
    for variant in text_variants(text):
        sq = squash(variant)
        if not sq:
            continue
        for kind, needle in needles:
            if needle in sq:
                found.add(kind)
        # chuỗi đảo ngược sau khi gộp (đảo cả câu rồi mới bỏ dấu câu)
        sq_rev = sq[::-1]
        for kind, needle in needles:
            if needle in sq_rev:
                found.add(kind)
    return found


def contains_protected_value(text: str) -> bool:
    return bool(find_protected_kinds(text))


def plain_secret_spans(text: str) -> list[re.Match]:
    """Vị trí xuất hiện nguyên văn (không mã hóa) của giá trị bảo vệ — để che đúng chỗ."""
    spans: list[re.Match] = []
    for vals in secret_groups().values():
        for v in vals:
            if len(v) >= 6:
                spans.extend(re.finditer(re.escape(v), text, re.IGNORECASE))
    return spans
