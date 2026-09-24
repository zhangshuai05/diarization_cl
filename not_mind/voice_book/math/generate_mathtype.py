"""Generate textbook equation assets with the locally installed MathType.

The input side uses MathType's ``Equation.DSMT4`` IDataObject.  This is the
same conversion path used by the official SDK examples: a TeX or MathML
clipboard format is converted to binary MTEF, then ``MTXFormEqn`` renders that
MTEF to a GIF/WMF file.  The COM calls are implemented with ctypes so this
script does not require pywin32 or comtypes.

Usage (from the repository root)::

    python book/math/generate_mathtype.py
    python book/math/generate_mathtype.py --ids cer ctc_path

MathType itself, including a valid local installation/license, is required.
The script fails loudly when MathType or a conversion step is unavailable; it
never substitutes a normal text rendering while claiming that MathType was
used.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid


MATH_DIR = Path(__file__).resolve().parent
ROOT = MATH_DIR.parents[1]
SOURCE_FILE = MATH_DIR / "formula_sources.json"
ASSET_DIR = MATH_DIR / "assets"
MANIFEST_FILE = MATH_DIR / "formula_manifest.json"
# MathType's GIF exporter defaults to a screen-resolution bitmap.  Keep that
# GIF as the archival source, but supersample the PNG companion before it is
# embedded in the PDF so zooming does not expose the 96-dpi pixel grid.
PNG_SCALE = 4

MT_OK = 0
MT_FILE_NOT_FOUND = -8
MT_MEMORY = -9
MT_BAD_PATH = -16
MT_FILE_ACCESS = -17
MT_FILE_WRITE_ERROR = -18
MT_TRANSLATOR_ERROR = -14

MTXFM_LOCAL = -3
MTXFM_FILE = -4
MTXFM_MTEF = 4
MTXFM_GIF = 9
MTXFM_EPS_NONE = 10
MTXFM_EPS_WMF = 11

TYMED_HGLOBAL = 1
TYMED_ISTREAM = 4
TYMED_ISTORAGE = 8
DVASPECT_CONTENT = 1
GMEM_MOVEABLE = 0x0002
GMEM_ZEROINIT = 0x0040
COINIT_APARTMENTTHREADED = 0x2
# COM's enum values.  MathType 7 registers Equation.DSMT4 as a local server;
# allowing both contexts also covers installations that register a proxy DLL.
CLSCTX_INPROC_SERVER = 0x1
CLSCTX_LOCAL_SERVER = 0x4

CLSID_EQUATION_DSMT4 = "{0002CE03-0000-0000-C000-000000000046}"
IID_IDATA_OBJECT = "{0000010E-0000-0000-C000-000000000046}"


class MathTypeError(RuntimeError):
    """A conversion failed and needs user action."""


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", wintypes.BYTE * 8),
    ]


class _FORMATETC(ctypes.Structure):
    # The natural Windows alignment is important on 64-bit Python: ptd starts
    # at offset 8 and the whole structure is 32 bytes.
    _fields_ = [
        ("cfFormat", wintypes.WORD),
        ("ptd", ctypes.c_void_p),
        ("dwAspect", wintypes.DWORD),
        ("lindex", wintypes.LONG),
        ("tymed", wintypes.DWORD),
    ]


class _STGMEDIUM(ctypes.Structure):
    # The union is represented by one pointer.  MathType currently returns an
    # HGLOBAL even when TYMED_ISTORAGE is requested (as in the SDK sample).
    _fields_ = [
        ("tymed", wintypes.DWORD),
        ("unionmember", ctypes.c_void_p),
        ("pUnkForRelease", ctypes.c_void_p),
    ]


class _RECT(ctypes.Structure):
    # MathType's MTAPI_DIMS uses 64-bit LONG values in the SDK declaration.
    _fields_ = [
        ("left", ctypes.c_longlong),
        ("top", ctypes.c_longlong),
        ("right", ctypes.c_longlong),
        ("bottom", ctypes.c_longlong),
    ]


class _DIMS(ctypes.Structure):
    _fields_ = [("baseline", ctypes.c_int), ("bounds", _RECT)]


def _guid(value: str) -> _GUID:
    return _GUID.from_buffer_copy(uuid.UUID(value).bytes_le)


def _u32(value: int) -> int:
    return ctypes.c_uint32(value).value


def _hr_text(value: int) -> str:
    unsigned = _u32(value)
    return f"0x{unsigned:08X}"


def _find_dll() -> Path:
    configured = os.environ.get("MATHTYPE_DLL")
    candidates = [Path(configured)] if configured else []
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    for base in (program_files_x86, program_files):
        root = Path(base) / "MathType"
        candidates.extend([root / "System" / "64" / "MT6.dll", root / "System" / "MT6.dll"])
    for path in candidates:
        if path and path.exists():
            return path
    details = "\n".join(f"  {p}" for p in candidates)
    raise MathTypeError(
        "MathType MT6.dll was not found. Install MathType 7 or set MATHTYPE_DLL. "
        f"Checked:\n{details}"
    )


class _ComDataObject:
    """Small ctypes wrapper around MathType's IDataObject conversion server."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise MathTypeError("MathType equation conversion requires Windows COM.")
        self.ole32 = ctypes.WinDLL("ole32.dll")
        self.user32 = ctypes.WinDLL("user32.dll")
        self.kernel32 = ctypes.WinDLL("kernel32.dll")
        self._configure_system_calls()
        self._ptr = ctypes.c_void_p()
        self._coinited = False
        self._open()

    def _configure_system_calls(self) -> None:
        hresult = ctypes.c_long
        self.ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        self.ole32.CoInitializeEx.restype = hresult
        self.ole32.CoUninitialize.argtypes = []
        self.ole32.CoUninitialize.restype = None
        self.ole32.CoCreateInstance.argtypes = [
            ctypes.POINTER(_GUID), ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p),
        ]
        self.ole32.CoCreateInstance.restype = hresult
        self.ole32.ReleaseStgMedium.argtypes = [ctypes.POINTER(_STGMEDIUM)]
        self.ole32.ReleaseStgMedium.restype = None
        self.user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
        self.user32.RegisterClipboardFormatW.restype = wintypes.WORD
        self.kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        self.kernel32.GlobalAlloc.restype = ctypes.c_void_p
        self.kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        self.kernel32.GlobalLock.restype = ctypes.c_void_p
        self.kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        self.kernel32.GlobalUnlock.restype = wintypes.BOOL
        self.kernel32.GlobalSize.argtypes = [ctypes.c_void_p]
        self.kernel32.GlobalSize.restype = ctypes.c_size_t
        self.kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
        self.kernel32.GlobalFree.restype = ctypes.c_void_p

    def _open(self) -> None:
        hr = self.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        # S_FALSE means COM was already initialized on this thread and is
        # still balanced by one CoUninitialize call.  RPC_E_CHANGED_MODE is
        # safe to continue with when the caller initialized COM differently.
        if _u32(hr) not in (0, 1, 0x80010106):
            raise MathTypeError(f"CoInitializeEx failed ({_hr_text(hr)}).")
        self._coinited = _u32(hr) in (0, 1)
        clsid = _guid(CLSID_EQUATION_DSMT4)
        iid = _guid(IID_IDATA_OBJECT)
        hr = self.ole32.CoCreateInstance(
            ctypes.byref(clsid), None,
            CLSCTX_INPROC_SERVER | CLSCTX_LOCAL_SERVER,
            ctypes.byref(iid), ctypes.byref(self._ptr),
        )
        if _u32(hr) != 0 or not self._ptr.value:
            self.close()
            raise MathTypeError(
                "Could not create MathType Equation.DSMT4 IDataObject "
                f"(HRESULT {_hr_text(hr)}). Is MathType installed and licensed?"
            )
        table = ctypes.cast(self._ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        self._release = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(table[2])
        self._get_data = ctypes.WINFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_FORMATETC),
            ctypes.POINTER(_STGMEDIUM),
        )(table[3])
        self._set_data = ctypes.WINFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_FORMATETC),
            ctypes.POINTER(_STGMEDIUM), wintypes.BOOL,
        )(table[7])
        self._formats: dict[str, int] = {}

    def close(self) -> None:
        if getattr(self, "_ptr", None) is not None and self._ptr.value:
            self._release(self._ptr)
            self._ptr = ctypes.c_void_p()
        if getattr(self, "_coinited", False):
            self.ole32.CoUninitialize()
            self._coinited = False

    def __enter__(self) -> "_ComDataObject":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _format(self, name: str) -> int:
        if name not in self._formats:
            value = int(self.user32.RegisterClipboardFormatW(name))
            if value == 0:
                raise MathTypeError(f"RegisterClipboardFormatW failed for {name!r}.")
            self._formats[name] = value
        return self._formats[name]

    def _set_text(self, format_name: str, text: str) -> None:
        raw = text.encode("utf-16le") + b"\0\0"
        handle = self.kernel32.GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT, len(raw))
        if not handle:
            raise MathTypeError("GlobalAlloc failed while preparing equation input.")
        try:
            locked = self.kernel32.GlobalLock(handle)
            if not locked:
                raise MathTypeError("GlobalLock failed while preparing equation input.")
            ctypes.memmove(locked, raw, len(raw))
            self.kernel32.GlobalUnlock(handle)
            fmt = _FORMATETC(self._format(format_name), None, DVASPECT_CONTENT, -1, TYMED_HGLOBAL)
            medium = _STGMEDIUM(TYMED_HGLOBAL, handle, None)
            hr = self._set_data(self._ptr, ctypes.byref(fmt), ctypes.byref(medium), False)
            if _u32(hr) != 0:
                raise MathTypeError(
                    f"MathType rejected {format_name} input (HRESULT {_hr_text(hr)})."
                )
        finally:
            # fRelease=False tells the server to copy the data.
            self.kernel32.GlobalFree(handle)

    def to_mtef(self, source_type: str, source: str) -> bytes:
        if source_type == "tex":
            # The input translator accepts both $...$ and $$...$$.  Wrapping
            # plain snippets also makes simple equalities/subscripts reliable.
            text = source.strip()
            if not (text.startswith("$") and text.endswith("$")):
                text = f"${text}$"
            self._set_text("TeX Input Language", text)
        elif source_type == "mathml":
            text = source.strip()
            if not text.lower().startswith("<math"):
                text = f"<math>{text}</math>"
            self._set_text("MathML", text)
        else:
            raise MathTypeError(f"Unsupported MathType source_type: {source_type!r}")

        fmt = _FORMATETC(
            self._format("MathType EF"), None, DVASPECT_CONTENT, -1, TYMED_ISTORAGE
        )
        medium = _STGMEDIUM()
        hr = self._get_data(self._ptr, ctypes.byref(fmt), ctypes.byref(medium))
        if _u32(hr) != 0:
            raise MathTypeError(
                f"MathType could not convert {source_type} to MTEF "
                f"(HRESULT {_hr_text(hr)})."
            )
        try:
            if medium.tymed != TYMED_HGLOBAL or not medium.unionmember:
                raise MathTypeError(
                    "MathType returned an unsupported storage medium "
                    f"(tymed={medium.tymed}); expected HGLOBAL."
                )
            size = int(self.kernel32.GlobalSize(medium.unionmember))
            if size <= 0:
                raise MathTypeError("MathType returned an empty MTEF equation.")
            locked = self.kernel32.GlobalLock(medium.unionmember)
            if not locked:
                raise MathTypeError("GlobalLock failed for MathType's MTEF result.")
            try:
                data = ctypes.string_at(locked, size)
            finally:
                self.kernel32.GlobalUnlock(medium.unionmember)
        finally:
            self.ole32.ReleaseStgMedium(ctypes.byref(medium))
        if len(data) < 8 or data[:2] != b"\x05\x01":
            raise MathTypeError("MathType returned bytes that do not look like MTEF.")
        return data


class _MathTypeRenderer:
    def __init__(self, dll_path: Path) -> None:
        self.dll_path = dll_path
        # MathType's DLL loads companion files from its own directory.
        self._dll_dir_cookie = None
        if hasattr(os, "add_dll_directory"):
            self._dll_dir_cookie = os.add_dll_directory(str(dll_path.parent))
        try:
            self.dll = ctypes.WinDLL(str(dll_path))
        except OSError as exc:
            if self._dll_dir_cookie:
                self._dll_dir_cookie.close()
                self._dll_dir_cookie = None
            raise MathTypeError(f"Could not load MathType DLL {dll_path}: {exc}") from exc
        self.dll.MTAPIConnect.argtypes = [ctypes.c_short, ctypes.c_short]
        self.dll.MTAPIConnect.restype = ctypes.c_int
        self.dll.MTAPIDisconnect.argtypes = []
        self.dll.MTAPIDisconnect.restype = ctypes.c_int
        self.dll.MTXFormReset.argtypes = []
        self.dll.MTXFormReset.restype = ctypes.c_int
        self.dll.MTXFormEqn.argtypes = [
            ctypes.c_short, ctypes.c_short, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_short, ctypes.c_short, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.POINTER(_DIMS),
        ]
        self.dll.MTXFormEqn.restype = ctypes.c_int
        self.connected = False

    def connect(self) -> None:
        status = int(self.dll.MTAPIConnect(0, 30))
        if status != MT_OK:
            raise MathTypeError(
                f"MTAPIConnect failed with {status} ({_hr_text(status)}). "
                "Check the MathType installation/license."
            )
        self.connected = True

    def close(self) -> None:
        if self.connected:
            self.dll.MTAPIDisconnect()
            self.connected = False
        if self._dll_dir_cookie:
            self._dll_dir_cookie.close()
            self._dll_dir_cookie = None

    def __enter__(self) -> "_MathTypeRenderer":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def render(self, mtef: bytes, destination: Path, output_format: int) -> None:
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = ctypes.create_string_buffer(mtef)
        dims = _DIMS()
        reset = int(self.dll.MTXFormReset())
        if reset != MT_OK:
            raise MathTypeError(f"MTXFormReset failed ({reset}).")
        status = int(self.dll.MTXFormEqn(
            MTXFM_LOCAL, MTXFM_MTEF,
            ctypes.cast(source, ctypes.c_void_p), len(mtef),
            MTXFM_FILE, output_format,
            None, 0, os.fsencode(destination), ctypes.byref(dims),
        ))
        if status != MT_OK or not destination.exists() or destination.stat().st_size == 0:
            raise MathTypeError(
                f"MTXFormEqn failed for {destination.name}: status={status}; "
                f"path={destination}; file_exists={destination.exists()}"
            )


def _load_sources(path: Path) -> list[dict[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MathTypeError(f"Could not read formula source file {path}: {exc}") from exc
    if not isinstance(data, list):
        raise MathTypeError("formula_sources.json must contain a list.")
    required = {"id", "source_type", "source"}
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict) or not required.issubset(item):
            raise MathTypeError(f"Invalid formula source entry: {item!r}")
        key = str(item["id"])
        if key in seen:
            raise MathTypeError(f"Duplicate formula id: {key}")
        seen.add(key)
    return data


def _gif_to_png(gif_path: Path, png_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise MathTypeError(
            "Pillow is needed to create PNG companions; install it or use the GIF assets."
        ) from exc
    with Image.open(gif_path) as image:
        # MathType GIFs use a palette with a white background.  A true-colour
        # supersampled PNG is more predictable when ReportLab embeds it and
        # keeps equation edges smooth at normal PDF zoom levels.
        rgb = image.convert("RGB")
        if PNG_SCALE != 1:
            rgb = rgb.resize(
                (rgb.width * PNG_SCALE, rgb.height * PNG_SCALE),
                Image.Resampling.LANCZOS,
            )
        rgb.save(png_path, format="PNG", optimize=True)


def _validate_image(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image, ImageChops
        with Image.open(path) as image:
            width, height = image.size
            if width < 3 or height < 3:
                raise MathTypeError(f"Generated image is implausibly small: {path}")
            rgb = image.convert("RGB")
            # Reject an all-white image, which usually means the MTEF was not
            # translated or a font/resource failed silently.
            extrema = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
            if extrema is None:
                raise MathTypeError(f"Generated image is blank: {path}")
            return width, height
    except MathTypeError:
        raise
    except Exception as exc:
        raise MathTypeError(f"Could not inspect generated image {path}: {exc}") from exc


def generate(entries: list[dict[str, str]], output_dir: Path = ASSET_DIR) -> dict:
    """Convert source entries and return a JSON-serialisable manifest."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dll_path = _find_dll()
    generated: dict[str, dict] = {}
    # Equation.DSMT4 keeps an editable equation as state.  Opening a fresh
    # IDataObject for each entry prevents MathType from appending the next
    # source to the previous equation in a batch conversion.
    with _MathTypeRenderer(dll_path) as renderer:
        for entry in entries:
            key = str(entry["id"])
            source_type = str(entry["source_type"]).lower()
            source = str(entry["source"])
            print(f"[mathtype] {key}: {source_type}")
            with _ComDataObject() as converter:
                mtef = converter.to_mtef(source_type, source)
            stem = output_dir / key
            gif = stem.with_suffix(".gif")
            wmf = stem.with_suffix(".wmf")
            png = stem.with_suffix(".png")
            renderer.render(mtef, gif, MTXFM_GIF)
            renderer.render(mtef, wmf, MTXFM_EPS_WMF)
            _gif_to_png(gif, png)
            width, height = _validate_image(png)
            gif_width, gif_height = _validate_image(gif)
            generated[key] = {
                "source_type": source_type,
                "source": source,
                "mtef_bytes": len(mtef),
                "gif": gif.relative_to(ROOT).as_posix(),
                "png": png.relative_to(ROOT).as_posix(),
                "wmf": wmf.relative_to(ROOT).as_posix(),
                "png_size": [width, height],
                "source_gif_size": [gif_width, gif_height],
                "png_scale": PNG_SCALE,
                "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
            }
            print(f"  -> {png.name} ({width}x{height})")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "math_type_dll": str(dll_path),
        "backend": "Equation.DSMT4 IDataObject -> MTEF -> MTXFormEqn",
        "formulas": generated,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", help="Only generate these formula ids.")
    parser.add_argument("--source-file", type=Path, default=SOURCE_FILE)
    parser.add_argument("--output-dir", type=Path, default=ASSET_DIR)
    args = parser.parse_args(argv)
    try:
        entries = _load_sources(args.source_file)
        if args.ids:
            wanted = set(args.ids)
            entries = [entry for entry in entries if str(entry["id"]) in wanted]
            missing = wanted - {str(entry["id"]) for entry in entries}
            if missing:
                raise MathTypeError("Unknown formula id(s): " + ", ".join(sorted(missing)))
        if not entries:
            raise MathTypeError("No formula entries selected.")
        manifest = generate(entries, args.output_dir)
        print(f"generated {len(manifest['formulas'])} MathType formula(s)")
        return 0
    except MathTypeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
