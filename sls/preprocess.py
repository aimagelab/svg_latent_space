import os
import re
import xml.etree.ElementTree as ET
import svgpathtools
from io import StringIO
import json

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_NUM_TRAILING_ZERO_RE = re.compile(r"(-?\d+)\.0(?!\d)")
_NUMBER_WITH_UNIT_RE = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)([A-Za-z%]*)\s*$"
)
_COMMAND_RE = re.compile(r"\s*([MmZzLlHhVvCcSsQqTtAa])\s*")
_COMMA_SPACE_RE = re.compile(r"\s*,\s*")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NUM_SEP_RE = re.compile(r"(?<=\d)\s+(?=[-+]?\d)")
_STYLE_KEYS = ("fill", "stroke", "stroke-width", "opacity")


def _parse_number(value):
    if value is None:
        return None, None
    value = value.strip()
    if value.endswith("%"):
        return float(value[:-1]), "%"
    match = _NUM_RE.search(value)
    if not match:
        return None, None
    return float(match.group(0)), "abs"


def extract_size(svg):
    """
    Return the canvas as (min_x, min_y, width, height).

    Accepts a file path, SVG string, or xml.etree.ElementTree.Element.
    """
    if isinstance(svg, ET.Element):
        root = svg
    else:
        if isinstance(svg, str) and os.path.exists(svg):
            root = ET.parse(svg).getroot()
        else:
            root = ET.fromstring(svg)

    view_box = root.get("viewBox") or root.get("viewbox")
    if view_box:
        nums = _NUM_RE.findall(view_box.replace(",", " "))
        if len(nums) >= 4:
            min_x, min_y, width, height = map(float, nums[:4])
            return (min_x, min_y, width, height)
        if len(nums) == 2:
            width, height = map(float, nums[:2])
            return (0.0, 0.0, width, height)

    width_attr = root.get("width")
    height_attr = root.get("height")
    if width_attr is None or height_attr is None:
        raise ValueError("SVG missing viewBox and width/height attributes.")

    w_val, w_unit = _parse_number(width_attr)
    h_val, h_unit = _parse_number(height_attr)
    if w_val is None or h_val is None:
        raise ValueError("SVG width/height could not be parsed.")
    if w_unit == "%" or h_unit == "%":
        raise ValueError("SVG width/height in % without viewBox.")

    return (0.0, 0.0, float(w_val), float(h_val))


def inherited_style_from_svg(svg):
    """
    Return inheritable root SVG style values that should be made explicit per path.
    """
    if isinstance(svg, ET.Element):
        root = svg
    else:
        if isinstance(svg, str) and os.path.exists(svg):
            root = ET.parse(svg).getroot()
        else:
            root = ET.fromstring(svg)

    style_props = {}
    for key in _STYLE_KEYS:
        val = root.get(key)
        if val is not None and str(val).strip() != "":
            style_props[key] = val
    style_props.update(_parse_style_attr(root.get("style")))
    return style_props


def _round_complex(z):
    return complex(int(round(z.real)), int(round(z.imag)))


def _round_segment(seg):
    if isinstance(seg, svgpathtools.Line):
        return svgpathtools.Line(_round_complex(seg.start), _round_complex(seg.end))
    if isinstance(seg, svgpathtools.CubicBezier):
        return svgpathtools.CubicBezier(
            _round_complex(seg.start),
            _round_complex(seg.control1),
            _round_complex(seg.control2),
            _round_complex(seg.end),
        )
    if isinstance(seg, svgpathtools.QuadraticBezier):
        return svgpathtools.QuadraticBezier(
            _round_complex(seg.start),
            _round_complex(seg.control),
            _round_complex(seg.end),
        )
    if isinstance(seg, svgpathtools.Arc):
        return svgpathtools.Arc(
            _round_complex(seg.start),
            _round_complex(seg.radius),
            seg.rotation,
            seg.large_arc,
            seg.sweep,
            _round_complex(seg.end),
        )
    return seg


def _round_path(p):
    return svgpathtools.Path(*[_round_segment(s) for s in p])


def _strip_trailing_dot_zero(d):
    return _NUM_TRAILING_ZERO_RE.sub(r"\1", d)


def _compact_path_d(d):
    d = _strip_trailing_dot_zero(d)
    d = _COMMAND_RE.sub(r"\1", d)  # remove spaces around commands
    d = _COMMA_SPACE_RE.sub(",", d)  # normalize commas
    d = _NUM_SEP_RE.sub(",", d)  # turn number separators into commas
    d = _MULTI_SPACE_RE.sub(" ", d).strip()
    d = d.replace(" ", "")
    return d


def _format_number(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _scale_stroke_width(value, scale):
    match = _NUMBER_WITH_UNIT_RE.match(str(value))
    if not match:
        return value

    number, unit = match.groups()
    if unit == "%":
        return value

    return f"{_format_number(float(number) * scale)}{unit}"


def _parse_style_attr(style):
    if not style:
        return {}

    props = {}
    for declaration in str(style).split(";"):
        if ":" not in declaration:
            continue
        key, value = declaration.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            props[key] = value
    return props


color_tab_str = '{"aliceblue":"f0f8ff","antiquewhite":"faebd7","aqua":"00ffff","aquamarine":"7fffd4","azure":"f0ffff","beige":"f5f5dc","bisque":"ffe4c4","black":"000000","blanchedalmond":"ffebcd","blue":"0000ff","blueviolet":"8a2be2","brown":"a52a2a","burlywood":"deb887","cadetblue":"5f9ea0","chartreuse":"7fff00","chocolate":"d2691e","coral":"ff7f50","cornflowerblue":"6495ed","cornsilk":"fff8dc","crimson":"dc143c","cyan":"00ffff","darkblue":"00008b","darkcyan":"008b8b","darkgoldenrod":"b8860b","darkgray":"a9a9a9","darkgreen":"006400","darkgrey":"a9a9a9","darkkhaki":"bdb76b","darkmagenta":"8b008b","darkolivegreen":"556b2f","darkorange":"ff8c00","darkorchid":"9932cc","darkred":"8b0000","darksalmon":"e9967a","darkseagreen":"8fbc8f","darkslateblue":"483d8b","darkslategray":"2f4f4f","darkslategrey":"2f4f4f","darkturquoise":"00ced1","darkviolet":"9400d3","deeppink":"ff1493","deepskyblue":"00bfff","dimgray":"696969","dimgrey":"696969","dodgerblue":"1e90ff","firebrick":"b22222","floralwhite":"fffaf0","forestgreen":"228b22","fuchsia":"ff00ff","gainsboro":"dcdcdc","ghostwhite":"f8f8ff","gold":"ffd700","goldenrod":"daa520","gray":"808080","green":"008000","greenyellow":"adff2f","grey":"808080","honeydew":"f0fff0","hotpink":"ff69b4","indianred":"cd5c5c","indigo":"4b0082","ivory":"fffff0","khaki":"f0e68c","lavender":"e6e6fa","lavenderblush":"fff0f5","lawngreen":"7cfc00","lemonchiffon":"fffacd","lightblue":"add8e6","lightcoral":"f08080","lightcyan":"e0ffff","lightgoldenrodyellow":"fafad2","lightgray":"d3d3d3","lightgreen":"90ee90","lightgrey":"d3d3d3","lightpink":"ffb6c1","lightsalmon":"ffa07a","lightseagreen":"20b2aa","lightskyblue":"87cefa","lightslategray":"778899","lightslategrey":"778899","lightsteelblue":"b0c4de","lightyellow":"ffffe0","lime":"00ff00","limegreen":"32cd32","linen":"faf0e6","magenta":"ff00ff","maroon":"800000","mediumaquamarine":"66cdaa","mediumblue":"0000cd","mediumorchid":"ba55d3","mediumpurple":"9370db","mediumseagreen":"3cb371","mediumslateblue":"7b68ee","mediumspringgreen":"00fa9a","mediumturquoise":"48d1cc","mediumvioletred":"c71585","midnightblue":"191970","mintcream":"f5fffa","mistyrose":"ffe4e1","moccasin":"ffe4b5","navajowhite":"ffdead","navy":"000080","oldlace":"fdf5e6","olive":"808000","olivedrab":"6b8e23","orange":"ffa500","orangered":"ff4500","orchid":"da70d6","palegoldenrod":"eee8aa","palegreen":"98fb98","paleturquoise":"afeeee","palevioletred":"db7093","papayawhip":"ffefd5","peachpuff":"ffdab9","peru":"cd853f","pink":"ffc0cb","plum":"dda0dd","powderblue":"b0e0e6","purple":"800080","red":"ff0000","rosybrown":"bc8f8f","royalblue":"4169e1","saddlebrown":"8b4513","salmon":"fa8072","sandybrown":"f4a460","seagreen":"2e8b57","seashell":"fff5ee","sienna":"a0522d","silver":"c0c0c0","skyblue":"87ceeb","slateblue":"6a5acd","slategray":"708090","slategrey":"708090","snow":"fffafa","springgreen":"00ff7f","steelblue":"4682b4","tan":"d2b48c","teal":"008080","thistle":"d8bfd8","tomato":"ff6347","turquoise":"40e0d0","violet":"ee82ee","wheat":"f5deb3","white":"ffffff","whitesmoke":"f5f5f5","yellow":"ffff00","yellowgreen":"9acd32"}'
color_tab = json.loads(color_tab_str)


def triplet_from_str(color_str):
    color_str = "#" + color_str
    if "url" in color_str:
        return f"{int('0xFF', 16)},{int('0xFF', 16)},{int('0xFF', 16)}"

    starts = color_str.split("#")[1]
    if len(starts) == 6:
        red = int("0x" + starts[:2], 16)
        green = int("0x" + starts[2:4], 16)
        blue = int("0x" + starts[4:], 16)
    elif len(starts) == 3:
        red = int("0x" + starts[0] * 2, 16)
        green = int("0x" + starts[1] * 2, 16)
        blue = int("0x" + starts[2] * 2, 16)
    else:
        while len(starts) < 6:
            starts += starts[-1]
        red = int("0x" + starts[:2], 16)
        green = int("0x" + starts[2:4], 16)
        blue = int("0x" + starts[4:], 16)

    return f"{red},{green},{blue}"


def replace_url(s):
    for col in re.findall(r"url\((.+)\)", s):
        s = s.replace(f"url({col})", "None")
    return s


def replace_rgb(s):

    for col in re.findall(r"#([0-9a-fA-F]{3,6})", s):
        s = s.replace(f"#{col}", f"rgb({triplet_from_str(col)})")

    for col in re.findall(r"rgb\(([^\)]+)\);", s):
        try:
            r, g, b = [x.strip("%").strip() for x in col.split(",")]
        except Exception:
            print(col)
            print([x for x in col.split(",")])
            print("ERROR in parsing a color")
        new_col = f"{r},{g},{b}"
        s = s.replace(f"rgb({col})", f"rgb({new_col})")
    return s


def replace_color(s):
    for c in color_tab:
        s = s.replace(c, f"rgb({triplet_from_str(color_tab[c.lower()])})")
    return s


def round_stroke_and_op(s):
    for col in re.findall(r"stroke-width:(\d*\.\d+)", s):
        s = s.replace(f"stroke-width:{col}", f"stroke-width:{round(float(col), 2)}")
    for col in re.findall(r"opacity:(\d+\.\d+)", s):
        s = s.replace(f"opacity:{col}", f"opacity:{round(float(col), 2)}")
    return s


def clean_hex(s):
    for col in re.findall(r"(fill|stroke)\:([0-9a-fA-F]{6})", s):
        s = s.replace(f"{col[1]}", f"rgb({triplet_from_str(col[1])})")
        print(col)
    return s


def clean_broken_triplets(s):
    for col in re.findall(r"(fill|stroke)\:(255,255,255[^;]+)", s):
        s = s.replace(
            f"{col[0]}:{col[1]}",
            f"{col[0]}:rgb({''.join([x * 2 for x in col[1][11:]])})",
        )
    return s


def clean_color(svg):

    svg = replace_url(svg)
    svg = clean_hex(svg)
    svg = clean_broken_triplets(svg)
    svg = replace_color(svg)
    svg = round_stroke_and_op(svg)
    svg = replace_rgb(svg)
    return svg


def scale_paths_to_512(paths, canvas):
    """
    paths: output list from svgpathtools.svg2paths
    canvas: (min_x, min_y, width, height)
    returns: new list of scaled/translated Path objects
    """
    min_x, min_y, w, h = canvas
    sx = 512.0 / w
    sy = 512.0 / h

    # translate by (-min_x, -min_y), then scale
    scaled = [p.translated(complex(-min_x, -min_y)).scaled(sx, sy) for p in paths]
    return [_round_path(p) for p in scaled]


def stroke_scale_to_512(canvas):
    _, _, w, h = canvas
    return ((512.0 / w) + (512.0 / h)) / 2.0


def preprocess_svg_to_512(svg):
    """
    svg: file path or SVG string
    returns: scaled paths, canvas
    """
    if os.path.exists(svg):
        paths, attributes = svgpathtools.svg2paths(svg)
    else:
        paths, attributes = svgpathtools.svg2paths(StringIO(svg))
    canvas = extract_size(svg)
    scaled_paths = scale_paths_to_512(paths, canvas)
    return scaled_paths, attributes


def paths_to_svg_string(
    paths,
    attributes,
    size=512,
    include_header=False,
    stroke_scale=None,
    inherited_style=None,
):
    """
    paths: list of svgpathtools.Path
    attributes: list of dicts from svgpathtools.svg2paths
    size: canvas size (int)
    returns: SVG string
    """
    header = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<svg viewBox="0 0 512 512" version="1.1" xmlns="http://www.w3.org/2000/svg">\n'
    )
    if include_header:
        out = [header]
    else:
        out = []

    inherited_style = inherited_style or {}
    for p, attr in zip(paths, attributes):
        style_props = _parse_style_attr(attr.get("style"))
        style_parts = []
        for key in _STYLE_KEYS:
            val = style_props.get(key)
            if val is None or str(val).strip() == "":
                val = attr.get(key)
            if val is None or str(val).strip() == "":
                val = inherited_style.get(key)
            if val is not None and str(val).strip() != "":
                if key == "stroke-width" and stroke_scale is not None:
                    val = _scale_stroke_width(val, stroke_scale)
                style_parts.append(f"{key}:{val}")
        style_attr = f"{';'.join(style_parts)}" if style_parts else ""
        d_str = _strip_trailing_dot_zero(p.d())
        d_str = _compact_path_d(d_str)
        out.append(f'<path  style="{style_attr}"\td="{d_str}"/>\n')

    if include_header:
        out.append("</svg>")
    return "".join(out)


def preprocess(svg, size=512, scale_to_512=True, include_header=False):
    """
    svg: file path or SVG string
    size: output canvas size (int)
    returns: SVG string scaled to size x size
    """
    if os.path.exists(svg):
        paths, attributes = svgpathtools.svg2paths(svg)
    else:
        paths, attributes = svgpathtools.svg2paths(StringIO(svg))

    if scale_to_512:
        canvas = extract_size(svg)
        scaled_paths = scale_paths_to_512(paths, canvas)
        stroke_scale = stroke_scale_to_512(canvas)
    else:
        scaled_paths = paths
        stroke_scale = None
    inherited_style = inherited_style_from_svg(svg)
    return clean_color(
        paths_to_svg_string(
            scaled_paths,
            attributes,
            size=size,
            include_header=include_header,
            stroke_scale=stroke_scale,
            inherited_style=inherited_style,
        )
    )
