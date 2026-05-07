from __future__ import annotations

import math
from dataclasses import dataclass

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.transforms import IdentityTransform


@dataclass(frozen=True, slots=True)
class IgorInlineSegment:
    text: str
    bold: bool
    italic: bool
    font_size: float


def prepare_igor_inline_segments(
    text: str,
    *,
    default_font_size: float,
) -> tuple[list[IgorInlineSegment], bool]:
    if not text:
        return ([], False)

    segments: list[IgorInlineSegment] = []
    has_markup = False
    bold = False
    italic = False
    font_size = float(default_font_size)
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        segment_text = "".join(buffer)
        buffer.clear()
        if segment_text:
            segments.append(
                IgorInlineSegment(
                    text=segment_text,
                    bold=bold,
                    italic=italic,
                    font_size=font_size,
                )
            )

    for in_math, chunk in _split_math_sections(text):
        if in_math:
            buffer.append(f"${chunk}$")
            continue

        index = 0
        while index < len(chunk):
            if chunk.startswith(r"\f", index) and index + 4 <= len(chunk):
                code = chunk[index + 2 : index + 4]
                if code in {"00", "01", "02"}:
                    flush()
                    has_markup = True
                    if code == "00":
                        bold = False
                        italic = False
                    elif code == "01":
                        bold = True
                    else:
                        italic = True
                    index += 4
                    continue
            if chunk.startswith(r"\Z", index):
                size_text, consumed = _parse_igor_font_size(chunk[index + 2 :])
                if consumed > 0 and size_text is not None:
                    flush()
                    has_markup = True
                    size_value = float(size_text)
                    font_size = (
                        float(default_font_size)
                        if size_value <= 0.0
                        else size_value
                    )
                    index += 2 + consumed
                    continue
            buffer.append(chunk[index])
            index += 1

    flush()
    return (segments, has_markup)


def has_igor_inline_markup(text: str) -> bool:
    _segments, has_markup = prepare_igor_inline_segments(
        text,
        default_font_size=12.0,
    )
    return has_markup


def igor_inline_to_mathtext(
    text: str,
    *,
    default_font_size: float,
) -> str:
    segments, has_markup = prepare_igor_inline_segments(
        text,
        default_font_size=default_font_size,
    )
    if not has_markup:
        return text
    if any(
        not math.isclose(segment.font_size, default_font_size)
        for segment in segments
    ):
        raise ValueError("Inline font-size changes require composite drawing")
    return _segments_to_mathtext(segments)


def apply_igor_inline_text_artist(
    text_artist: Text,
    raw_text: str,
    *,
    default_font_size: float,
    gid_prefix: str,
    target_axes=None,
) -> list[Text]:
    segments, has_markup = prepare_igor_inline_segments(
        raw_text,
        default_font_size=default_font_size,
    )
    if not has_markup:
        text_artist.set_text(raw_text)
        return []

    requires_composite = any(
        not math.isclose(segment.font_size, default_font_size)
        for segment in segments
    )
    if not requires_composite:
        text_artist.set_text(_segments_to_mathtext(segments))
        return []

    text_artist.set_text(" ")
    return _compose_inline_segments(
        text_artist,
        segments,
        gid_prefix=gid_prefix,
        target_axes=target_axes,
    )


def _parse_igor_font_size(chunk: str) -> tuple[str | None, int]:
    if not chunk:
        return (None, 0)
    if chunk.startswith("<"):
        end = chunk.find(">")
        if end <= 1:
            return (None, 0)
        value = chunk[1:end]
        try:
            float(value)
        except ValueError:
            return (None, 0)
        return (value, end + 1)

    end = 0
    while end < len(chunk) and (chunk[end].isdigit() or chunk[end] == "."):
        end += 1
    if end == 0:
        return (None, 0)
    value = chunk[:end]
    try:
        float(value)
    except ValueError:
        return (None, 0)
    return (value, end)


def _split_math_sections(text: str) -> list[tuple[bool, str]]:
    sections: list[tuple[bool, str]] = []
    buffer: list[str] = []
    in_math = False

    def flush() -> None:
        if buffer:
            sections.append((in_math, "".join(buffer)))
            buffer.clear()

    index = 0
    while index < len(text):
        char = text[index]
        if char == "$" and (index == 0 or text[index - 1] != "\\"):
            flush()
            in_math = not in_math
            index += 1
            continue
        buffer.append(char)
        index += 1
    flush()
    return sections


def _segments_to_mathtext(segments: list[IgorInlineSegment]) -> str:
    body = "".join(_segment_mathtext_body(segment) for segment in segments)
    return f"${body}$"


def _segment_mathtext_body(segment: IgorInlineSegment) -> str:
    pieces: list[str] = []
    for in_math, chunk in _split_math_sections(segment.text):
        if not chunk:
            continue
        if in_math:
            if chunk.startswith("^") or chunk.startswith("_"):
                pieces.append(chunk)
            else:
                pieces.append(
                    _wrap_with_math_style(
                        chunk,
                        bold=segment.bold,
                        italic=segment.italic,
                    )
                )
            continue
        pieces.append(
            _wrap_with_math_style(
                _escape_literal_for_mathtext(chunk),
                bold=segment.bold,
                italic=segment.italic,
            )
        )
    return "".join(pieces)


def _wrap_with_math_style(
    text: str,
    *,
    bold: bool,
    italic: bool,
) -> str:
    if not text:
        return ""
    if bold and italic:
        return rf"\mathbf{{\mathit{{{text}}}}}"
    if bold:
        return rf"\mathbf{{{text}}}"
    if italic:
        return rf"\mathit{{{text}}}"
    return rf"\mathregular{{{text}}}"


def _escape_literal_for_mathtext(text: str) -> str:
    escaped = (
        text.replace("\\", r"\backslash ")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("_", r"\_")
        .replace("^", r"\^{}")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )
    return escaped.replace(" ", r"\ ")


def _compose_inline_segments(
    text_artist: Text,
    segments: list[IgorInlineSegment],
    *,
    gid_prefix: str,
    target_axes=None,
) -> list[Text]:
    if not segments:
        return []

    figure = text_artist.figure
    if figure is None:
        return []
    renderer = _measurement_renderer(figure)
    anchor_transform = text_artist.get_transform()
    anchor = anchor_transform.transform(text_artist.get_position())
    rotation = float(text_artist.get_rotation())
    ha = str(text_artist.get_ha())
    va = str(text_artist.get_va())
    color = text_artist.get_color()
    zorder = text_artist.get_zorder()
    font_family = _font_family_name(text_artist)
    axes = target_axes if target_axes is not None else text_artist.axes
    if axes is None:
        return []

    metrics: list[tuple[IgorInlineSegment, str, float, float, float]] = []
    total_width = 0.0
    max_ascent = 0.0
    max_descent = 0.0
    for segment in segments:
        mathtext = f"${_segment_mathtext_body(segment)}$"
        font_properties = FontProperties(size=segment.font_size)
        if font_family:
            font_properties.set_family(font_family)
        width, height, descent = renderer.get_text_width_height_descent(
            mathtext,
            font_properties,
            ismath=True,
        )
        ascent = height - descent
        total_width += width
        max_ascent = max(max_ascent, ascent)
        max_descent = max(max_descent, descent)
        metrics.append((segment, mathtext, width, ascent, descent))

    x_offset = _horizontal_alignment_offset(total_width, ha)
    y_offset = _vertical_alignment_offset(max_ascent, max_descent, va)
    angle = math.radians(rotation)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    current_x = x_offset

    artists: list[Text] = []
    for index, (segment, mathtext, width, _ascent, _descent) in enumerate(
        metrics
    ):
        display_x = (
            anchor[0] + (current_x * cos_angle) - (y_offset * sin_angle)
        )
        display_y = (
            anchor[1] + (current_x * sin_angle) + (y_offset * cos_angle)
        )
        artist = axes.text(
            display_x,
            display_y,
            mathtext,
            transform=IdentityTransform(),
            ha="left",
            va="baseline",
            rotation=rotation,
            rotation_mode="anchor",
            clip_on=False,
            fontsize=segment.font_size,
            color=color,
            zorder=zorder,
        )
        if font_family:
            artist.set_fontfamily(font_family)
        artist.set_gid(f"{gid_prefix}-{index}")
        artists.append(artist)
        current_x += width
    return artists


def _measurement_renderer(figure) -> RendererAgg:
    width = max(1, int(math.ceil(figure.bbox.width)))
    height = max(1, int(math.ceil(figure.bbox.height)))
    return RendererAgg(width, height, figure.dpi)


def _font_family_name(text_artist: Text) -> str:
    family = text_artist.get_fontfamily()
    if isinstance(family, str):
        return family
    if family:
        return str(family[0])
    return ""


def _horizontal_alignment_offset(total_width: float, ha: str) -> float:
    if ha == "center":
        return -(0.5 * total_width)
    if ha == "right":
        return -total_width
    return 0.0


def _vertical_alignment_offset(
    max_ascent: float,
    max_descent: float,
    va: str,
) -> float:
    if va in {"center", "center_baseline"}:
        return -((max_ascent - max_descent) * 0.5)
    if va == "top":
        return -max_ascent
    if va == "bottom":
        return max_descent
    return 0.0


__all__ = [
    "IgorInlineSegment",
    "apply_igor_inline_text_artist",
    "has_igor_inline_markup",
    "igor_inline_to_mathtext",
    "prepare_igor_inline_segments",
]
