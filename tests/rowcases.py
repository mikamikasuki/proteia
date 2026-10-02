# SPDX-License-Identifier: Apache-2.0
"""Synthetic rows for the row-box detection tests (#51).

A band is a horizontal smear, ``depth * fx(x) * fy(y)``: along x a flat-topped
super-Gaussian ``exp(-0.5 |(x - cx) / sx|**4)`` (or a Gaussian, whose tails are
longer), along y a Gaussian. The membrane is 50000 in 16-bit units, darkened by
the bands, with Gaussian noise, clipped to [0, 65535] (an over-exposed dark band
is flat at 0) and rounded; a light-on-dark row is the inverse ``65535 - image``.

A band's reference rect is its own noise-free, unclipped extent at 20% of its
peak depth, rounded inward to whole pixels (half-open), so neighbouring
references may overlap. The row box spans every declared lane's nominal band,
the empty ones included, plus a margin.

* :func:`synthetic_row` and :func:`bench_cases`: the fifteen rows of the #51
  benchmark (bench51), rebuilt bit for bit: same parameters, same random draws.
* :func:`adversarial_row` and :data:`ADVERSARIAL`: the accuracy judge's
  generator and the recipes the tests use (neighbouring rows, doublets,
  streaks, stains, dust between lanes, bubbles), plus rows of the tests' own
  (a row large enough for the fits' subsample, guards that bind).
* :func:`jpeg`: a row as an 8-bit JPEG export, read back (block artefacts).
* :func:`fuzz_row`: seeded random geometry for the invariant checks.
"""

import dataclasses
import io
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from proteia.core.model import Rect

MEMBRANE = 50000.0
FULL_SCALE = 65535.0
NOISE_SIGMA = 400.0
REF_FRACTION = 0.2
_UX4 = (2.0 * math.log(1.0 / REF_FRACTION)) ** 0.25  # super-Gaussian half-extent at 20%
_UX2 = math.sqrt(2.0 * math.log(1.0 / REF_FRACTION))  # Gaussian half-extent at 20%

Artefact = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RowCase:
    """One synthetic row: ``image`` (float64, integer-valued), the ``row`` box,
    ``n_lanes`` and the ``reference`` rect of each lane with a band.
    ``lane_cx`` and ``lane_cy`` are every lane's true centre, the empty ones
    included."""

    name: str
    description: str
    image: np.ndarray
    dark_on_light: bool
    row: Rect
    n_lanes: int
    reference: dict[int, Rect]
    lane_cx: tuple[float, ...]
    lane_cy: tuple[float, ...]


def _ref_rect(cx: float, cy: float, w: float, h: float) -> Rect:
    return (
        math.ceil(cx - w / 2),
        math.ceil(cy - h / 2),
        math.floor(cx + w / 2) + 1,
        math.floor(cy + h / 2) + 1,
    )


# --- The bench51 rows ---

_BENCH_MARGIN_X = 70  # px of membrane left and right of the row
_BENCH_HEIGHT = 120
_BENCH_ROW_CY = 60.0


def synthetic_row(
    name: str,
    description: str,
    seed: int,
    *,
    n: int = 6,
    pitch: float = 70.0,
    w: float = 44.0,
    h: float = 12.0,
    pitches: Sequence[float] | None = None,
    missing: Sequence[int] = (),
    smile: float = 0.0,
    depth_range: tuple[float, float] = (18000.0, 30000.0),
    depth_override: Mapping[int, float | str] | None = None,
    gradient: tuple[float, float] = (0.0, 0.0),
    light_on_dark: bool = False,
    mx: int = 10,
    my: int = 6,
    x_jitter: float = 2.0,
    y_jitter: float = 1.0,
    w_var: float = 0.08,
    h_var: float = 0.10,
    noise_sigma: float = NOISE_SIGMA,
) -> RowCase:
    """One bench51 row: ``n`` lanes at ``pitch`` (or the steps ``pitches``), bands
    about ``w`` x ``h`` px (the 20% extents), ``missing`` lanes empty. ``smile``
    lifts the outer bands; ``gradient`` tilts the membrane across and down the
    row; ``depth_override`` sets a lane's depth, ``"faint10"`` a tenth of the
    others' mean; ``mx``/``my`` are the row box margins (negative cuts bands)."""
    rng = np.random.default_rng(seed)
    steps = list(pitches) if pitches is not None else [pitch] * (n - 1)
    cx0 = _BENCH_MARGIN_X + w / 2
    lane_cx = cx0 + np.concatenate([[0.0], np.cumsum(steps)])
    lane_cx = lane_cx + rng.uniform(-x_jitter, x_jitter, n)
    xc = 0.5 * (lane_cx[0] + lane_cx[-1])
    half = max(0.5 * (lane_cx[-1] - lane_cx[0]), 1.0)
    u = (lane_cx - xc) / half
    lane_cy = _BENCH_ROW_CY + smile * (0.5 - u**2) + rng.uniform(-y_jitter, y_jitter, n)
    widths = w * rng.uniform(1 - w_var, 1 + w_var, n)
    heights = h * rng.uniform(1 - h_var, 1 + h_var, n)
    depths = rng.uniform(*depth_range, n)
    for lane, value in (depth_override or {}).items():
        if value == "faint10":
            depths[lane] = float(np.mean([depths[i] for i in range(n) if i != lane])) / 10.0
        else:
            depths[lane] = float(value)

    width_img = int(math.ceil(lane_cx[-1] + widths[-1] / 2 + _BENCH_MARGIN_X))
    xs = np.arange(width_img, dtype=float)
    ys = np.arange(_BENCH_HEIGHT, dtype=float)
    gx, gy = gradient
    bg = MEMBRANE + np.zeros((_BENCH_HEIGHT, width_img))
    if gx or gy:
        bg = (
            bg + gx * ((xs - xc) / half)[None, :] + gy * ((ys - _BENCH_ROW_CY) / (2.0 * h))[:, None]
        )
    depth_map = np.zeros_like(bg)
    all_refs: dict[int, Rect] = {}
    reference: dict[int, Rect] = {}
    for i in range(n):
        all_refs[i] = _ref_rect(lane_cx[i], lane_cy[i], widths[i], heights[i])
        if i in set(missing):
            continue
        sx = widths[i] / (2.0 * _UX4)
        sy = heights[i] / (2.0 * _UX2)
        fx = np.exp(-0.5 * np.abs((xs - lane_cx[i]) / sx) ** 4)
        fy = np.exp(-0.5 * ((ys - lane_cy[i]) / sy) ** 2)
        depth_map += depths[i] * np.outer(fy, fx)
        reference[i] = all_refs[i]
    noise = rng.normal(0.0, noise_sigma, bg.shape)
    image = np.round(np.clip(bg - depth_map + noise, 0.0, FULL_SCALE))
    if light_on_dark:
        image = FULL_SCALE - image
    row = (
        max(0, min(r[0] for r in all_refs.values()) - mx),
        max(0, min(r[1] for r in all_refs.values()) - my),
        min(width_img, max(r[2] for r in all_refs.values()) + mx),
        min(_BENCH_HEIGHT, max(r[3] for r in all_refs.values()) + my),
    )
    return RowCase(
        name,
        description,
        image.astype(np.float64),
        not light_on_dark,
        row,
        n,
        reference,
        tuple(float(c) for c in lane_cx),
        tuple(float(c) for c in lane_cy),
    )


def bench_cases() -> list[RowCase]:
    """The fifteen bench51 rows (91 reference bands), rebuilt on every call."""
    base = 51  # all_present and the missing-lane rows share geometry and noise
    return [
        synthetic_row("all_present", "6 lanes, pitch 70, W~44 H~12, depths 18k-30k", base),
        synthetic_row("missing_first", "lane 0 empty (the box covers it)", base, missing=[0]),
        synthetic_row("missing_middle", "lane 2 empty", base, missing=[2]),
        synthetic_row("missing_last", "lane 5 empty (the box covers it)", base, missing=[5]),
        synthetic_row("missing_two", "lanes 0 and 3 empty", base, missing=[0, 3]),
        synthetic_row(
            "touching",
            "pitch 48, W 60: neighbours overlap at half height, no valley between them",
            52,
            pitch=48.0,
            w=60.0,
            w_var=0.0,
            x_jitter=0.0,
            depth_range=(22000.0, 26000.0),
        ),
        synthetic_row("smile", "outer bands 8 px higher than the middle", 53, smile=8.0),
        synthetic_row(
            "uneven_spacing",
            "pitches 56/84/63/80/60 (70 +/-20%), W~40",
            54,
            w=40.0,
            pitches=[56.0, 84.0, 63.0, 80.5, 59.5],
            x_jitter=0.0,
        ),
        synthetic_row("faint_band", "lane 3 ten times fainter", 55, depth_override={3: "faint10"}),
        synthetic_row(
            "overexposed", "lane 2 clipped flat at 0", 56, depth_override={2: 1.5 * MEMBRANE}
        ),
        synthetic_row(
            "uneven_background",
            "membrane -6000..+6000 across the row, +/-1000 down it",
            57,
            gradient=(6000.0, 1000.0),
        ),
        synthetic_row(
            "light_on_dark", "inverted: membrane 15535, bright bands", 58, light_on_dark=True
        ),
        synthetic_row("loose_box", "margins half a pitch across, 15 px down", 59, mx=35, my=15),
        synthetic_row("tight_box", "cuts 4 px off both outer bands, 1 px down", 60, mx=-4, my=1),
        synthetic_row(
            "twelve_lanes", "12 lanes, pitch 50, W~34 H~10", 61, n=12, pitch=50.0, w=34.0, h=10.0
        ),
    ]


# --- The accuracy judge's generator ---


def _band(
    xs: np.ndarray,
    ys: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    depth: float,
    shape: str,
) -> np.ndarray:
    if shape == "super":
        fx = np.exp(-0.5 * np.abs((xs - cx) / (w / (2 * _UX4))) ** 4)
    else:
        fx = np.exp(-0.5 * ((xs - cx) / (w / (2 * _UX2))) ** 2)
    fy = np.exp(-0.5 * ((ys - cy) / (h / (2 * _UX2))) ** 2)
    return depth * np.outer(fy, fx)


def adversarial_row(
    name: str,
    seed: int,
    *,
    n: int = 6,
    pitch: float = 70.0,
    pitches: Sequence[float] | None = None,
    w: float = 44.0,
    h: float = 12.0,
    missing: Sequence[int] = (),
    depth_range: tuple[float, float] = (18000.0, 30000.0),
    depths: Mapping[int, float] | None = None,
    widths: Mapping[int, float] | None = None,
    heights: Mapping[int, float] | None = None,
    shape: str = "super",
    tilt: float = 0.0,
    smile: float = 0.0,
    doublet: Mapping[int, tuple[float, float]] | None = None,
    holes: Sequence[tuple[int, float, float, float]] = (),
    neighbour_dy: float | None = None,
    neighbour_rel: float = 1.0,
    neighbour_rels: Mapping[int, float] | None = None,
    shifts: Mapping[int, float] | None = None,
    artefacts: Sequence[Artefact] = (),
    noise: float = NOISE_SIGMA,
    light_on_dark: bool = False,
    mx: int = 10,
    my: int = 6,
    box_adjust: tuple[int, int, int, int] = (0, 0, 0, 0),
    img_h: int = 160,
    x_jitter: float = 2.0,
    y_jitter: float = 1.0,
    margin_left: int = 70,
    margin_right: int = 70,
) -> RowCase:
    """A row from the accuracy judge's generator (acc_adv), the same draws.

    Beyond :func:`synthetic_row`: ``doublet`` maps a lane to ``(dy, frac)``, two
    components ``dy`` apart, the second ``frac`` as deep (reference: their union);
    ``holes`` are ``(lane, dx, dy, r)``: a round hole of radius ``r`` punched into
    the bands (multiplicative, a transfer bubble) that far from the lane's centre;
    ``neighbour_dy`` adds a neighbouring row that far below (above if
    negative), ``neighbour_rel`` as deep (``neighbour_rels`` sets a lane's
    own); ``shifts`` moves single lanes' bands down (up if negative) by that
    many px, as a montage's panel or a mark beside the row lies, the row box
    spanning them;
    ``artefacts`` add darkening maps ``f(X, Y, lane_cx, lane_cy)``; ``widths``,
    ``heights`` and ``depths`` override single bands; ``box_adjust`` moves the
    row box's edges; ``margin_left`` and ``margin_right`` are the membrane
    left of the first band and right of the last (room for what lies beside
    the row, :func:`beside`)."""
    rng = np.random.default_rng(seed)
    steps = list(pitches) if pitches is not None else [pitch] * (n - 1)
    lane_cx = (margin_left + w / 2 + np.concatenate([[0.0], np.cumsum(steps)])) + rng.uniform(
        -x_jitter, x_jitter, n
    )
    row_cy = img_h / 2
    xc = 0.5 * (lane_cx[0] + lane_cx[-1])
    half = max(0.5 * (lane_cx[-1] - lane_cx[0]), 1.0)
    u = (lane_cx - xc) / half
    lane_cy = row_cy + smile * (0.5 - u**2) + tilt * u / 2 + rng.uniform(-y_jitter, y_jitter, n)
    for k, v in (shifts or {}).items():
        lane_cy[k] += v
    ws = w * rng.uniform(0.92, 1.08, n)
    hs = h * rng.uniform(0.9, 1.1, n)
    for k, v in (widths or {}).items():
        ws[k] = v
    for k, v in (heights or {}).items():
        hs[k] = v
    dps = rng.uniform(*depth_range, n)
    for k, v in (depths or {}).items():
        dps[k] = v
    width_img = int(math.ceil(lane_cx[-1] + ws[-1] / 2 + margin_right))
    xs = np.arange(width_img, dtype=float)
    ys = np.arange(img_h, dtype=float)
    base = np.full((img_h, width_img), MEMBRANE)
    dmap = np.zeros_like(base)
    refs_all: dict[int, Rect] = {}
    reference: dict[int, Rect] = {}
    for i in range(n):
        refs_all[i] = _ref_rect(lane_cx[i], lane_cy[i], ws[i], hs[i])
        if i in set(missing):
            continue
        if doublet and i in doublet:
            dy, frac = doublet[i]
            hh = hs[i] * 0.7
            c1, c2 = lane_cy[i] - dy / 2, lane_cy[i] + dy / 2
            dmap += _band(xs, ys, lane_cx[i], c1, ws[i], hh, dps[i], shape)
            dmap += _band(xs, ys, lane_cx[i], c2, ws[i], hh, dps[i] * frac, shape)
            r1, r2 = _ref_rect(lane_cx[i], c1, ws[i], hh), _ref_rect(lane_cx[i], c2, ws[i], hh)
            reference[i] = (
                min(r1[0], r2[0]),
                min(r1[1], r2[1]),
                max(r1[2], r2[2]),
                max(r1[3], r2[3]),
            )
        else:
            dmap += _band(xs, ys, lane_cx[i], lane_cy[i], ws[i], hs[i], dps[i], shape)
            reference[i] = refs_all[i]
    for lane, hdx, hdy, hr in holes:
        hx, hy = lane_cx[lane] + hdx, lane_cy[lane] + hdy
        dmap *= 1.0 - np.exp(
            -0.5 * (((xs[None, :] - hx) / hr) ** 2 + ((ys[:, None] - hy) / hr) ** 2)
        )
    if neighbour_dy is not None:
        for i in range(n):
            cy = lane_cy[i] + neighbour_dy
            rel = (neighbour_rels or {}).get(i, neighbour_rel)
            dmap += _band(xs, ys, lane_cx[i], cy, ws[i], h, dps[i] * rel, shape)
    for f in artefacts:
        dmap += f(xs[None, :], ys[:, None], lane_cx, lane_cy)
    nz = rng.normal(0.0, 1.0, base.shape)
    image = np.clip(base - dmap + noise * nz, 0.0, FULL_SCALE)
    if light_on_dark:
        image = FULL_SCALE - image
    image = np.round(image)  # after the inversion, as the judge's generator
    row = (
        max(0, min(r[0] for r in refs_all.values()) - mx + box_adjust[0]),
        max(0, min(r[1] for r in refs_all.values()) - my + box_adjust[1]),
        min(width_img, max(r[2] for r in refs_all.values()) + mx + box_adjust[2]),
        min(img_h, max(r[3] for r in refs_all.values()) + my + box_adjust[3]),
    )
    return RowCase(
        name,
        "",
        image.astype(np.float64),
        not light_on_dark,
        row,
        n,
        reference,
        tuple(float(c) for c in lane_cx),
        tuple(float(c) for c in lane_cy),
    )


def blob(
    lane: int,
    r: float,
    depth: float,
    *,
    ry: float | None = None,
    dy: float = 0.0,
    dx: float = 0.0,
) -> Artefact:
    """A round dark blob (dust, a stain) of radius ``r`` on ``lane``'s band centre
    (``dy`` px below it, ``dx`` px right of it); a negative depth is a light
    spot. ``ry`` makes it an ellipse, ``r`` px across and ``ry`` px high."""
    ry = r if ry is None else ry

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        cx, cy = lcx[lane] + dx, lcy[lane] + dy
        return depth * np.exp(-0.5 * (((X - cx) / r) ** 2 + ((Y - cy) / ry) ** 2))

    return f


def blob_between(left: int, r: float, depth: float) -> Artefact:
    """A round blob of radius ``r`` midway between lanes ``left`` and ``left + 1``,
    at the band height of ``left``; a negative depth is a light spot."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        cx = 0.5 * (lcx[left] + lcx[left + 1])
        return depth * np.exp(-0.5 * (((X - cx) / r) ** 2 + ((Y - lcy[left]) / r) ** 2))

    return f


def band_between(left: int, w: float, h: float, depth: float, dy: float = 0.0) -> Artefact:
    """An extra flat-topped band, ``w`` x ``h`` px at 20%, midway between lanes
    ``left`` and ``left + 1``, at the band height of ``left`` (``dy`` px below
    it)."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        cx = 0.5 * (lcx[left] + lcx[left + 1])
        return _band(X[0], Y[:, 0], cx, lcy[left] + dy, w, h, depth, "super")

    return f


def beside(
    lanes: float,
    w: float,
    h: float,
    depth: float,
    *,
    marks: int = 1,
    gap: float = 0.0,
    dy: float = 0.0,
) -> Artefact:
    """What lies beside the row and is no lane of it: ``marks`` flat-topped
    marks ``w`` x ``h`` px at 20%, ``gap`` px apart, the first centred
    ``lanes`` lane steps past the last lane's centre (before the first lane's
    if negative, the marks then running left), at the row's mean band height
    (``dy`` px below it). A ladder's band or a tick mark is one narrow mark, a
    label's text a few thin strokes, a neighbouring panel's lane a band."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        if lanes >= 0:
            x0, step = lcx[-1] + lanes * (lcx[-1] - lcx[-2]), w + gap
        else:
            x0, step = lcx[0] + lanes * (lcx[1] - lcx[0]), -(w + gap)
        cy = float(np.mean(lcy)) + dy
        out = np.zeros((Y.shape[0], X.shape[1]))
        for k in range(marks):
            out += _band(X[0], Y[:, 0], x0 + k * step, cy, w, h, depth, "super")
        return out

    return f


def vstreak(lane: int, half_w: float, depth: float) -> Artefact:
    """A dark vertical streak down ``lane`` over the whole image height."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        return depth * np.exp(-0.5 * np.abs((X - lcx[lane]) / half_w) ** 4) + 0 * Y

    return f


def hstripe(half_h: float, depth: float) -> Artefact:
    """A darker stretch of membrane: a flat stripe across the whole image width,
    ``2 * half_h`` px high about the row's mean band height."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        return depth * (np.abs(Y - float(np.mean(lcy))) < half_h) + 0 * X

    return f


def frame(dy: float, px: int, depth: float, *, dx: float = 50.0, slope: float = 0.0) -> Artefact:
    """A panel's drawn frame: sharp lines ``px`` px thick, ``dy`` px above and
    below the row's mean band height, joined by sides ``dx`` px outside the
    end lanes' centres. ``slope`` tilts the lines about the row's centre (a
    scan turned a little); the sides run between them."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        cy = float(np.mean(lcy))
        shift = slope * (X - 0.5 * (lcx[0] + lcx[-1]))
        top, bottom = np.floor(cy - dy + shift), np.floor(cy + dy + shift)
        left, right = math.floor(lcx[0] - dx), math.floor(lcx[-1] + dx)
        across = (X >= left) & (X < right + px)
        down = (Y >= top) & (Y < bottom + px)
        lines = ((Y >= top) & (Y < top + px)) | ((Y >= bottom) & (Y < bottom + px))
        sides = ((X >= left) & (X < left + px)) | ((X >= right) & (X < right + px))
        return depth * ((across & lines) | (down & sides))

    return f


def bottom_strip(rows: int, depth: float, fade: float = 0.25) -> Artefact:
    """A dark strip over the image's bottom ``rows`` rows, across its whole
    width (a screenshot's toolbar): ``depth`` on its top row, fading by
    ``fade`` of that to the image's last row."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        top = Y.max() + 1 - rows
        return depth * (Y >= top) * (1.0 - fade * (Y - top) / max(1, rows - 1)) + 0 * X

    return f


def dark_edge(cols: int, depth: float) -> Artefact:
    """The image's right edge darkening over its last ``cols`` columns, to
    ``depth`` on its last one (a vignetted scan)."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        ramp = np.clip((X - (X.max() - cols)) / cols, 0.0, 1.0)
        return depth * ramp**2 + 0 * Y

    return f


def shade_above(end: int, rows: int, depth: float) -> Artefact:
    """The membrane darkening over the ``rows`` rows above image row ``end``,
    to ``depth`` on the last of them, across the whole width: a vignetted
    scan whose image ends at ``end`` (cut it there with :func:`image_cut`)."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        ramp = np.clip((Y - (end - 1 - rows)) / rows, 0.0, 1.0)
        return depth * ramp**2 * (Y < end) + 0 * X

    return f


def image_cut(case: RowCase, *, top: int = 0, bottom: int | None = None) -> RowCase:
    """``case`` with its image cut to the rows ``[top, bottom)`` and its row
    box dragged to each cut (to the image's new top or bottom row), as over a
    tightly cropped image; the reference and the lanes' heights move with the
    image."""
    x0, y0, x1, y1 = case.row
    end = case.image.shape[0] if bottom is None else bottom
    return dataclasses.replace(
        case,
        image=case.image[top:end],
        row=(x0, 0 if top else y0, x1, (end if bottom is not None else y1) - top),
        reference={
            lane: (r[0], r[1] - top, r[2], r[3] - top) for lane, r in case.reference.items()
        },
        lane_cy=tuple(cy - top for cy in case.lane_cy),
    )


# The judge's recipes the tests use (seeds 1000 and up, as the judge ran them),
# then rows of the tests' own.
ADVERSARIAL: dict[str, dict] = {
    "nbr_above_miss": {
        "missing": [2],
        "neighbour_dy": -22.0,
        "box_adjust": (0, -10, 0, 0),
    },
    "blotch_empty": {"missing": [4], "artefacts": [blob(4, 40, 6000)]},
    "vstreak_empty": {"missing": [4], "artefacts": [vstreak(4, 14, 3000)]},
    "box_omits_empty_first": {"missing": [0], "box_adjust": (70, 0, 0, 0)},
    "doublet_two": {"doublet": {1: (9, 0.6), 4: (9, 0.6)}},
    "gauss_tails_miss": {"shape": "gauss", "pitch": 56.0, "missing": [2]},
    "twenty_lanes": {
        "n": 20,
        "pitch": 36.0,
        "w": 24.0,
        "h": 10.0,
        "missing": [0, 7, 8, 19],
        "mx": 6,
    },
    # Dust between lanes 2 and 3: one piece more than lanes, merged or dropped.
    "blob_gap": {"artefacts": [blob_between(2, 5, 15000)]},
    # A bubble splits lane 1's band in two; a light spot between lanes 3 and 4.
    "bubble_band": {"holes": [(1, 0.0, 0.0, 7.0)], "artefacts": [blob_between(3, 7, -1500)]},
    # The tests' own rows. Lane 3 above twice the median height: size_outlier.
    "tall_band": {"heights": {3: 30.0}},
    # Lane 2 holds two components 14 px apart, the lower one 0.7x as deep.
    "doublet_deep": {"doublet": {2: (14, 0.7)}, "my": 8},
    # A smile, a tall band and a tight box: the vertical placement clamp binds.
    "smile_tall_tight": {"smile": 10.0, "heights": {4: 23.7}, "box_adjust": (0, 3, 0, -3)},
    # Three times the bench scale: every fit and noise estimate subsamples.
    "wide_row": {
        "pitch": 210.0,
        "w": 132.0,
        "h": 36.0,
        "img_h": 240,
        "mx": 30,
        "my": 18,
        "missing": [2],
    },
    # Touching bands, the first a quarter as deep as the rest: its end of the
    # run is trimmed against its neighbour's peak once the envelope reaches it.
    "touching_weak_end": {"pitch": 48.0, "w": 60.0, "depths": {0: 6000.0}},
    # Lane 2 spread to 1.4 pitches over empty lane 3 (the judge's
    # wide_band_next_empty): the piece is lane 2's, and the kept signal it
    # spills into lane 3 fits no lane.
    "wide_next_empty": {"widths": {2: 100.0}, "missing": [3]},
    # A tilted row (lane 1 highest) whose box's top edge runs 3 px above the
    # highest band's centre: it cuts the bands of lanes 1 and 2 only.
    "tilt_cut": {"tilt": 10.0, "box_adjust": (0, 9, 0, 0)},
    # #111: a first row box that also covers what lies beside the row, read a
    # lane or more off. A ladder's band 1.8 lane steps past the last lane,
    # lane 0 empty: read as the last lane, each band a lane early.
    "ladder_beside": {
        "missing": [0],
        "artefacts": [beside(1.8, 26, 12, 14000)],
        "margin_right": 200,
        "box_adjust": (0, 0, 150, 0),
    },
    # The same before the first lane, the last lane empty.
    "ladder_before": {
        "missing": [5],
        "artefacts": [beside(-1.8, 26, 12, 14000)],
        "margin_left": 200,
        "box_adjust": (-150, 0, 0, 0),
    },
    # A label's text, four strokes 1.3 lane steps past the last lane, lane 0 empty.
    "label_beside": {
        "missing": [0],
        "artefacts": [beside(1.3, 10, 14, 20000, marks=4, gap=4)],
        "margin_right": 200,
        "box_adjust": (0, 0, 150, 0),
    },
    # Five lanes, the first two empty; an arrow 0.8 lane steps past the last
    # and a neighbouring panel's band 2.2 past it.
    "panel_beside": {
        "n": 5,
        "missing": [0, 1],
        "artefacts": [beside(0.8, 20, 6, 15000), beside(2.2, 44, 12, 25000)],
        "margin_right": 240,
        "box_adjust": (0, 0, 190, 0),
    },
    # #121: lane 1's band pale across its middle, its two ends about 1.7x as
    # dark (a dumbbell); the same band over-exposed, clipped flat at 0 around
    # its lighter centre (a hollow band).
    "dumbbell_band": {"depths": {1: 24000.0}, "artefacts": [blob(1, 6.0, -11000.0, ry=12.0)]},
    "hollow_band": {
        "depths": {1: 1.5 * MEMBRANE},
        "artefacts": [blob(1, 6.0, -45000.0, ry=12.0)],
    },
    # #121: lane 1's band 2.5x as deep as the membrane, clipped at 0 all around
    # a lighter centre, back to 40000 there (a ring: burnt out in its middle);
    # the band of hollow_band with dark spots at its two ends, 6 px above its
    # middle (a notch in its top edge, as a band whose ends curve up shows).
    "hollow_ring": {
        "depths": {1: 2.5 * MEMBRANE},
        "artefacts": [blob(1, 7.0, -2.3 * MEMBRANE, ry=2.0)],
        "my": 10,
    },
    "notched_band": {
        "depths": {1: 1.5 * MEMBRANE},
        "artefacts": [blob(1, 4.0, 60000.0, dx=x, dy=-6.0) for x in (-13.0, 13.0)],
        "my": 10,
    },
}


def adversarial(key: str, seed: int) -> RowCase:
    """The recipe ``key`` of :data:`ADVERSARIAL` at ``seed``."""
    return adversarial_row(key, seed, **ADVERSARIAL[key])


def jpeg(case: RowCase, quality: int = 75, membrane: float = 200.0) -> RowCase:
    """A dark-on-light ``case`` as an 8-bit JPEG export of that quality, read
    back: the membrane at ``membrane`` grey levels and the bands scaled with it
    (about 70 to 120 levels deep), rounded, then compressed. On a smooth
    membrane the compression leaves 8x8 block artefacts a few levels deep, as
    a JPEG blot shows them (#121)."""
    scaled = np.clip(np.round(case.image * membrane / MEMBRANE), 0.0, 255.0).astype(np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(scaled).save(encoded, format="JPEG", quality=quality)
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        image = np.asarray(decoded, dtype=np.float64)
    return dataclasses.replace(case, image=image)


def fuzz_row(seed: int) -> RowCase:
    """A row of random geometry for the invariant checks: 2 to 8 lanes, some
    empty, a smile, one band up to 2.5x the usual height, the box edges moved
    (the vertical ones tight) and the declared lane count off by one either way.
    What detection proposes here is not checked; the invariants of a result are.
    The reference keeps the generated lanes."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 9))
    tall = int(rng.integers(n))
    missing = [i for i in range(n) if rng.random() < 0.2]
    case = adversarial_row(
        f"fuzz/{seed}",
        seed,
        n=n,
        missing=missing,
        smile=float(rng.uniform(0.0, 12.0)),
        heights={tall: float(rng.uniform(12.0, 30.0))},
        box_adjust=(
            int(rng.integers(-40, 41)),
            int(rng.integers(-4, 7)),
            int(rng.integers(-40, 41)),
            int(rng.integers(-6, 5)),
        ),
    )
    return dataclasses.replace(case, n_lanes=max(1, n + int(rng.integers(-1, 2))))
