import numpy as np
from typing import Tuple

from colors_of_meaning.domain.model.lab_color import LabColor


def rgb_to_lab(r: float, g: float, b: float) -> LabColor:
    r_linear = _gamma_expand(r / 255.0)
    g_linear = _gamma_expand(g / 255.0)
    b_linear = _gamma_expand(b / 255.0)

    x = r_linear * 0.4124564 + g_linear * 0.3575761 + b_linear * 0.1804375
    y = r_linear * 0.2126729 + g_linear * 0.7151522 + b_linear * 0.0721750
    z = r_linear * 0.0193339 + g_linear * 0.1191920 + b_linear * 0.9503041

    x_n = 95.047
    y_n = 100.000
    z_n = 108.883

    x = (x * 100.0) / x_n
    y = (y * 100.0) / y_n
    z = (z * 100.0) / z_n

    fx = _lab_f(x)
    fy = _lab_f(y)
    fz = _lab_f(z)

    lightness = 116.0 * fy - 16.0
    a_val = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    clamped_l = max(0.0, min(100.0, lightness))
    clamped_a = max(-128.0, min(127.0, a_val))
    clamped_b = max(-128.0, min(127.0, b_val))

    return LabColor(l=clamped_l, a=clamped_a, b=clamped_b)


def lab_to_rgb(lab: LabColor) -> Tuple[int, int, int]:
    fy = (lab.l + 16.0) / 116.0
    fx = lab.a / 500.0 + fy
    fz = fy - lab.b / 200.0

    x_n = 95.047
    y_n = 100.000
    z_n = 108.883

    x = _lab_f_inv(fx) * x_n / 100.0
    y = _lab_f_inv(fy) * y_n / 100.0
    z = _lab_f_inv(fz) * z_n / 100.0

    r_linear = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g_linear = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_linear = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    r = _gamma_compress(r_linear)
    g = _gamma_compress(g_linear)
    b_val = _gamma_compress(b_linear)

    r_int = int(np.clip(np.round(r * 255.0), 0, 255))
    g_int = int(np.clip(np.round(g * 255.0), 0, 255))
    b_int = int(np.clip(np.round(b_val * 255.0), 0, 255))

    return (r_int, g_int, b_int)


def delta_e(lab1: LabColor, lab2: LabColor) -> float:
    dl = lab1.l - lab2.l
    da = lab1.a - lab2.a
    db = lab1.b - lab2.b
    return float(np.sqrt(dl * dl + da * da + db * db))


def _gamma_expand(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    else:
        return float(((c + 0.055) / 1.055) ** 2.4)


def _gamma_compress(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    else:
        return float(1.055 * (c ** (1.0 / 2.4)) - 0.055)


def _lab_f(t: float) -> float:
    delta = 6.0 / 29.0
    if t > delta**3:
        return float(t ** (1.0 / 3.0))
    else:
        return t / (3.0 * delta**2) + 4.0 / 29.0


def _lab_f_inv(t: float) -> float:
    delta = 6.0 / 29.0
    if t > delta:
        return t**3
    else:
        return 3.0 * delta**2 * (t - 4.0 / 29.0)


def scale_to_lab_range(value: float, min_val: float, max_val: float, lab_component: str) -> float:
    if lab_component == "l":
        target_min, target_max = 0.0, 100.0
    elif lab_component in ["a", "b"]:
        target_min, target_max = -128.0, 127.0
    else:
        raise ValueError(f"Unknown Lab component: {lab_component}")

    if max_val == min_val:
        return (target_min + target_max) / 2.0

    normalized = (value - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return float(np.clip(scaled, target_min, target_max))
