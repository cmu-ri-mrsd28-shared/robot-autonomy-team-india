"""Targets transcribed from Figure 2; row-major cells start at zero."""

PATTERNS = {
    "pickaxe": ("BBB", ".R.", ".R."),
    "axe": ("BB.", "BR.", ".R."),
    "box": ("BRB", "R.R", "BRB"),
    "bow": ("BR.", "B.R", "BR."),
}


def target_mask(rows):
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("A target must be a 3 by 3 grid")
    if any(ch not in ".BR" for row in rows for ch in row):
        raise ValueError("Unknown color")
    if any("".join(rows).count(color) > 4 for color in "BR"):
        raise ValueError("Insufficient block inventory")
    return sum(1 << i for i, ch in enumerate("".join(rows)) if ch != ".")
