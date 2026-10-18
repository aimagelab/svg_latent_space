import re
import random


def generate_similar_rgb(base_rgb, delta=50):
    return tuple(max(0, min(255, c + random.randint(-delta, delta))) for c in base_rgb)


def generate_random_rgb():
    return tuple(random.randint(0, 255) for _ in range(3))


color_set = [
    (64, 64, 64),
    (192, 192, 192),
    (128, 128, 128),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]


def replace_fill_with_random(path):
    def replacer(match):
        new_rgb = generate_random_rgb()
        return f"fill:rgb({new_rgb[0]},{new_rgb[1]},{new_rgb[2]});"

    return re.sub(r"fill:rgb\(([0-9]{1,3},[0-9]{1,3},[0-9]{1,3})\);", replacer, path)


def replace_fill_with_nearby(path):
    def replacer(match):
        base_rgb = tuple(map(int, match.group(1).split(",")))
        new_rgb = generate_similar_rgb(base_rgb)
        return f"fill:rgb({new_rgb[0]},{new_rgb[1]},{new_rgb[2]});"

    return re.sub(r"fill:rgb\(([0-9]{1,3},[0-9]{1,3},[0-9]{1,3})\);", replacer, path)


class ReplaceBlackWithRandom:
    def __init__(self, threshold=0.33):
        self.threshold = threshold

    def __call__(self, path):

        if "fill:rgb(0,0,0)" in path and random.random() < self.threshold:
            return replace_fill_with_random(path)
        else:
            return path


class ReplaceFillWithRandom:
    def __init__(self, threshold=0.33):
        self.threshold = threshold

    def __call__(self, path):
        if random.random() < self.threshold:
            return replace_fill_with_random(path)
        else:
            return path


class ReplaceFillWithNearby:
    def __init__(self, threshold=0.33):
        self.threshold = threshold

    def __call__(self, path):
        if random.random() < self.threshold:
            return replace_fill_with_nearby(path)
        else:
            return path


def switch_style(path):
    """Swap style and d attributes order in SVG path element."""
    swap_regex = (
        "(<path\\s+)style=([\"\\'])([^\"\\']*)(\\2)\\s+d=([\"\\'])([^\"\\']*)(\\5)"
    )
    swap_replacer = "\\1d=\\5\\6\\7 style=\\2\\3\\4"
    path = re.sub(swap_regex, swap_replacer, path)
    return path


class SwitchStyle:
    def __init__(self, threshold=0.5):
        # self.ratio = ratio
        pass

    def __call__(self, path):
        if random.random() < 0.5:
            path = switch_style(path)
        # print(path)
        return path


transformations = {
    "replace_black_with_random": ReplaceBlackWithRandom,
    "replace_fill_with_random": ReplaceFillWithRandom,
    "replace_fill_with_nearby": ReplaceFillWithNearby,
    "switch_style": SwitchStyle,
}
