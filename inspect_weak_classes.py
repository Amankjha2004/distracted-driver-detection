
from pathlib import Path
from PIL import Image, ImageDraw
import random
import math

ROOT = Path(__file__).resolve().parent
VAL_DIR = ROOT / "outputs" / "yolov12_cnn_v2" / "dataset_v2" / "val"
OUTPUT = ROOT / "weak_classes_contact_sheet.jpg"

CLASSES = [
    "fatigue_somnolence",
    "singing_or_dancing",
    "smoking",
    "taking_picture",
]

random.seed(42)
thumb_w, thumb_h = 220, 170
label_h = 35
columns = 4
samples_per_class = 4

tiles = []

for class_name in CLASSES:
    folder = VAL_DIR / class_name
    images = []

    if folder.exists():
        images = [
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]

    selected = random.sample(images, min(samples_per_class, len(images)))

    for path in selected:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb_w, thumb_h))
                tile = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
                x = (thumb_w - im.width) // 2
                y = (thumb_h - im.height) // 2
                tile.paste(im, (x, y))

                draw = ImageDraw.Draw(tile)
                draw.text((5, thumb_h + 3), class_name, fill="black")
                draw.text((5, thumb_h + 18), path.name[:30], fill="black")
                tiles.append(tile)
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

if not tiles:
    raise RuntimeError(f"No images found under {VAL_DIR}")

rows = math.ceil(len(tiles) / columns)
sheet = Image.new(
    "RGB",
    (columns * thumb_w, rows * (thumb_h + label_h)),
    "#dddddd",
)

for i, tile in enumerate(tiles):
    x = (i % columns) * thumb_w
    y = (i // columns) * (thumb_h + label_h)
    sheet.paste(tile, (x, y))

sheet.save(OUTPUT)
print(f"Saved contact sheet: {OUTPUT}")
print(f"Images included: {len(tiles)}")
