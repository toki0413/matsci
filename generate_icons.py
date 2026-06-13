from pathlib import Path
from PIL import Image, ImageDraw

icons_dir = Path("desktop/src-tauri/icons")
icons_dir.mkdir(parents=True, exist_ok=True)

def make_image(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (30, 64, 175, 255))
    draw = ImageDraw.Draw(img)
    # Draw a simple white "M" letter
    margin = size // 8
    thickness = max(1, size // 16)
    # left vertical
    draw.line([(margin, margin), (margin, size - margin)], fill="white", width=thickness)
    # right vertical
    draw.line([(size - margin, margin), (size - margin, size - margin)], fill="white", width=thickness)
    # two diagonals meeting in middle bottom-ish
    mid = size // 2
    bottom = size - margin
    draw.line([(margin, margin), (mid, bottom)], fill="white", width=thickness)
    draw.line([(size - margin, margin), (mid, bottom)], fill="white", width=thickness)
    return img

# PNG icons
for size in [32, 128]:
    make_image(size).save(icons_dir / f"{size}x{size}.png")
# high-DPI 128@2x is 256x256
make_image(256).save(icons_dir / "128x128@2x.png")

# ICO with common sizes
ico_sizes = [16, 24, 32, 48, 64, 128, 256]
images = [make_image(s) for s in ico_sizes]
images[0].save(icons_dir / "icon.ico", sizes=[(img.width, img.height) for img in images], append_images=images[1:])

# Minimal ICNS for macOS bundles: Pillow can read but not write ICNS.
# Create a 1024x1024 PNG that can be converted by tauri icon command later.
make_image(1024).save(icons_dir / "icon.png")

print("Generated placeholder icons in", icons_dir)
