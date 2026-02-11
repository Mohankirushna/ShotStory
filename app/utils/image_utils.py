"""Image processing utilities."""
import io
import base64
import logging
from PIL import Image
from app.config import settings

# Register HEIF/HEIC support (common on macOS / iPhones)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def process_uploaded_image(contents: bytes) -> Image.Image:
    """Open uploaded image bytes and resize if too large."""
    if not contents or len(contents) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    logger.info("Processing uploaded image (%d bytes)", len(contents))

    buf = io.BytesIO(contents)

    try:
        image = Image.open(buf)
    except Exception as exc:
        # Log first few bytes for debugging
        header = contents[:16].hex()
        raise ValueError(
            f"Cannot open image (header: {header}). "
            f"Make sure the file is a valid PNG, JPEG, or WebP image. Error: {exc}"
        ) from exc

    # Convert palette / RGBA / LA / etc. to RGB
    if image.mode not in ("RGB",):
        image = image.convert("RGB")

    max_dim = settings.IMAGE_RESIZE_MAX
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    logger.info("Image ready: %s, %s", image.size, image.mode)
    return image


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
