from PIL import Image, ImageOps
import cv2
import io
import os
import numpy as np
import random

def sample(rounds, img):
    interpolation = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
                     cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT,
                     cv2.INTER_NEAREST_EXACT]
    scale = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    # Store original dimensions
    original_h, original_w = img.shape[:2]

    img_back = img.copy()
    for _ in range(2 ** rounds):
      scale_random = random.randint(0, len(scale) - 1)

      img_small = cv2.resize(img_back, dsize = None, fx = scale[scale_random], fy = scale[scale_random],
                             interpolation = interpolation[random.randint(0, len(interpolation) - 1)])
      img_back = cv2.resize(img_small, dsize = None, fx = 1 / scale[scale_random], fy = 1 / scale[scale_random],
                            interpolation = interpolation[random.randint(0, len(interpolation) - 1)])

    # CRITICAL: Ensure we return to original dimensions (rounding can cause drift)
    if img_back.shape[:2] != (original_h, original_w):
        img_back = cv2.resize(img_back, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    return img_back

def compression(rounds, img):
  # Store original size
  original_size = img.size  # (width, height)

  if img.mode != 'RGB':
        img = img.convert('RGB')

  img.save("temp.jpeg", "JPEG")

  for _ in range(2 ** rounds):
    format = random.randint(0, 3)

    with Image.open("temp.jpeg") as temp_img:
      match format:
        case 0:
          temp_file = ("temp.bmp")
        case 1:
          temp_file = ("temp.tiff")
        case 2:
          temp_file = ("temp.webp")
        case 3:
          temp_file = ("temp.png")

      temp_img.save(temp_file)

      with Image.open(temp_file) as temp_img2:
            if temp_img2.mode != 'RGB':
                temp_img2 = temp_img2.convert('RGB')

            temp_img2.save("temp.jpeg")

  with Image.open("temp.jpeg") as result_img:
      # Ensure output matches original size
      if result_img.size != original_size:
          result_img = result_img.resize(original_size, Image.BILINEAR)

      # Load into memory before closing file
      result_img.load()
      final_img = result_img.copy()

  return final_img

def quantize(rounds, img):
    # Store original dimensions
    original_size = img.size  # (width, height) for PIL

    for i in range(rounds):
        quant_img = img.copy()
        quant_img = img.quantize(colors = 256 // (2 ** (i + 1)), method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)

        rgb_quant_img = quant_img.convert("RGB")

    # Ensure output matches original size
    if rgb_quant_img.size != original_size:
        rgb_quant_img = rgb_quant_img.resize(original_size, Image.BILINEAR)

    # Return as numpy array (RGB format, not BGR)
    arr_quant_img = np.array(rgb_quant_img)
    return arr_quant_img.astype(np.uint8)

def color_change(img):
    # Store original size
    original_size = img.size

    color_change_selector = random.randint(0, 3)
    match color_change_selector:
        case 0:
            img = ImageOps.invert(img)
        case 1:
            img = ImageOps.autocontrast(img)
        case 2:
            img = ImageOps.equalize(img)
        case 3:
            img = ImageOps.grayscale(img)
            # Convert grayscale back to RGB
            img = img.convert('RGB')

    # Ensure dimensions are preserved
    if img.size != original_size:
        img = img.resize(original_size, Image.BILINEAR)

    return img

def flipper(img):
    # Store original size
    original_size = img.size

    flipper_selector = random.randint(0,2)
    match flipper_selector:
        case 0:
            img = ImageOps.flip(img)
        case 1:
            img = ImageOps.mirror(img)
        case 2:
            img = ImageOps.flip(img)
            img = ImageOps.mirror(img)

    # Ensure dimensions are preserved (flip/mirror shouldn't change size, but just in case)
    if img.size != original_size:
        img = img.resize(original_size, Image.BILINEAR)

    return img