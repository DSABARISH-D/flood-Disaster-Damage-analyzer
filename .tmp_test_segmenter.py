from models.segmenter import FloodSegmenter
from PIL import Image
print('Creating FloodSegmenter...')
seg = FloodSegmenter()
print('Processor:', type(seg.processor))
print('Model:', None if seg.model is None else type(seg.model))
# Quick run on a small blank image to ensure segment() doesn't crash when model present
if seg.model is not None:
    img = Image.new('RGB', (512, 512), color=(0,0,255))
    res = seg.segment(img)
    print('Segmentation keys:', list(res.keys()))
else:
    print('Model not loaded; skipping segmentation test')
