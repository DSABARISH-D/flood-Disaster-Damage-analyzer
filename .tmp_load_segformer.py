from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import config
print('Transformers version:', __import__('transformers').__version__)
model_id = config.SEGFORMER_MODEL_ID
print('Model ID:', model_id)
print('Downloading processor...')
proc = SegformerImageProcessor.from_pretrained(model_id)
print('Processor OK')
print('Downloading model (this may take a while)...')
model = SegformerForSemanticSegmentation.from_pretrained(model_id)
print('Model OK, num_labels=', model.config.num_labels)
