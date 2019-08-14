
# How to predict whole word previously tokenized as sub-words for bert-base-multilingual-cased


import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)


USE_GPU = 1
# Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')


# Load pre-trained model tokenizer (vocabulary)
pretrained_model = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model)

text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
# ['[CLS]', 'Who', 'was', 'Jim', 'Hen', '##son', '?', '[SEP]', 'Jim', 'Hen', '##son', 'was', 'a', 'pu', '##ppet', '##eer', '[SEP]']

# Mask a token that we will try to predict back with `BertForMaskedLM`
mask1 = 13
mask2 = 14
mask3 = 15
tokenized_text[mask1] = '[MASK]'
tokenized_text[mask2] = '[MASK]'
tokenized_text[mask3] = '[MASK]'
assert tokenized_text == ['[CLS]', 'Who', 'was', 'Jim', 'Hen', '##son', '?', '[SEP]', 'Jim', 'Hen', '##son', 'was', 'a', '[MASK]', '[MASK]', '[MASK]', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0,0, 1, 1, 1, 1, 1, 1, 1,1,1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained(pretrained_model)
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to(device)
segments_tensors = segments_tensors.to(device)
model.to(device)

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# get predicted tokens

#prediction for mask1
predicted_index = torch.argmax(predictions[0, mask1]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)

#prediction for mask2
predicted_index = torch.argmax(predictions[0, mask2]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)

#prediction for mask3
predicted_index = torch.argmax(predictions[0, mask3]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)


#mm= torch.mean(predictions[0,18:21,:], dim=1)