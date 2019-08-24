#%%
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM



USE_GPU = 1
# Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')


# Load pre-trained model tokenizer (vocabulary)

# models: 'bert-base-uncased'
pretrained_model = 'bert-base-multilingual-cased'

tokenizer = BertTokenizer.from_pretrained(pretrained_model)



#%%
# Tokenize input

text = '''Będąc młodym programistą (hoho), czytałem "Dziady" w 1983r. się. następnie(później) zrobiłem doktorat w 05.06.2018 roku a o 
17:53 idę trenować. Kupiłem czarno-brązowy ciągnik. Po powrocie z USA gdzie  spotkałem Barack Obama w New York i zachód. Poszli razem w czwórkę 
do sklepu i zobaczywszy kotem zrobili samochodem.
'''
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

#%%

text = "[CLS] Ja i Marcin poszliśmy we wtorek na ryby ? [SEP] Marcin złowił szczupaka a ja dwa okonie. [SEP]"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)


# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 17
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'Ja', 'i', 'Marcin', 'pos', '##zli', '##ś', '##my', 'we', 'w', '##tore', '##k', 'na', 'ry', '##by', '?', '[SEP]', '[MASK]', 'zł', '##owi', '##ł', 'sz', '##czu', '##paka', 'a', 'ja', 'dwa', 'oko', '##nie', '.', '[SEP]']


# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])





#%%
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

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'Marcin'

top_k_predictions = torch.topk(predictions[0, masked_index],5)[1]
top_k_predictions = top_k_predictions.cpu().numpy()

top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_predictions)
print(top_k_tokens)




# Maks whole word
text = "[CLS] Ja i Marcin poszliśmy we wtorek na ryby ? [SEP] Marcin złowił szczupaka a ja dwa okonie. [SEP]"
tokenized_text2 = tokenizer.tokenize(text)

mask1 = 18
mask2 = 19
mask3=20
tokenized_text2[mask1] = '[MASK]'
tokenized_text2[mask2] = '[MASK]'
tokenized_text2[mask3] = '[MASK]'
#assert tokenized_text2 == ['[CLS]', 'Ja', 'i', 'Marcin', 'pos', '##zli', '##ś', '##my', 'we', 'w', '##tore', '##k', 'na', 'ry', '##by', '?', '[SEP]', 'Marcin', 'zł', '##owi', '##ł', 'sz', '##czu', '##paka', 'a', 'ja', 'dwa', '[MASK]', '[MASK]', '.', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text2)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to(device)
segments_tensors = segments_tensors.to(device)


# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, mask1]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)

predicted_index = torch.argmax(predictions[0, mask2]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)


top_k_predictions = torch.topk(predictions[0, mask1],5)[1]
top_k_predictions = top_k_predictions.cpu().numpy()
top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_predictions)
print(top_k_tokens)

top_k_predictions = torch.topk(predictions[0, mask2],5)[1]
top_k_predictions = top_k_predictions.cpu().numpy()
top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_predictions)
print(top_k_tokens)


