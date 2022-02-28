import json
import torch

from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval(); # turning off the dropout

with open("paragraphs.json","r") as f:
    for line in f:
        jsn = json.loads(line)
        paragraph = jsn['body']
        words = jsn['words']
        
        text = paragraph.replace("______", "[MASK]")

       
def fill_the_gaps(text):
   text = '[CLS] ' + text + ' [SEP]'
   tokenized_text = tokenizer.tokenize(text)
   indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
   segments_ids = [0] * len(tokenized_text)
   tokens_tensor = torch.tensor([indexed_tokens])
   segments_tensors = torch.tensor([segments_ids])
   with torch.no_grad():
      predictions = model(tokens_tensor, segments_tensors)
   results = []
   for i, t in enumerate(tokenized_text):
       if t == '[MASK]':
           predicted_index = torch.argmax(predictions[0, i]).item()
           predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
           results.append(predicted_token)
   return results
       