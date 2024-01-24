from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import torch
from tqdm import tqdm

# We load the model
model_path = "gpt2"  # I choose gpt2 as a training model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token 

# we open and read our data
csv_file = "data\SymptomsXanswer-chat.csv"
data = pd.read_csv(csv_file)


# we defined a new Tokenizer
def prepare_data(text):
    
    tokenized = tokenizer.encode(text, return_tensors='pt')
    return tokenized


total_loss = 0.0

# We choose the max sequence length
max_sequence_length = 128

for index, row in tqdm(data.iterrows(), total=len(data)):
    input_text = row['symptoms'] + ", " + row['chatxanswer']
    target_text = row['diseases']

    # Tokeniz the input and target texts with padding and truncation
    inputs = tokenizer.encode_plus(input_text, max_length=max_sequence_length, pad_to_max_length=True, truncation=True, return_tensors='pt')
    targets = tokenizer.encode_plus(target_text, max_length=max_sequence_length, pad_to_max_length=True, truncation=True, return_tensors='pt')

    input_ids = inputs['input_ids']
    target_ids = targets['input_ids']

    # We start the training 
    model.train()


    outputs = model(input_ids, labels=target_ids)
    loss = outputs.loss
    total_loss += loss.item()

    print(f"Verlust in Iteration {index+1}: {loss.item()}")

# we calculate the average lost for all iteration
average_loss = total_loss / len(data)
print(f"Durchschnittlicher Verlust: {average_loss}")

# Save our model
output_model_path = "gpt2_model"
model.save_pretrained(output_model_path)
