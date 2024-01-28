import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# symptoms2disease
symptoms2disease_model_path = 'symptoms2disease_gpt2_fine-tuned.pth'
symptoms2disease_model = GPT2LMHeadModel.from_pretrained('gpt2')
symptoms2disease_model.load_state_dict(torch.load(symptoms2disease_model_path, map_location=device))
symptoms2disease_model = symptoms2disease_model.to(device)

# disease2diagnosis
disease2diagnosis_model_path = 'disease2diagnosis_gpt2_fine-tuned.pth'
disease2diagnosis_model = GPT2LMHeadModel.from_pretrained('gpt2')
disease2diagnosis_model.load_state_dict(torch.load(disease2diagnosis_model_path, map_location=device))
disease2diagnosis_model = disease2diagnosis_model.to(device)

while True:
    input_symptoms = input("Enter your symptoms(enter exit to exit): ")
    if input_symptoms.lower() == 'exit':
        break

    # symptoms2disease
    symptoms_input_ids = tokenizer.encode(input_symptoms, return_tensors='pt').to(device)
    symptoms_attention_mask = torch.ones(symptoms_input_ids.shape, dtype=torch.long, device=device)

    predicted_disease_ids = symptoms2disease_model.generate(
        symptoms_input_ids,
        attention_mask=symptoms_attention_mask,
        max_length=50,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )

    generated_text = tokenizer.decode(predicted_disease_ids[0], skip_special_tokens=True)
    predicted_disease = generated_text.split('[SEP]')[1].strip() if '[SEP]' in generated_text else "No prediction"
    print(f"Predicted Disease: {predicted_disease}")

    # disease2diagnosis
    disease_input_ids = tokenizer.encode(predicted_disease, return_tensors='pt').to(device)
    disease_attention_mask = torch.ones(disease_input_ids.shape, dtype=torch.long, device=device)

    diagnosis_output = disease2diagnosis_model.generate(
        disease_input_ids,
        attention_mask=disease_attention_mask,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.5,
        top_k=20
    )

    diagnosis_advice = tokenizer.decode(diagnosis_output[0], skip_special_tokens=True)
    print("Diagnosis: ", diagnosis_advice)
