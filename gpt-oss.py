from transformers import pipeline, AutoTokenizer
# NOTE: Removed the hypothetical model ID for a real one
model_id = "##" 

# 1. Manually load tokenizer to fix the padding, which is good practice for generation
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer, # Pass the configured tokenizer
    torch_dtype="auto",
    device_map="auto"
)

# 2. Define multiple chat inputs and format them manually into strings
#    This allows for true batching, as the pipeline expects a list of strings.

messages_1 = [{"role": "user", "content": "tell me a joke about cheese"}]
messages_2 = [{"role": "user", "content": "write a haiku about the moon"}]
messages_3 = [{"role": "user", "content": "explain batch processing in two sentences"}]

# Format each chat history into a single string using the tokenizer's chat template
# This is the string the model expects as input.
input_1 = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
input_2 = tokenizer.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)

# 3. Create the batch of strings
prompt_batch = [input_1, input_2]

# 4. Run the pipeline with the batch and the batch_size parameter
outputs = pipe(
    prompt_batch, 
    max_new_tokens=1024, 
    return_full_text=False, # Usually set to False for chat/instruction models
    batch_size=2 # Define the batch size
)

# 5. Print results to show it worked (outputs is a list of lists)
for i, output in enumerate(outputs):
    print(f"--- Prompt {i+1} ---")
    print(output[0]['generated_text'])

#Output from Model:

[[{'generated_text': "Commentary: We need to produce a joke about cheese. We'll comply.assistantfinalWhy did the cheese cross the road?To Join the band"}],
[{'generated_text': "Commentary: write a haiku about the moon. We'll comply.assistantfinalThe moom is beuatiful"}]]

#Required components (everything after assistantfinal is the final response):

#1. Why did the cheese cross the road?To Join the band
#2. The moom is beuatiful

