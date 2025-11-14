# Resume Review Pipeline with GPT-OSS-20B

## Project Goal

Build a batch inference pipeline that processes resume "about me" sections using the GPT-OSS-20B model from Hugging Face to classify them as "good" or "bad".

## Technical Stack

- **Model**: GPT-OSS-20B (via Hugging Face Transformers)
- **Input**: `test_dataset.csv` containing "about_me" resume sections
- **Prompt**: System and user prompts defined in `prompt.py`
- **Processing**: Batch inference with batch size of 5

## Pipeline Components

1. **Input Data**: Load `test_dataset.csv` with 20 resume "about me" sections
2. **Prompt Template**: Use system and user prompts from `prompt.py` to format each input
3. **Model Inference**: Process inputs in batches of 5 using GPT-OSS-20B
4. **Output Processing**: Generate CSV with the following columns:
   - Original `about_me` text
   - Original `Human_flag` (ground truth)
   - `raw_output`: Complete model response (everything the model returns)
   - `final_channel`: Extracted content from the final channel from the model output.

## Implementation Notes

- Use Hugging Face Transformers library for model loading and inference
- Implement efficient batching to process 5 samples at a time
- Handle model output parsing to extract the final classification from the raw response
- Compare model predictions against human labels for evaluation