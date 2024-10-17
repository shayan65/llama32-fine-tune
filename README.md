
# Fine-tuning Llama 3.2 1B Model

This repository contains code for fine-tuning the Llama 3.2 1B model using Hugging Face's `transformers` library, `PEFT` for parameter-efficient fine-tuning, and LoRA (Low-Rank Adaptation). It demonstrates how to load, fine-tune, and run inference with the Llama 3.2 model.

## Requirements

Make sure you have the following libraries installed:

```bash
pip install transformers==4.44.2
pip install datasets
pip install accelerate
pip install peft
pip install bitsandbytes
pip install wandb
```

## Steps

### 1. Load the Pre-trained Model and Tokenizer

The pre-trained Llama 3.2 1B model is loaded from the Hugging Face model hub.

### 2. Prepare Dataset for Fine-tuning

You'll need a dataset to fine-tune the model on. The Hugging Face `datasets` library is used for loading datasets.

### 3. Set Up the Trainer

We configure the trainer and training arguments for fine-tuning the model.

### 4. Fine-tune the Model

The model is fine-tuned on the specified dataset using LoRA for efficient training.

### 5. Save the Fine-tuned Model

The fine-tuned model can be saved locally and optionally pushed to the Hugging Face model hub.

### 6. Run Inference

After fine-tuning, you can run inference using the fine-tuned model to generate text.

## Inference Example

Here’s how you can run inference with the fine-tuned model:

```python
inputs = tokenizer("Your test input", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## License

This project is licensed under the MIT License.
