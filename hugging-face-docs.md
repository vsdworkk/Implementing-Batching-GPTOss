# Using Transformers for Text generation


Text generation is the most popular application for large language models (LLMs). A LLM is trained to generate the next word (token) given some initial text (prompt) along with its own generated outputs up to a predefined length or when it reaches an end-of-sequence (`EOS`) token.

In Transformers, the [generate()](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) API handles text generation, and it is available for all models with generative capabilities. This guide will show you the basics of text generation with [generate()](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) and some common pitfalls to avoid.

> [!TIP]
> You can also chat with a model directly from the command line. ([reference](./conversations.md#transformers-cli))
> ```shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

## Default generate

Before you begin, it's helpful to install [bitsandbytes](https://hf.co/docs/bitsandbytes/index) to quantize really large models to reduce their memory usage.

```bash
!pip install -U transformers bitsandbytes
```

Bitsandbytes supports multiple backends in addition to CUDA-based GPUs. Refer to the multi-backend installation [guide](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend) to learn more.

Load a LLM with [from_pretrained()](/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) and add the following two parameters to reduce the memory requirements.

- `device_map="auto"` enables Accelerates' [Big Model Inference](./models#big-model-inference) feature for automatically initiating the model skeleton and loading and dispatching the model weights across all available devices, starting with the fastest device (GPU).
- `quantization_config` is a configuration object that defines the quantization settings. This examples uses bitsandbytes as the quantization backend (see the [Quantization](./quantization/overview) section for more available backends) and it loads the model in [4-bits](./quantization/bitsandbytes).

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", quantization_config=quantization_config)
```

Tokenize your input, and set the `padding_side()` parameter to `"left"` because a LLM is not trained to continue generation from padding tokens. The tokenizer returns the input ids and attention mask.

> [!TIP]
> Process more than one prompt at a time by passing a list of strings to the tokenizer. Batch the inputs to improve throughput at a small cost to latency and memory.

```py
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
```

Pass the inputs to [generate()](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) to generate tokens, and [batch_decode()](/docs/transformers/v4.57.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode) the generated tokens back to text.

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
"A list of colors: red, blue, green, yellow, orange, purple, pink,"
```

## Generation configuration

All generation settings are contained in [GenerationConfig](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationConfig). In the example above, the generation settings are derived from the `generation_config.json` file of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1). A default decoding strategy is used when no configuration is saved with a model.

Inspect the configuration through the `generation_config` attribute. It only shows values that are different from the default configuration, in this case, the `bos_token_id` and `eos_token_id`.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
model.generation_config
GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2
}
```

You can customize [generate()](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) by overriding the parameters and values in [GenerationConfig](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationConfig). See [this section below](#common-options) for commonly adjusted parameters.

```py
# enable beam search sampling strategy
model.generate(**inputs, num_beams=4, do_sample=True)
```

[generate()](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) can also be extended with external libraries or custom code:

1. the `logits_processor` parameter accepts custom [LogitsProcessor](/docs/transformers/v4.57.1/en/internal/generation_utils#transformers.LogitsProcessor) instances for manipulating the next token probability distribution;
2. the `stopping_criteria` parameters supports custom [StoppingCriteria](/docs/transformers/v4.57.1/en/internal/generation_utils#transformers.StoppingCriteria) to stop text generation;
3. other custom generation methods can be loaded through the `custom_generate` flag ([docs](generation_strategies.md/#custom-decoding-methods)).

Refer to the [Generation strategies](./generation_strategies) guide to learn more about search, sampling, and decoding strategies.

### Saving

Create an instance of [GenerationConfig](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationConfig) and specify the decoding parameters you want.

```py
from transformers import AutoModelForCausalLM, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("my_account/my_model")
generation_config = GenerationConfig(
    max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
)
```

Use [save_pretrained()](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationConfig.save_pretrained) to save a specific generation configuration and set the `push_to_hub` parameter to `True` to upload it to the Hub.

```py
generation_config.save_pretrained("my_account/my_model", push_to_hub=True)
```

Leave the `config_file_name` parameter empty. This parameter should be used when storing multiple generation configurations in a single directory. It gives you a way to specify which generation configuration to load. You can create different configurations for different generative tasks (creative text generation with sampling, summarization with beam search) for use with a single model.

```py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

translation_generation_config = GenerationConfig(
    num_beams=4,
    early_stopping=True,
    decoder_start_token_id=0,
    eos_token_id=model.config.eos_token_id,
    pad_token=model.config.pad_token_id,
)

translation_generation_config.save_pretrained("/tmp", config_file_name="translation_generation_config.json", push_to_hub=True)

generation_config = GenerationConfig.from_pretrained("/tmp", config_file_name="translation_generation_config.json")
inputs = tokenizer("translate English to French: Configuration files are easy to use!", return_tensors="pt")
outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

## Common Options

[generate()](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) is a powerful tool that can be heavily customized. This can be daunting for a new users. This section contains a list of popular generation options that you can define in most text generation tools in Transformers: [generate()](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate), [GenerationConfig](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationConfig), `pipelines`, the `chat` CLI, ...

| Option name | Type | Simplified description |
|---|---|---|
| `max_new_tokens` | `int` | Controls the maximum generation length. Be sure to define it, as it usually defaults to a small value. |
| `do_sample` | `bool` | Defines whether generation will sample the next token (`True`), or is greedy instead (`False`). Most use cases should set this flag to `True`. Check [this guide](./generation_strategies) for more information. |
| `temperature` | `float` | How unpredictable the next selected token will be. High values (`>0.8`) are good for creative tasks, low values (e.g. `<0.4`) for tasks that require "thinking". Requires `do_sample=True`. |
| `num_beams` | `int` | When set to `>1`, activates the beam search algorithm. Beam search is good on input-grounded tasks. Check [this guide](./generation_strategies) for more information. |
| `repetition_penalty` | `float` | Set it to `>1.0` if you're seeing the model repeat itself often. Larger values apply a larger penalty. |
| `eos_token_id` | `list[int]` | The token(s) that will cause generation to stop. The default value is usually good, but you can specify a different token. |

## Pitfalls

The section below covers some common issues you may encounter during text generation and how to solve them.

### Output length

[generate()](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) returns up to 20 tokens by default unless otherwise specified in a models [GenerationConfig](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationConfig). It is highly recommended to manually set the number of generated tokens with the `max_new_tokens` parameter to control the output length. [Decoder-only](https://hf.co/learn/nlp-course/chapter1/6?fw=pt) models returns the initial prompt along with the generated tokens.

```py
model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to(model.device)
```

<hfoptions id="output-length">
<hfoption id="default length">

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'
```

</hfoption>
<hfoption id="max_new_tokens">

```py
generated_ids = model.generate(**model_inputs, max_new_tokens=50)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

</hfoption>
</hfoptions>

### Decoding strategy

The default decoding strategy in [generate()](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) is *greedy search*, which selects the next most likely token, unless otherwise specified in a models [GenerationConfig](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationConfig). While this decoding strategy works well for input-grounded tasks (transcription, translation), it is not optimal for more creative use cases (story writing, chat applications).

For example, enable a [multinomial sampling](./generation_strategies#multinomial-sampling) strategy to generate more diverse outputs. Refer to the [Generation strategy](./generation_strategies) guide for more decoding strategies.

```py
model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to(model.device)
```

<hfoptions id="decoding">
<hfoption id="greedy search">

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

</hfoption>
<hfoption id="multinomial sampling">

```py
generated_ids = model.generate(**model_inputs, do_sample=True)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

</hfoption>
</hfoptions>

### Padding side

Inputs need to be padded if they don't have the same length. But LLMs aren't trained to continue generation from padding tokens, which means the `padding_side()` parameter needs to be set to the left of the input.

<hfoptions id="padding">
<hfoption id="right pad">

```py
model_inputs = tokenizer(
    ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 33333333333'
```

</hfoption>
<hfoption id="left pad">

```py
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model_inputs = tokenizer(
    ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 3, 4, 5, 6,'
```

</hfoption>
</hfoptions>

### Prompt format

Some models and tasks expect a certain input prompt format, and if the format is incorrect, the model returns a suboptimal output. You can learn more about prompting in the [prompt engineering](./tasks/prompting) guide.

For example, a chat model expects the input as a [chat template](./chat_templating). Your prompt should include a `role` and `content` to indicate who is participating in the conversation. If you try to pass your prompt as a single string, the model doesn't always return the expected output.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-alpha", device_map="auto", load_in_4bit=True
)
```

<hfoptions id="format">
<hfoption id="no format">

```py
prompt = """How many cats does it take to change a light bulb? Reply as a pirate."""
model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
input_length = model_inputs.input_ids.shape[1]
generated_ids = model.generate(**model_inputs, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
"Aye, matey! 'Tis a simple task for a cat with a keen eye and nimble paws. First, the cat will climb up the ladder, carefully avoiding the rickety rungs. Then, with"
```

</hfoption>
<hfoption id="chat template">

```py
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many cats does it take to change a light bulb?"},
]
model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
input_length = model_inputs.shape[1]
generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
"Arr, matey! According to me beliefs, 'twas always one cat to hold the ladder and another to climb up it an’ change the light bulb, but if yer looking to save some catnip, maybe yer can
```

</hfoption>
</hfoptions>

## Resources

Take a look below for some more specific and specialized text generation libraries.

- [Optimum](https://github.com/huggingface/optimum): an extension of Transformers focused on optimizing training and inference on specific hardware devices
- [Outlines](https://github.com/dottxt-ai/outlines): a library for constrained text generation (generate JSON files for example).
- [SynCode](https://github.com/uiuc-focal-lab/syncode): a library for context-free grammar guided generation (JSON, SQL, Python).
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference): a production-ready server for LLMs.
- [Text generation web UI](https://github.com/oobabooga/text-generation-webui): a Gradio web UI for text generation.
- [logits-processor-zoo](https://github.com/NVIDIA/logits-processor-zoo): additional logits processors for controlling text generation.


<EditOnGithub source="https://github.com/huggingface/transformers/blob/main/docs/source/en/llm_tutorial.md" />





# Pipeline

The [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) is a simple but powerful inference API that is readily available for a variety of machine learning tasks with any model from the Hugging Face [Hub](https://hf.co/models).

Tailor the [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) to your task with task specific parameters such as adding timestamps to an automatic speech recognition (ASR) pipeline for transcribing meeting notes. [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) supports GPUs, Apple Silicon, and half-precision weights to accelerate inference and save memory.

<Youtube id=tiZFewofSLM/>

Transformers has two pipeline classes, a generic [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) and many individual task-specific pipelines like [TextGenerationPipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.TextGenerationPipeline) or [VisualQuestionAnsweringPipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.VisualQuestionAnsweringPipeline). Load these individual pipelines by setting the task identifier in the `task` parameter in [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline). You can find the task identifier for each pipeline in their API documentation.

Each task is configured to use a default pretrained model and preprocessor, but this can be overridden with the `model` parameter if you want to use a different model.

For example, to use the [TextGenerationPipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.TextGenerationPipeline) with [Gemma 2](./model_doc/gemma2), set `task="text-generation"` and `model="google/gemma-2-2b"`.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}]
```

When you have more than one input, pass them as a list.

```py
from transformers import pipeline, infer_device

device = infer_device()

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device)
pipeline(["the secret to baking a really good cake is ", "a baguette is "])
[[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}],
 [{'generated_text': 'a baguette is 100% bread.\n\na baguette is 100%'}]]
```

This guide will introduce you to the [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline), demonstrate its features, and show how to configure its various parameters.

## Tasks

[Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) is compatible with many machine learning tasks across different modalities. Pass an appropriate input to the pipeline and it will handle the rest.

Here are some examples of how to use [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) for different tasks and modalities.

<hfoptions id="tasks">
<hfoption id="summarization">

```py
from transformers import pipeline

pipeline = pipeline(task="summarization", model="google/pegasus-billsum")
pipeline("Section was formerly set out as section 44 of this title. As originally enacted, this section contained two further provisions that 'nothing in this act shall be construed as in any wise affecting the grant of lands made to the State of California by virtue of the act entitled 'An act authorizing a grant to the State of California of the Yosemite Valley, and of the land' embracing the Mariposa Big-Tree Grove, approved June thirtieth, eighteen hundred and sixty-four; or as affecting any bona-fide entry of land made within the limits above described under any law of the United States prior to the approval of this act.' The first quoted provision was omitted from the Code because the land, granted to the state of California pursuant to the Act cite, was receded to the United States. Resolution June 11, 1906, No. 27, accepted the recession.")
[{'summary_text': 'Instructs the Secretary of the Interior to convey to the State of California all right, title, and interest of the United States in and to specified lands which are located within the Yosemite and Mariposa National Forests, California.'}]
```

</hfoption>
<hfoption id="automatic speech recognition">

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</hfoption>
<hfoption id="image classification">

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/vit-base-patch16-224")
pipeline(images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
[{'label': 'lynx, catamount', 'score': 0.43350091576576233},
 {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
  'score': 0.034796204417943954},
 {'label': 'snow leopard, ounce, Panthera uncia',
  'score': 0.03240183740854263},
 {'label': 'Egyptian cat', 'score': 0.02394474856555462},
 {'label': 'tiger cat', 'score': 0.02288915030658245}]
```

</hfoption>
<hfoption id="visual question answering">

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```

</hfoption>
</hfoptions>

## Parameters

At a minimum, [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) only requires a task identifier, model, and the appropriate input. But there are many parameters available to configure the pipeline with, from task-specific parameters to optimizing performance.

This section introduces you to some of the more important parameters.

### Device

[Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) is compatible with many hardware types, including GPUs, CPUs, Apple Silicon, and more. Configure the hardware type with the `device` parameter. By default, [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) runs on a CPU which is given by `device=-1`.

<hfoptions id="device">
<hfoption id="GPU">

To run [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) on a GPU, set `device` to the associated CUDA device id. For example, `device=0` runs on the first GPU.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=0)
pipeline("the secret to baking a really good cake is ")
```

You could also let [Accelerate](https://hf.co/docs/accelerate/index), a library for distributed training, automatically choose how to load and store the model weights on the appropriate device. This is especially useful if you have multiple devices. Accelerate loads and stores the model weights on the fastest device first, and then moves the weights to other devices (CPU, hard drive) as needed. Set `device_map="auto"` to let Accelerate choose the device.

> [!TIP]
> Make sure have [Accelerate](https://hf.co/docs/accelerate/basic_tutorials/install) is installed.
>
> ```py
> !pip install -U accelerate
> ```

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device_map="auto")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
<hfoption id="Apple silicon">

To run [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) on Apple silicon, set `device="mps"`.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="mps")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
</hfoptions>

### Batch inference

[Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) can also process batches of inputs with the `batch_size` parameter. Batch inference may improve speed, especially on a GPU, but it isn't guaranteed. Other variables such as hardware, data, and the model itself can affect whether batch inference improves speed. For this reason, batch inference is disabled by default.

In the example below, when there are 4 inputs and `batch_size` is set to 2, [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) passes a batch of 2 inputs to the model at a time.

```py
from transformers import pipeline, infer_device()

device = infer_device()

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device, batch_size=2)
pipeline(["the secret to baking a really good cake is", "a baguette is", "paris is the", "hotdogs are"])
[[{'generated_text': 'the secret to baking a really good cake is to use a good cake mix.\n\ni’'}],
 [{'generated_text': 'a baguette is'}],
 [{'generated_text': 'paris is the most beautiful city in the world.\n\ni’ve been to paris 3'}],
 [{'generated_text': 'hotdogs are a staple of the american diet. they are a great source of protein and can'}]]
```

Another good use case for batch inference is for streaming data in [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline).

```py
from transformers import pipeline, infer_device
from transformers.pipelines.pt_utils import KeyDataset
import datasets

device = infer_device()

# KeyDataset is a utility that returns the item in the dict returned by the dataset
dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

Keep the following general rules of thumb in mind for determining whether batch inference can help improve performance.

1. The only way to know for sure is to measure performance on your model, data, and hardware.
2. Don't batch inference if you're constrained by latency (a live inference product for example).
3. Don't batch inference if you're using a CPU.
4. Don't batch inference if you don't know the `sequence_length` of your data. Measure performance, iteratively add to `sequence_length`, and include out-of-memory (OOM) checks to recover from failures.
5. Do batch inference if your `sequence_length` is regular, and keep pushing it until you reach an OOM error. The larger the GPU, the more helpful batch inference is.
6. Do make sure you can handle OOM errors if you decide to do batch inference.

### Task-specific parameters

[Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) accepts any parameters that are supported by each individual task pipeline. Make sure to check out each individual task pipeline to see what type of parameters are available. If you can't find a parameter that is useful for your use case, please feel free to open a GitHub [issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml) to request it!

The examples below demonstrate some of the task-specific parameters available.

<hfoptions id="task-specific-parameters">
<hfoption id="automatic speech recognition">

Pass the `return_timestamps="word"` parameter to [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) to return when each word was spoken.

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline(audio="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac", return_timestamp="word")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.',
 'chunks': [{'text': ' I', 'timestamp': (0.0, 1.1)},
  {'text': ' have', 'timestamp': (1.1, 1.44)},
  {'text': ' a', 'timestamp': (1.44, 1.62)},
  {'text': ' dream', 'timestamp': (1.62, 1.92)},
  {'text': ' that', 'timestamp': (1.92, 3.7)},
  {'text': ' one', 'timestamp': (3.7, 3.88)},
  {'text': ' day', 'timestamp': (3.88, 4.24)},
  {'text': ' this', 'timestamp': (4.24, 5.82)},
  {'text': ' nation', 'timestamp': (5.82, 6.78)},
  {'text': ' will', 'timestamp': (6.78, 7.36)},
  {'text': ' rise', 'timestamp': (7.36, 7.88)},
  {'text': ' up', 'timestamp': (7.88, 8.46)},
  {'text': ' and', 'timestamp': (8.46, 9.2)},
  {'text': ' live', 'timestamp': (9.2, 10.34)},
  {'text': ' out', 'timestamp': (10.34, 10.58)},
  {'text': ' the', 'timestamp': (10.58, 10.8)},
  {'text': ' true', 'timestamp': (10.8, 11.04)},
  {'text': ' meaning', 'timestamp': (11.04, 11.4)},
  {'text': ' of', 'timestamp': (11.4, 11.64)},
  {'text': ' its', 'timestamp': (11.64, 11.8)},
  {'text': ' creed.', 'timestamp': (11.8, 12.3)}]}
```

</hfoption>
<hfoption id="text generation">

Pass `return_full_text=False` to [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) to only return the generated text instead of the full text (prompt and generated text).

[__call__()](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.TextGenerationPipeline.__call__) also supports additional keyword arguments from the [generate()](/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) method. To return more than one generated sequence, set `num_return_sequences` to a value greater than 1.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openai-community/gpt2")
pipeline("the secret to baking a good cake is", num_return_sequences=4, return_full_text=False)
[{'generated_text': ' how easy it is for me to do it with my hands. You must not go nuts, or the cake is going to fall out.'},
 {'generated_text': ' to prepare the cake before baking. The key is to find the right type of icing to use and that icing makes an amazing frosting cake.\n\nFor a good icing cake, we give you the basics'},
 {'generated_text': " to remember to soak it in enough water and don't worry about it sticking to the wall. In the meantime, you could remove the top of the cake and let it dry out with a paper towel.\n"},
 {'generated_text': ' the best time to turn off the oven and let it stand 30 minutes. After 30 minutes, stir and bake a cake in a pan until fully moist.\n\nRemove the cake from the heat for about 12'}]
```

</hfoption>
</hfoptions>

## Chunk batching

There are some instances where you need to process data in chunks.

- for some data types, a single input (for example, a really long audio file) may need to be chunked into multiple parts before it can be processed
- for some tasks, like zero-shot classification or question answering, a single input may need multiple forward passes which can cause issues with the `batch_size` parameter

The [ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) class is designed to handle these use cases. Both pipeline classes are used in the same way, but since [ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) can automatically handle batching, you don't need to worry about the number of forward passes your inputs trigger. Instead, you can optimize `batch_size` independently of the inputs.

The example below shows how it differs from [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline).

```py
# ChunkPipeline
all_model_outputs = []
for preprocessed in pipeline.preprocess(inputs):
    model_outputs = pipeline.model_forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs =pipeline.postprocess(all_model_outputs)

# Pipeline
preprocessed = pipeline.preprocess(inputs)
model_outputs = pipeline.forward(preprocessed)
outputs = pipeline.postprocess(model_outputs)
```

## Large datasets

For inference with large datasets, you can iterate directly over the dataset itself. This avoids immediately allocating memory for the entire dataset, and you don't need to worry about creating batches yourself. Try [Batch inference](#batch-inference) with the `batch_size` parameter to see if it improves performance.

```py
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline, infer_device
from datasets import load_dataset

device = infer_device()

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

Other ways to run inference on large datasets with [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) include using an iterator or generator.

```py
def data():
    for i in range(1000):
        yield f"My example {i}"

pipeline = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipeline(data()):
    generated_characters += len(out[0]["generated_text"])
```

## Large models

[Accelerate](https://hf.co/docs/accelerate/index) enables a couple of optimizations for running large models with [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline). Make sure Accelerate is installed first.

```py
!pip install -U accelerate
```

The `device_map="auto"` setting is useful for automatically distributing the model across the fastest devices (GPUs) first before dispatching to other slower devices if available (CPU, hard drive).

[Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) supports half-precision weights (torch.float16), which can be significantly faster and save memory. Performance loss is negligible for most models, especially for larger ones. If your hardware supports it, you can enable torch.bfloat16 instead for more range.

> [!TIP]
> Inputs are internally converted to torch.float16 and it only works for models with a PyTorch backend.

Lastly, [Pipeline](/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.Pipeline) also accepts quantized models to reduce memory usage even further. Make sure you have the [bitsandbytes](https://hf.co/docs/bitsandbytes/installation) library installed first, and then add `load_in_8bit=True` to `model_kwargs` in the pipeline.

```py
import torch
from transformers import pipeline, BitsAndBytesConfig

pipeline = pipeline(model="google/gemma-7b", dtype=torch.bfloat16, device_map="auto", model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)})
pipeline("the secret to baking a good cake is ")
[{'generated_text': 'the secret to baking a good cake is 1. the right ingredients 2. the right'}]
```


<EditOnGithub source="https://github.com/huggingface/transformers/blob/main/docs/source/en/pipeline_tutorial.md" />