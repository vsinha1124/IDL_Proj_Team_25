# IDL_Proj_Team_25

## Running and Finetuning Llama 3.2 Models

#### 1. Activate the environment and install dependencies
`source llama_env/bin/activate`

`pip install -r requirements.txt`

#### 2. Log into HuggingFace

`huggingface-cli login`

Then create an access token on the huggingface website and enter it.

#### 3. Change Model/Quanitization if Needed

Change the model to whatever model you want to use by editing the following line

`model_name = "meta-llama/Llama-3.2-3B"`

Remove/Comment out Quantization if needed:

`bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)`


#### 4. Change the finetuning dataset 

Change the following line in `finetuning.py` to whatever dataset you want to finetune on:
`dataset = load_dataset("tatsu-lab/alpaca")`

You can change it to a local dataset defined in a json file:
`dataset = load_dataset("json", data_files="file.json")`

#### 5. Change test examples

Change the following line in `finetuning.py` to whatever test examples you want to run before and after finetuning:

`test_prompts = [
    "Give three tips for staying healthy.",
    "How can I improve my productivity?",
    "Explain quantum computing in simple terms."
]`

#### 6. Run the finetuning testing script

`python finetuning.py`
