# Table of Contents

1.  [CodeLlama Evaluation Guide](#codellama-evaluation-guide)
    * [Prerequisites](#prerequisites)
        * [Hugging Face Account and Access Token](#hugging-face-account-and-access-token)
        * [Meta's Agreement](#metas-agreement)
    * [Evaluation Steps](#evaluation-steps)
        * [Open Colab Notebook](#open-colab-notebook)
        * [Clone and Install HumanEval](#clone-and-install-humaneval)
        * [Using HumanEval](#using-humaneval)
        * [Run the evaluation notebook](#run-the-evaluation-notebook)
    * [Important Notes](#important-notes)
    * [Results](#results)

# CodeLlama Evaluation Guide

This document outlines the steps to evaluate CodeLlama using the HumanEval benchmark on Google Colab.

## Prerequisites

### Hugging Face Account and Access Token
  * Create an account on [Hugging Face](https://huggingface.co/).
  * Generate an access token with "write" permissions. Refer to the official documentation: [Hugging Face Security Tokens](https://huggingface.co/docs/hub/en/security-tokens).
### Metas Agreement
  * Accept Meta's agreement for CodeLlama-7b-Python-hf on Hugging Face: [CodeLlama-7b-Python-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Python-hf).

## Evaluation Steps

### Open Colab Notebook
  * Open the `codellama/evaluation.ipynb` notebook on Google Colab.
  * Ensure you select a runtime with a GPU.

### Clone and Install HumanEval
  * The notebook will first clone the HumanEval repository from GitHub: `https://github.com/openai/human-eval`.
  * It will then install the necessary dependencies for HumanEval.

### Using HumanEval
  * The `evaluation.ipynb` notebook contains code to interact with the HumanEval dataset.
  * The HumanEval dataset consists of programming problems and their corresponding solutions.
  * The dataset is stored in JSONL format, where each line is a JSON object.
  * Example files are located in `human-eval/data/example_problem.jsonl` and `human-eval/data/example_solutions.jsonl`.
  * Example code to load and view the data is included below:

  Method

  ```python
  import json

  # Read JSONL file line by line
  with open('human-eval/data/example_problem.jsonl', 'r') as file:
      data = [json.loads(line.strip()) for line in file if line.strip()]
      # or
      # line = file.readline() # read the first line.
      # data = json.loads(line)

  # Print the JSON data
  print(data)
  ```

  * Output:

  ```
  {'task_id': 'test/0', 'prompt': 'def return1():\n', 'canonical_solution': '    return 1', 'test': 'def check(candidate):\n    assert candidate() == 1', 'entry_point': 'return1'}
  ```

  * **Explanation of the JSON fields:**
      * `task_id`: Unique identifier for the problem.
      * `prompt`: The function signature and docstring.
      * `canonical_solution`: The ground truth solution.
      * `test`: The unit test to verify the solution.
      * `entry_point`: The name of the function to be evaluated.

### Run the evaluation notebook
  * Execute the cells in the `evaluation.ipynb` notebook sequentially.
  * The notebook will use the HumanEval dataset and the loaded CodeLlama model to generate code completions.
  * The generated completions will be evaluated against the test cases using the HumanEval evaluation script.
  * The evaluation results will be displayed in the notebook.

  ```python
  problems = read_problems()

  def generate_one_completion(prompt):
      # Replace this with your actual code generation logic
      # This could call an API or use a local model
      return "# Your generated code here\ndef solution():\n    return 42"

  num_samples_per_task = 1  # Start small for testing
  samples = [
      dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
      for task_id in problems
      for _ in range(num_samples_per_task)
  ]

  write_jsonl("samples.jsonl", samples)
  ```

**Key Components:**

* **`read_problems()`:**
    * Loads programming problems from the HumanEval dataset.
    * Provides access to prompts, solutions, and tests.
* **`generate_one_completion(prompt)`:**
    * This function is where your LLM is integrated.
    * It takes a HumanEval `prompt` as input.
    * **LLM Integration:**
        * Connect your LLM (e.g., via API or local inference) within this function.
        * Pass the `prompt` to the LLM.
        * **Zero-Shot Chain-of-Thought (CoT):**
            * Implement a Zero-shot CoT prompt within the LLM interaction. For example: "You are a Python expert..." before the Human-eval prompt.
        * **Context Management:**
            * **No Memory:**
                * If the LLM has no memory, you'll need to manage context explicitly.
                * Pass the previous CoT, the previous output, and the current HumanEval `prompt` to the LLM.
                * This may be limited by the LLM's context window size.
            * **Memory:**
                * If the model had memory, then only the Cot and current prompt would be needed.
    * **Output Saving:**
        * Save the LLM's output (generated code) to a `.jsonl` file. Save outputs in separate document.
* **Sample Generation:**
    * Iterates through problems and generates code completions.
    * `num_samples_per_task` controls the number of samples per problem.
* **`write_jsonl("samples.jsonl", samples)`:**
    * Saves generated completions in the required JSON Lines format.
    * Creates the input file for the HumanEval evaluation script.

**Model Correctness Evaluation:**

* Use the `evaluate_functional_correctness` script to assess the LLM's code generation accuracy.

* **`evaluate_functional_correctness output1.jsonl` and `evaluate_functional_correctness output2.jsonl`:**
    * Evaluates the two output files.
    * Produces `output1_results.jsonl` and `output2_results.jsonl`.
    * These files indicate pass/fail and execution results ("passed", "timed out", "failed").
* **Metric Calculation:**
    * Parse `output1_results.jsonl` and `output2_results.jsonl`.
    * Calculate the following metrics:
        * **t1:** Number of problems passed in `output1.jsonl`.
        * **t2:** Number of problems passed in `output2.jsonl`.
        * **Δt (delta t):** Difference between t2 and t1 (t2 - t1).
        * **tic:** Number of problems passed in `output1.jsonl` and failed `output2.jsonl`.
        * **tci:** Number of problems passed failed in `output1.jsonl` and passed in `output2.jsonl`.
        
## Loading and Using the Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/CodeLlama-7b-Python-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=500,  # Adjust based on your needs
        temperature=0.
```
## Important Notes

* Ensure you have a stable internet connection for downloading the model and dependencies.
* GPU acceleration is highly recommended for faster evaluation.
* The evaluation process may take some time, depending on the model and the size of the dataset.
* Make sure you have accepted the Meta agreement, otherwise the model will not be downloadable.
* Make sure you have the correct huggingface token and that it is properly set in the notebook.

## Results
| Metric | Accuracy|
|----------|----------|
| Accuracy@t1 | 27.44 |
| Accuracy@t2 | 28.68 |
| delta(t1,t2) | 1.29 |
| delta(t1,t2) i to c | 1.829 |.
| delta(t2,t1) c to i | 0.609 |.
