<<<<<<< HEAD
# **Model Details**
=======
# LLaMA Model Card
>>>>>>> 53011c3 (Fix typos in MODEL_CARD.md)

Meta developed and released the Llama 2 family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama-2-Chat, are optimized for dialogue use cases. Llama-2-Chat models outperform open-source chat models on most benchmarks we tested, and in our human evaluations for helpfulness and safety, are on par with some popular closed-source models like ChatGPT and PaLM.

**Model Developers** Meta

**Variations** Llama 2 comes in a range of parameter sizes — 7B, 13B, and 70B — as well as pretrained and fine-tuned variations.

**Input** Models input text only.

**Output** Models generate text only.

**Model Architecture** Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

||Training Data|Params|Context Length|GQA|Tokens|LR|
|---|---|---|---|---|---|---|
Llama 2|*A new mix of publicly available online data*|7B|4k|&#10007;|2.0T|3.0 x 10<sup>-4</sup>
Llama 2|*A new mix of publicly available online data*|13B|4k|&#10007;|2.0T|3.0 x 10<sup>-4</sup>
Llama 2|*A new mix of publicly available online data*|70B|4k|&#10004;|2.0T|1.5 x 10<sup>-4</sup>

**Llama 2 family of models.** Token counts refer to pretraining data only. All models are trained with a global batch-size of 4M tokens. The 70B version uses Grouped-Query Attention (GQA) for improved inference scalability.

**Model Dates** Llama 2 was trained between January 2023 and July 2023.

**Status** This is a static model trained on an offline dataset. Future versions of the tuned models will be released as we improve model safety with community feedback.

**License** A custom commercial license is available at: [https://ai.meta.com/resources/models-and-libraries/llama-downloads/](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

**Research Paper** More information can be found in the paper "Llama-2: Open Foundation and Fine-tuned Chat Models", available at https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/.

**Where to send questions or comments about the model** Instructions on how to provide feedback or comments on the model can be found in the model [README](README.md).

# **Intended Use**
**Intended Use Cases** Llama 2 is intended for commercial and research use in English. Tuned models are intended for assistant-like chat, whereas pretrained models can be adapted for a variety of natural language generation tasks.

**Out-of-scope Uses** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in any other way that is prohibited by the Acceptable Use Policy and Llama 2 Community License. Use in languages other than English**. 

**Note: Developers may fine-tune Llama 2 models for languages beyond English provided they comply with the Llama 2 Community License and the Acceptable Use Policy.

# **Hardware and Software**
**Training Factors** We used custom training libraries, Meta's Research Super Cluster, and production clusters for pretraining. Fine-tuning, annotation, and evaluation were also performed on third-party cloud compute.

**Carbon Footprint** Pretraining utilized a cumulative 3.3M GPU hours of computation on hardware of type A100-80GB (TDP of 350-400W). Estimated total emissions were 539 tCO2eq, 100% of which were offset by Meta’s sustainability program.

||Time (GPU hours)|Power Consumption (W)|Carbon Emitted(tCO<sub>2</sub>eq)|
|---|---|---|---|
|Llama 2 7B|184320|400|31.22|
|Llama 2 13B|368640|400|62.44|
|Llama 2 70B|1720320|400|291.42|
|Total|3311616||539.00|

**CO<sub>2</sub> emissions during pretraining.** Time: total GPU time required for training each model. Power Consumption: peak power capacity per GPU device for the GPUs used adjusted for power usage efficiency. 100% of the emissions are directly offset by Meta's sustainability program, and because we are openly releasing these models, the pretraining costs do not need to be incurred by others.

# **Training Data**
**Overview** Llama 2 was pretrained on 2 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over one million new human-annotated examples. Neither the pretraining nor the fine-tuning datasets include Meta user data.

**Data Freshness** The pretraining data has a cutoff of September 2022, but some tuning data is more recent, up to July 2023.

# **Evaluation Results**

In this section, we report the results for the Llama 1 and Llama 2 models on standard academic benchmarks.
For all the evaluations, we use our internal evaluations library.

|Model|Size|Code|Commonsense Reasoning|World Knowledge|Reading Comprehension|Math|MMLU|BBH|AGI Eval|
|---|---|---|---|---|---|---|---|---|---|
|Llama 1|7B|14.1|60.8|46.2|58.5|6.95|35.1|30.3|23.9|
|Llama 1|13B|18.9|66.1|52.6|62.3|10.9|46.9|37.0|33.9|
|Llama 1|33B|26.0|70.0|58.4|67.6|21.4|57.8|39.8|41.7|
|Llama 1|65B|30.7|70.7|60.5|68.6|30.8|63.4|43.5|47.6|
|Llama 2|7B|16.8|63.9|48.9|61.3|14.6|45.3|32.6|29.3|
|Llama 2|13B|24.5|66.9|55.4|65.8|28.7|54.8|39.4|39.1|
|Llama 2|70B|**37.5**|**71.9**|**63.6**|**69.4**|**35.2**|**68.9**|**51.2**|**54.2**|

**Overall performance on grouped academic benchmarks.** *Code:* We report the average pass@1 scores of our models on HumanEval and MBPP. *Commonsense Reasoning:* We report the average of PIQA, SIQA, HellaSwag, WinoGrande, ARC easy and challenge, OpenBookQA, and CommonsenseQA. We report 7-shot results for CommonSenseQA and 0-shot results for all other benchmarks. *World Knowledge:* We evaluate the 5-shot performance on NaturalQuestions and TriviaQA and report the average. *Reading Comprehension:* For reading comprehension, we report the 0-shot average on SQuAD, QuAC, and BoolQ. *MATH:* We report the average of the GSM8K (8 shot) and MATH (4 shot) benchmarks at the top 1.

|||TruthfulQA|Toxigen|
|---|---|---|---|
|Llama 1|7B|27.42|23.00|
|Llama 1|13B|41.74|23.08|
|Llama 1|33B|44.19|22.57|
|Llama 1|65B|48.71|21.77|
|Llama 2|7B|33.29|**21.25**|
|Llama 2|13B|41.86|26.10|
|Llama 2|70B|**50.18**|24.60|

**Evaluation of pretrained LLMs on automatic safety benchmarks.** For TruthfulQA, we present the percentage of generations that are both truthful and informative (the higher the better). For ToxiGen, we present the percentage of toxic generations (the smaller the better).


<<<<<<< HEAD
|||TruthfulQA|Toxigen|
|---|---|---|---|
|Llama-2-Chat|7B|57.04|**0.00**|
|Llama-2-Chat|13B|62.18|**0.00**|
|Llama-2-Chat|70B|**64.14**|0.01|
=======
<table>
    <thead>
            <tr>
            <th >LLaMA</th> <th colspan=6>Model hyper parameters </th>
            </tr>
            <tr>
            <th>Number of parameters</th><th>dimension</th><th>n heads</th><th>n layers</th><th>Learn rate</th><th>Batch size</th><th>n tokens</th>
            </tr>           
        </thead>
    <tbody>       
        <tr>
            <th>7B</th> <th>4096</th> <th>32</th> <th>32</th> <th>3.0E-04</th><th>4M</th><th>1T 
        </tr>
        <tr>
            <th>13B</th><th>5120</th><th>40</th><th>40</th><th>3.0E-04</th><th>4M</th><th>1T
        </tr>
        <tr>
            <th>33B</th><th>6656</th><th>52</th><th>60</th><th>1.5.E-04</th><th>4M</th><th>1.4T
        </tr>
        <tr>
            <th>65B</th><th>8192</th><th>64</th><th>80</th><th>1.5.E-04</th><th>4M</th><th>1.4T
        </tr>     
    </tbody>
</table>
>>>>>>> 53011c3 (Fix typos in MODEL_CARD.md)

**Evaluation of fine-tuned LLMs on different safety datasets.** Same metric definitions as above.

# **Ethical Considerations and Limitations**
Llama 2 is a new technology that carries risks with use. Testing conducted to date has been in English, and has not covered, nor could it cover all scenarios. For these reasons, as with all LLMs, Llama 2’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate, biased or other objectionable responses to user prompts. Therefore, before deploying any applications of Llama 2, developers should perform safety testing and tuning tailored to their specific applications of the model.

<<<<<<< HEAD
Please see the Responsible Use Guide available at [https://ai.meta.com/llama/responsible-use-guide/](https://ai.meta.com/llama/responsible-use-guide/)
=======
We present our results on eight standard common sense reasoning benchmarks in the table below. 
<table>
    <thead>
            <tr>
            <th>LLaMA</th>  <th colspan=9>Reasoning tasks </th>
            </tr>
            <tr>
            <th>Number of parameters</th> <th>BoolQ</th><th>PIQA</th><th>SIQA</th><th>HellaSwag</th><th>WinoGrande</th><th>ARC-e</th><th>ARC-c</th><th>OBQA</th><th>COPA</th>
            </tr>           
        </thead>
    <tbody>    
    <tr>   
        <th>7B</th><th>76.5</th><th>79.8</th><th>48.9</th><th>76.1</th><th>70.1</th><th>76.7</th><th>47.6</th><th>57.2</th><th>93
        </th>   
    <tr><th>13B</th><th>78.1</th><th>80.1</th><th>50.4</th><th>79.2</th><th>73</th><th>78.1</th><th>52.7</th><th>56.4</th><th>94
</th>
    <tr><th>33B</th><th>83.1</th><th>82.3</th><th>50.4</th><th>82.8</th><th>76</th><th>81.4</th><th>57.8</th><th>58.6</th><th>92
</th>
    <tr><th>65B</th><th>85.3</th><th>82.8</th><th>52.3</th><th>84.2</th><th>77</th><th>81.5</th><th>56</th><th>60.2</th><th>94</th></tr> 
    </tbody>
</table>

*Table 2 - Summary of LLama Model Performance on Reasoning tasks*


We present our results on bias in the table below. Note that lower value is better indicating lower bias. 


| No  | Category             | FAIR LLM |
| --- | -------------------- | -------- |
| 1   | Gender               | 70.6     |
| 2   | Religion             | 79       |
| 3   | Race/Color           | 57       |
| 4   | Sexual orientation   | 81       |
| 5   | Age                  | 70.1     |
| 6   | Nationality          | 64.2     |
| 7   | Disability           | 66.7     |
| 8   | Physical appearance  | 77.8     |
| 9   | Socioeconomic status | 71.5     |
|     | LLaMA Average        | 66.6     |

*Table 3 - Summary bias of our model output*



## Ethical considerations
**Data**
The data used to train the model is collected from various sources, mostly from the Web. As such, it contains offensive, harmful and biased content. We thus expect the model to exhibit such biases from the training data.

**Human life**
The model is not intended to inform decisions about matters central to human life, and should not be used in such a way.

**Mitigations**
We filtered the data from the Web based on its proximity to Wikipedia text and references. For this, we used a Kneser-Ney language model and a fastText linear classifier.

**Risks and harms**
Risks and harms of large language models include the generation of harmful, offensive or biased content. These models are often prone to generating incorrect information, sometimes referred to as hallucinations. We do not expect our model to be an exception in this regard.

**Use cases**
LLaMA is a foundational model, and as such, it should not be used for downstream applications without further investigation and mitigations of risks. These risks and potential fraught use cases include, but are not limited to: generation of misinformation and generation of harmful, biased or offensive content.
>>>>>>> 53011c3 (Fix typos in MODEL_CARD.md)
