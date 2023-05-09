# LIQUID
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/liquid-a-framework-for-list-question/question-answering-on-multispanqa)](https://paperswithcode.com/sota/question-answering-on-multispanqa?p=liquid-a-framework-for-list-question) <br>
This is the official repository for the paper "[**LIQUID: A Framework for List Question Answering Dataset Generation**](https://arxiv.org/abs/2302.01691)" (presented at [***AAAI 2023***](https://aaai.org/Conferences/AAAI-23/)). This repository provides the implementation of the LIQUID model, guidelines on how to run the model to synthesize list QA data. Also, you can download generated datasets without having to create them from scratch (see **[here](#data-downloads)**).


## Quick Links

* [Overview](#overview)
* [Data Downloads](#data-downloads)
* [Requirements](#requirements)
* [Dataset Generation](#dataset-generation)
* [List Question Answering](#list-question-answering)
* [Reference](#reference)
* [Contact](#contact)

## Overview

LIQUID is an automated framework for generating list QA datasets from unlabeled corpora. Generated datasets by LIQUID can be used to improve list QA performance by supplementing insufficient human-labeled data. When training a list QA model using the generated data and then fine-tuning it on the target training data, we achieved a new **state-of-the-art** performance on **[MultiSpanQA](https://multi-span.github.io/)** and outperformed baselines on several benchmakrs including **[Quoref](https://arxiv.org/abs/1908.05803)** and **[BioASQ](http://bioasq.org/)**.

![Model-1](https://user-images.githubusercontent.com/72010172/185115620-3dbd69cd-5e37-4da0-acd6-e9dfb9ab0021.png)

LIQUID comprises the following four stages (please refer to **[our paper](https://arxiv.org/abs/2302.01691)** for details).

* (1) Answer extraction: the named entities belonging to the same entity type (e.g., organization type) in a summary are extracted by an NER model and used as candidate answers. 
* (2) Question generation: the candidate answers and the original passage are fed into a QG model to generate list questions. 
* (3) Iterative filtering: incorrect answers (e.g., Hanszen) are iteratively filtered based on the confidence score assigned by a QA model. 
* (4) Answer expansion: correct but omitted answers (e.g., Yale) are identified by the QA model.

## Data Downloads

Use the links below to download the synthetic datasets without having to create a dataset from scratch. ✶ indicates they are the same data used in our experiments. Our data format follows that of SQuAD-v1.1.

|              Name              | Corpus | Size | Link |
|:----------------------------------|:--------|:--------|:--------|
| liquid-wiki-140k (✶) | Wikipedia | 140k | http://nlp.dmis.korea.edu/projects/liquid-lee-et-al-2023/liquid-wiki-140k.json |
| liquid-pubmed-140k (✶) | PubMed | 140k | http://nlp.dmis.korea.edu/projects/liquid-lee-et-al-2023/liquid-pubmed-140k.json |

## Requirements

Download this repository and set up an environment as follows.

```bash
# Clone the repository
git clone https://github.com/sylee0520/LIQUID.git
cd LIQUID

# Create a conda virtual environment
conda create -n liquid python=3.8
conda activate liquid

# Install all requirements
pip install -r requirements.txt --no-cache-dir
```

### Unlabeled Corpus

Download an unlabeled source corpus to be annotated and extract/unpack it to the correct directory. Choose either Wikipedia or PubMed depending on your target domain. ✶ indicates they are the same data used in our experiments.

|              Description              | Directory | Link |
|:----------------------------------|:--------|:--------|
| 2018-12-20 version of **Wikipedia** (✶) | `./data/unlabeled/wiki/` | http://nlp.dmis.korea.edu/projects/liquid-lee-et-al-2023/wiki181220.zip |
| 2019-01-02 version of **PubMed** (✶) | `./data/unlabeled/pubmed/` | http://nlp.dmis.korea.edu/projects/liquid-lee-et-al-2023/pubmed190102.zip |

Note that passages in each file have not been shuffled. You will have to randomly sample passages from the entire corpus files (e.g., "0000.json" to "5620.json" for Wikipedia) if you want to use sampled passages.

### NER Models

In LIQUID, two types of NER models are used to extract candidate answers for the *general* and *biomedical* domains, respectively. Please refer to the instructions below to install the NER models.

* For the *general* domain, run `python -m spacy download en_core_web_sm` to install **spaCy** NER system.
* For the *biomedican* domain, install **BERN2** from the official GitHub repository (**[link](https://github.com/dmis-lab/BERN2)**). After installation is complete, refer to the instructions below and run the model in the background. Note that you need to create a new conda environment for BERN2, instead of reusing the environment for LIQUID.

```bash
# Run BERN2 model
export CUDA_VISIBLE_DEVICES=0
conda activate BERN2
cd BERN2/scripts

# For Linux and MacOS
bash run_bern2.sh

# For Windows
bash run_bern2_windows.sh
```

## Dataset Generation

Once you have installed all the requirements, you are ready to create your list QA datasets. Please see the example script below. 

```bash
export CUDA_VISIBLE_DEVICES=0
export DATA_FILE=./data/unlabeled/wiki/0000.json
export OUTPUT_FILE=./data/synthetic/wiki/0000.json
python generate.py \
    	--data_file ${DATA_FILE} \
    	--output_file ${OUTPUT_FILE} \
    	--batch_size 8 \
    	--summary_min_length 64 \
    	--summary_max_length 128 \
    	--summary_model_name_or_path facebook/bart-large-cnn \
    	--qg_min_length 64 \
    	--qg_max_length 128 \
    	--qg_model_name_or_path mrm8488/t5-base-finetuned-question-generation-ap \
    	--qa_model_name_or_path thatdramebaazguy/roberta-base-squad \
    	--do_summary \
    	--device 0
```

### Argument Description
- `batch_size`: Number of passages to process simultaneously in one batch.
- `summary_min_length`, `summary_max_length`, `qg_min_length`, and `qg_max_length`: Minimum and maximum lengths of the output summary and question, respectively.
- `summary_model_name_or_path`, `qg_model_name_or_path`, and `qa_model_name_or_path`: Model path for loading the summarization model, question-generation model, and question-answering model, respectively. For the *biomedical* domain, you can use `dmis-lab/biobert-base-cased-v1.1-squad` as the QA model.
- `is_biomedical`: Use this option when the target domain is biomedicine.
- `do_summary`: (**Recommended**) Use this option if you want to summarize input passages and extract candidate answers from the summaries.
- `device`: Set to `0` if you want to use our framework on GPU; otherwise `-1`.

## List Question Answering

To be updated soon.


## Reference
Please cite our paper if it is helpful or relevant to your work.

```bash
@article{lee2023liquid,
  title={LIQUID: A Framework for List Question Answering Dataset Generation},
  author={Lee, Seongyun and Kim, Hyunjae and Kang, Jaewoo},
  journal={arXiv preprint arXiv:2302.01691},
  year={2023}
}
```

## Contact
Feel free to email us (`sy-lee@korea.ac.kr` and `hyunjae-kim@korea.ac.kr`) if you have any!
