**Towards the Semi-Automated Population of the Ancient Greek WordNet**

⚠️ This repository is associated with a paper currently under submission. The results and methods may be subject to revision.


This repository contains the datasets, fine-tuning data, and code used to explore the use of large language models (LLMs) in semi-automating the population of the Ancient Greek WordNet (AGWN) through synonym generation.
To cite this work, please acknowledge the repository appropriately and check back for future updates upon publication.

**Description**

The experiments in the associated study evaluate three approaches to synonym generation for Ancient Greek:

- Zero-shot prompting
- Few-shot prompting
- Fine-tuning
  
The goal is to assist the semi-automatic enrichment of the Ancient Greek WordNet, by leveraging recent advances in natural language processing in a human-in-the-loop scenario, where a human annotator validates the outputs of the LLM.

**Datasets**

The Ancient Greek datasets were manually crafted and divided into:

- a (roughly) monosemous dataset, collecting lemmas associated to just one meaning according to the LSJ lexicon (Liddel et al., 1996).
- a (roughly) polysemous dataset, collecting lemmas associated to more than one meaning according to the LSJ lexicon (Liddel et al., 1996).

An English baseline is included for comparison with a high-resource and modern language. The following datasets were manually created:

- a (roughly) monosemous dataset, collecting translations of the AG dataset that are roughly monosemous English words.
- a (roughly) polysemous dataset, collecting translations of the AG dataset that are mainly polysemous English words.
  
Each of the datasets is balanced for part-of-speech (PoS), thus cointaining 10 nouns, 10 verbs, 10 adjectives and 10 adverbs.

**Fine-tuning data**

The fine-tuning data in jsonl format is divided into training data (80%) and validation data (20%). The data consists in sets of synonyms automatically collected from back-translation dictionaries of Ancient Greek.

**Fine-tuning script**

The Python script used for fine-tuning Mistral-Nemo on the fine-tuning data collected is available in this repository. The code relies on a GPU node to run.

**Contact**

For questions or collaboration, feel free to open an issue or contact:

Chiara Zanchi
Università di Pavia
chiara.zanchi@unipv.it

Beatrice Marchesi
Università di Pavia
beatrice.marchesi01@universitadipavia.it

**Acknowledgments**

The authors wish to express their sincere gratitude to Cristiano Chesi for granting access to the GPU node employed for fine-tuning, which was made available through the High-Performance Computing (HPC) cluster at IUSS Pavia.

Research for this study was funded through the European Union Funding Program – NextGenerationEU – Missione 4 Istruzione e ricerca - componente 2, investimento 1.1” Fondo per il Programma Nazionale della Ricerca (PNR) e Progetti di Ricerca di Rilevante Interesse Nazionale (PRIN)” progetto PRIN\_2022 2022YAPFNJ "Linked WordNets for Ancient Indo-European Languages" CUP F53D2300490 0001 - Dipartimento Studi Umanistici (Università di Pavia) and CUP J53D23008370001 – Dipartimento di Filologia classica, Papirologia e Linguistica storica (Università Cattolica del Sacro Cuore, Milano).
