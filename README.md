# HGCR
Hypothesis Generation Context Retrieval

Code for Hypothesis Generation Context Retrieval module.

`subgraph_retrieval.py` file contains `GraphContextManager` class.
It takes the input from [Agatha Semantic Visualizer](https://github.com/IlyaTyagin/AgathaSemanticVisualizer) checkpoint and constructs a co-occurrence network, where concepts are UMLS terms and edges represent their co-occurrences in the corresponding MEDLINE abstracts. This network is used to retrieve a _coherent_ context, which is supposed to explain a potential connection between a pair of concepts of interest (hypothesis). The plausibility of this hypothesis is calculated with [AGATHA](https://github.com/IlyaTyagin/AGATHA-C-GP) system.

Work in progress.
Stay tuned!
