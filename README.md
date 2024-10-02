1. Intro

This repository containes code for the thesis project "TinyML for bird song detection" in Data Science master program.

Low-power ARUs can be used to collect bird song recordings in various locations, operating for long periods of time without recharging. Because of their memory and energy limitations it essential for the DL models deployed on these devices to be small enough to run efficiently, balancing model size with classification performance.

In this projetct we implemented different DL architectures such as CNN, SqueezeNet, tiny Transformer and BNN, characterized by a small size, and apply the Post-Training Quantization (PTQ) technique to them. Additionally, we investigate the classification performance, latency and memory consumption of these models with both log Mel spectrograms and raw signal time-series as inputs.

2. Repository navigation

All model files are located in the folder "models".

In "notebooks" folder the development of all DL architectures and results are presented.

In the jupyter notebook "notebooks.additional.results_with_calibrated_thresholds.ipynb" evaluation results of the models are presented.
