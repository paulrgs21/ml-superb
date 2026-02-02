# Academic project based on "[ML-SUPERB: Multilingual Speech Universal PERformance Benchmark", Shi et al. (2023)](https://arxiv.org/abs/2305.10615).

- Project supervisor: M. Poli
- Code of the original article: [https://github.com/espnet/espnet/tree/master/egs2/ml_superb](https://github.com/espnet/espnet/tree/master/egs2/ml_superb)
- Abstract: Self-supervised speech representation learning (SSL) is the core component of modern speech processing systems. These models are trained on large quantities of unlabeled speech using a pretext task that enables them to learn contextualized representations. The use of such representations has led to significant improvements in downstream tasks, such as speech recognition, speaker diarization, or emotion recognition. Initially, these models were developed in English only. However, there has been a growing interest in the speech community in applying SSL to multilingual and low-resource settings. The project will consist of using a pretrained SSL model (HuBERT, wav2vec 2.0, etc.), freezing its parameters, and implementing speech recognition with a CTC loss using only 10 minutes / 1h of speech in a low-resource language, following the procedure outlined in ML-SUPERB.
  - Some ideas for extensions:
    - Comparison with other monolingual or multilingual models
    - Comparison of performance between finetuning languages
    - Other tasks (phone recognition, language identification...)
    - Other training procedure (LoRA or other PEFT methods as in ML-SUPERB 2.0, etc.)
    - Multilingual finetuning vs. monolingual finetuning.
