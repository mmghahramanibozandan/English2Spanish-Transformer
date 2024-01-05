# English2Spanish-Transformer

This repository constitutes a segment of my Master's thesis (text-conditioned image generation) to test my customized transformer, where I developed a PyTorch-based machine translator from English to Spanish. It serves as an exemplary demonstration of constructing a compact transformer for proficiently translating English sentences into Spanish. All components, including the transformer, have been created entirely from the ground up. The model is conveniently testable and adaptable through GPU utilization in Google Colab (Standard Subscription).

## Dataset

I utilized the `spa.txt` dataset, comprising pairs of English-Spanish sentences. The dataset contains a total of `M=100,000` data samples, making it a suitable choice for achieving robust generalization in Machine Translation tasks. The data file is accessible at `data/spa.txt`. The source vocabulary (English vocab) comprises `src_vocab=10,904` tokens, while the target vocabulary (Spanish vocab) encompasses `trg_vocab=24,189` tokens (including `<st>` and `<end>` tokens for Spanish sentences). To standardize the length of each sentence, whether in English or Spanish, zero-padding (`<pad>` token) is applied to reach a length of `MAX_LEN=21`. This length corresponds to the longest sentence present in our dataset.

## Training the Transformer

In constructing the transformer, `my_transformer.py`, I configured parameters as follows: `d_model=40`, `heads=2`, and `N (number of encoder/decoder layers)=1` (the remaining parameters are configured in accordance with the specifications outlined in the original paper: 'Attention Is All You Need'). The optimization of the transformer occurs in `my_train.py` with epochs set to `epochs=450`. The training process unfolds across four distinct phases: (i) `epochs=200, lr=0.001`, (ii) `epochs=100, lr=0.0005`, (iii) `epochs=100, lr=0.0001`, and (iv) `epochs=50, lr=0.00005`. The training function prints the translation of a sample English test sentence, **test_sen = "we spent the night in a cheap hotel"** every `print_step = 10` steps to observe progress as the loss decreases. To review these steps, refer to the `English2Spanish.ipynb` notebook where all the mentioned configurations are implemented. The total training time amounts to **3 hours and 30 minutes**. The transformer requires training on a total of **2,747,405** parameters. To enhance the model's performance, increasing `d_model` is an option. (will increase computational expenses and elevate the likelihood of overfitting. So, be careful!)

## Saving the Params

Once the model is trained, `SRC_Tokens, TRG_Tokens` are saved as a pickle file. This will eliminate the need to re-read the entire dataset during model testing. The pickle file can be found at `saved_params/params.pkl`.

## Translation

The model's weights file is available at `trained_model/En2Sp.pt`. To evaluate the model, one can clone the repository and execute the command: **python3 translate.py \<sentence> \<path_to_model> \<path_to_params>**, where `<sentence>` is an English language, `<path_to_model>` is the path to model's weights file (trained_model/En2Sp.pt) and `<path_to_params>` is the path to the pickle file (saved_params/params.pkl).

## Examples

(1) hello! -> Hola                                \<CORRECT>

(2) what is my name? -> ¿Cómo es mi nombre        \<CORRECT>

(3) how are you? -> ¿Cómo haces                   \<WRONG>

(4) she is pretty. -> Está muy hermosa            \<CORRECT>

(5) we spent the night in a cheap hotel -> Pasamos el hotel barato de noche              \<CORRECT>
