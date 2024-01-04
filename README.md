# English2Spanish-Transformer

This repository constitutes a segment of my Master's thesis to test my customized transformer, where I developed a PyTorch-based machine translator from English to Spanish. It serves as an exemplary demonstration of constructing a compact transformer for proficiently translating English sentences into Spanish. All components, including the transformer, have been created entirely from the ground up. The model is conveniently testable and adaptable through GPU utilization in Google Colab (Standard Subscription).

## Dataset

I utilized the **spa.txt** dataset, comprising pairs of English-Spanish sentences. The dataset contains a total of **M=100,000** data samples, making it a suitable choice for achieving robust generalization in Machine Translation tasks. The data file is accessible at **data/spa.txt**. The source vocabulary (English vocab) comprises **src_vocab=10,904** tokens, while the target vocabulary (Spanish vocab) encompasses **trg_vocab=24,189** tokens (including **\<st>** and **\<end>** tokens for Spanish sentences). To standardize the length of each sentence, whether in English or Spanish, zero-padding (**\<pad>** token) is applied to reach a length of **\MAX_LEN=21**. This length corresponds to the longest sentence present in our dataset.

## Training the Transformer

To build up the transformer, **my_transformer.py**, I set **d_model=40**, **heads=2**, **N(number of in encoder/decoder layers)=1**. The transformer is optimized in **my_train.py** using **epochs=450** . It consists of 4 different phases: (i) **epochs=200, lr=0.001**, (ii) **epochs=100, lr=0.0005**, (iii) **epochs=100, lr=0.0001** and (iv) **epochs=50, lr=0.00005**. Each **print_step = 10** times, the training function prints out the translation of a sample English language **test_sen = "we spent the night in a cheap hotel"** to see the progress when loss decreases. One can take a look at it by reviewing **English2Spanish.ipynb** notebook, where all the mentioned steps are implemented. (**total training time : 3 hours and 30 minutes**)

## Saving the Params

Once the model is trained, **SRC_Tokens, TRG_Tokens** are saved as a pickle file. This will allow us not to have to read the entire dataset again when testing the model. This pickle file is available at **saved_params/params.pkl**.

## Translation

The model's weights are available at **trained_model/En2Sp.pt**. To evaluate the model, one can clone the repository and execute the command: **python3 translate.py \<sentence> \<path_to_model> \<path_to_params>**, where \<sentence> is an English language, \<path_to_model> is the path to model's weights (trained_model/En2Sp.pt) and \<path_to_params> is the path to the pickle file (saved_params/params.pkl).

## Examples

(1) hello! -> Hola                                \<CORRECT>

(2) what is my name? -> ¿Cómo es mi nombre        \<CORRECT>

(3) how are you? -> ¿Cómo haces                   \<WRONG>(¿Cómo haces: how do you do?)

(4) she is pretty. -> Está muy hermosa            \<CORRECT>

(5) we spent the night in a cheap hotel -> Pasamos el hotel barato de noche              <CORRECT>
