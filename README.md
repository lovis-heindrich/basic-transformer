# basic-transformer

This repository contains a simple implementation of a decoder-only transformer implemented from scratch in PyTorch. 

Additionally, the implementation is tested by training a small transformer model Shakespeare's works.

## Model

The transformer architecture is implemented from scratch in ```src/model.py```. The trained model consists of 4 transformer blocks with 8 attention heads, and an embedding size of 128. 

## Training

The model is trained on Shakespeare's works, which can be downloaded [here](https://www.gutenberg.org/files/100/100-0.txt). The file should be placed under ```data/shakespeare.txt```.

The training data is preprocessed using the code in ```src/load_data.py```. The following preprocessing steps are applied:
- Remove new lines, duplicate whitespaces, apostrophes, and brackets
- Tokenize the data using word boundaries
- Replace all words with less than 10 occurrences with an additional ```<unkown>``` token

The current model has 1573632 parameters and was trained for 200 epochs (~100000 steps). Training code can be found in ```train_model.ipynb``` and ```src/train.py```.

The model was trained using a 20% validation split. The final model reached a validation accuracy of 0.2155.

## Sentence generation

Here are some example sentences generated with the trained model. Prompts in the sentences are marked with brackets. Generation code can be found in ```generate_sentences.ipynb```. 

- [if thou didst] not , thou art a fool . i am a fool , and thou art a fool . thou art a fool . clown . i am
- [i] have a woman , and i have a good heart . but i will be a man of mine , and i will not be satisfied .
- [i] am a man , and i am sure of it . i have said , and i will not be sworn to thee . i am a king ,
- [you are] a merciful old wench . i am not able to wait upon my friend . i am glad to have merry a launcelet , if i had a
- [you are] a beauteous blossom , and i will encounter thee with my nan . i am a proper friend . anne . i would not play a fool ,
- [you are] in the field . enter a messenger . messenger . news , madam , news ! cleopatra . what news ? messenger . the news ? messenger . the king is gone . messenger . the lord mortimer is coming . antony . the noble antony is come
- [like the sky] , and the wind and the moon . o , what a one that is in the other ?
- [i] am a gentleman of love , and a man of all the world .

## Used resources

I created this repository while loosely following along the recommended steps of Jacob Hilton's transformers curriculum. While implementing, I mainly refered to the original transformer paper and the illustrated transformer blogpost. While I didn't look at the code samples while implementing, I've read through the given implementation in the Natural language processing with transformers book before.

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [GPT-2 Architecture graphic](https://en.wikipedia.org/wiki/GPT-2#/media/File:Full_GPT_architecture.png)
- [Jacob Hilton's Deep Learning Curriculum](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/1-Transformers.md)
- [Natural Language Processing with Transformers book](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) 