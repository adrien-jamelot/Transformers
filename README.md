My short term goal is to reproduce a practical I did 4-5 years ago where we used an LSTM trained on Shakespeare's texts to generate more Shakespeare-like tests. This practicla was very probably inspired from Karpathy's famous blogpost on [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). Since then, the Deep Learning world has moved to Transformers and I want to understand more in depth what makes them different. To spice up the challenge, I'm using as little external resources as possible and try to keep myself to the research paper itself and pytorch API's doc.

At this stage, I have implemented most blocks of the neural network and branched them together. The natural next steps are to: 
1. format the Shakespeare's text into pytorch-ready data.
2. add a positional encoding block to my transformer model
3. figure out how the model is supposed to be initialized since it's auto-regressive (it consumes the last output to generate its next output).
