# Copy Task for Recurrent Neural Networks

[Here is the reference paper.](https://arxiv.org/pdf/1602.06662)

## Task Description

Look at section 3.1 of the paper.

## Training & Validation

1. Use the **same** hyper-parameters: hidden size, batch size, learning rate, etc.
2. Minimize **cross entropy loss** only on the predicted sequence
3. Plot training vs validation curves showing **loss & accuracy vs epochs**
4. Try all of the above for different sequence lengths: ${100,200,500,1000}$

## Testing

1. Mean performance and standard error over 3 trials for each sequence length
