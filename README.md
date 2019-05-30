# Fast TDNN layer implementation 

This is an alternative implementation of the TDNN layer, proposed by Waibel _et al._ [1].
The main difference compared to other implementations is that it exploits the 
[Pytorch Conv1d](https://pytorch.org/docs/stable/nn.html?highlight=conv1d#torch.nn.Conv1d) dilatation argument, 
making it multitudes faster than other popular implementations such as 
[SiddGururani's PyTorch-TDNN](https://github.com/SiddGururani/Pytorch-TDNN).  

## Usage
```python
# Create a TDNN layer 
layer_context = [-2, 0, 2]
input_n_feat = previous_layer_n_feat 
tddn_layer = TDNN(context=layer_context, input_channels=input_n_feat, output_channels=512, full_context=False)

# Run a forward pass; batch.size = [BATCH_SIZE, INPUT_CHANNELS, SEQUENCE_LENGTH]
out = tdnn_layer(batch)
```

## References
[\[1\] A. Waibel, T. Hanazawa, G. Hinton, and K. Shikano, 
“Phoneme Recognition Using Time-Delay Neural Networks,”, 1989](http://www.cs.toronto.edu/~fritz/absps/waibelTDNN.pdf)

