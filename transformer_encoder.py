
# d_model = This parameter refers to the dimensionality or the number of units in the model's hidden layers.
# In the transformer architecture, the input and output embeddings have a size of d_model.
# It's a hyperparameter that you can set based on the requirements of your specific task.
# Common values for d_model range from 128 to 1024.

# num_heads = The transformer model uses multi-head attention to capture different aspects of the input data in parallel.
# num_heads represents the number of attention heads.
# Each head attends to different parts of the input sequence independently, allowing the model to focus on different relationships and dependencies.
# The total computation cost is proportional to the number of heads.
# Common choices for num_heads are between 8 and 16.

# drop_prob = This parameter indicates the dropout probability, which is used as a regularization technique to prevent overfitting during training.
# Dropout randomly sets a fraction of the input units to zero during each forward and backward pass.
# The drop_prob value typically ranges between 0.1 and 0.5.

# batch_size = The batch size refers to the number of training examples fed into the model at each training step.
# # A larger batch size can potentially speed up training by taking advantage of parallel processing, but it also requires more memory.
# # The appropriate value for batch_size depends on the available resources (e.g., GPU memory) and the size of your dataset.

# max_sequence_length = This parameter defines the maximum length of input sequences allowed in the model.
# Transformers have a fixed input size, and sequences longer than max_sequence_length need to be truncated or split.
# It is important to choose a suitable value for max_sequence_length based on the length distribution of your data.
# Longer sequences increase computational complexity and memory requirements.

# ffn_hidden  = ffn_hidden is the number of hidden units in the feed-forward neural network (FFN) component of the transformer model.
# The FFN is applied after the self-attention mechanism in each transformer block.
# It consists of two linear layers separated by a non-linear activation function (commonly ReLU).
# The value of ffn_hidden is typically set based on the d_model dimension;
# it should be large enough to allow the model to capture complex patterns in the data.


# num_layers = num_layers denotes the number of encoder transformer layers stacked on top of each other.
# Each layer processes the input data sequentially, and the output of one layer serves as the input to the next.
# More layers can potentially capture finer details in the data, but they also increase the model's computational complexity.
# The choice of num_layers is a trade-off between model performance and computational resources
import nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()

class Encoder (nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
