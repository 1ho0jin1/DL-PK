import torch
from torchsummary import summary
from gru import GRUPK
from lstm import LSTMPK
from transformer import TransformerPK



if __name__ == "__main__":
    x = torch.randn(8, 2, 3)  # (batch_size, seq_len, input_dim)
    model1 = LSTMPK()
    model2 = GRUPK()
    model3 = TransformerPK()
    
    summary(model1, x)
    summary(model2, x)
    summary(model3, x)
    
    print()