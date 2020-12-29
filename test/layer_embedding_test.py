# encoding:utf-8

import torch
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../"))

from models.layers.embedding_layer import EmbeddingLayer

if __name__ == "__main__":
    device = torch.device("cuda")
    embedding_layer = EmbeddingLayer(718, 10240, is_frozen=False).to(device)
    input_tensor = torch.randint(0, 718, [128, 56]).to(device)
    output_tensor = embedding_layer(input_tensor)
    print(output_tensor.shape)
