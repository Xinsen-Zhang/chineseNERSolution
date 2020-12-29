# encoding:utf-8

import torch
import unittest
from models.layers.bi_lstm import BiLSTM
from models.layers.embedding_layer import EmbeddingLayer


class LayerTest(unittest.TestCase):
    def test_bilstm_layer(self):
        batch_input = torch.randn([128, 49, 300])
        length = torch.ones([128]).fill_(value=49).long()
        bilstm = BiLSTM(num_layers=2, hidden_size=512, input_size=300)
        batch_output = bilstm(batch_input, length)
        shape = batch_output.shape
        self.assertEqual(f'{shape[0]}, {shape[1]}, {shape[2]}', '128, 49, 9')

    def test_embedding_layer(self):
        device = torch.device("cuda")
        embedding_layer = EmbeddingLayer(
            718, 10240, is_frozen=False).to(device)
        input_tensor = torch.randint(0, 718, [128, 56]).to(device)
        output_tensor = embedding_layer(input_tensor)
        shape = output_tensor.shape
        self.assertEqual(f'{shape[0]}, {shape[1]}, {shape[2]}', '128, 56, 718')


if __name__ == "__main__":
    unittest.main()
