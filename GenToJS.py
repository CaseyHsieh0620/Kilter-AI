import torch
from torch import onnx
from TrainGenerator import KilterGenerator


model = KilterGenerator()
model.load_state_dict(torch.load("checkpoints/GenAI_epoch:24"))
model.eval()

example_tokens = torch.randint(0, 4846, (1, 10), dtype=torch.int64)
attention = torch.ones((1, 10), dtype=torch.float32)
casual = torch.nn.Transformer.generate_square_subsequent_mask(10)

torch.onnx.export(model,(example_tokens, attention, casual), "KilterGen.onnx", input_names = ["tokens", "attention", "casual"], output_names=["logits"], opset_version = 17, dynamic_axes={"tokens": {1: "seq_len"}, "attention": {1: "seq_len"}, "casual": {0: "seq_len", 1: "seq_len"}, "logits": {1: "seq_len"},}, external_data=False)
