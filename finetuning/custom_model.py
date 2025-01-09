import math

import loralib
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import BertConfig


class Adapter(nn.Module, loralib.LoRALayer):
    """
    Adapter class similar to that of the original code from Microsoft.
    """
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = False,
            **kwargs
    ):
        nn.Module.__init__(self)
        loralib.LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        # self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        self.lora_A = nn.Parameter(torch.zeros((in_features, r)))
        self.lora_B = nn.Parameter(torch.zeros((r, out_features)))

        self.scaling = self.lora_alpha / self.r
        # Freezing the pre-trained weight matrix
        # weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # if mode:
        #     self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling

        # TODO: Check this
        # if mode:
        #     if self.merge_weights and self.merged:
        #         # Make sure that the weights are not merged
        #         if self.r > 0:
        #             self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
        #         self.merged = False
        # else:
        #     if self.merge_weights and not self.merged:
        #         # Merge the weights and mark it
        #         if self.r > 0:
        #             self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
        #         self.merged = True

    def forward(self, x: torch.Tensor):
        result = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling

        return result


class MultiLinear(nn.Linear):
    """
    MultiLinear class to add multiple adapters to the BERT model. It has the original linear layer and multiple adapters.
    """
    def __init__(self, in_features, out_features, r, lora_alpha, lora_dropout=0., num_adapters=2,
                 merge_weights=False, **kwargs):
        self.num_adapters = num_adapters
        super().__init__(in_features, out_features, **kwargs)

        self.adapters = nn.ModuleList(Adapter(in_features, out_features, r, lora_alpha, lora_dropout, merge_weights) for _ in range(num_adapters))

        # self.reset_parameters()

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        pass

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

        for i in range(self.num_adapters):
            self.adapters[i].train(mode)

    def forward(self, x, masking):
        result = F.linear(x, self.weight, bias=self.bias)

        # result += self.adapters[masking](x)

        for i in range(self.num_adapters):
            result += masking[:, i].view(-1, 1, 1) * self.adapters[i](x)

        return result


class CustomBert(transformers.PreTrainedModel):
    """
    Custom BERT model with LoRA layers applied to the query, key, and value layers of the self-attention mechanism.
    """
    def __init__(self, bert, num_adapters=2, rank=8, alpha=8):
        super().__init__(config=BertConfig.from_pretrained('google-bert/bert-base-uncased'))
        self.bert = bert
        self.l1 = nn.Linear(bert.config.hidden_size, 1)
        self.do = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Add LoRA layers to the BERT model
        for i, (name, module) in enumerate(self.bert.named_modules()):
            if isinstance(module, nn.Linear) and "encoder" and "attention" in name:
                idx = int(name.split(".")[2])
                new_layer = MultiLinear(module.in_features,
                                        module.out_features,
                                        r=rank,
                                        lora_alpha=alpha,
                                        num_adapters=num_adapters
                                        )
                if "query" in name:
                    new_layer.load_state_dict(module.state_dict(), strict=False)
                    self.bert.encoder.layer[idx].attention.self.query = new_layer.to(module.weight.device)

                    assert torch.allclose(module.weight, self.bert.encoder.layer[idx].attention.self.query.weight)
                elif "key" in name:
                    new_layer.load_state_dict(module.state_dict(), strict=False)
                    self.bert.encoder.layer[idx].attention.self.key = new_layer.to(module.weight.device)

                    assert torch.allclose(module.weight, self.bert.encoder.layer[idx].attention.self.key.weight)
                elif "value" in name:
                    new_layer.load_state_dict(module.state_dict(), strict=False)
                    self.bert.encoder.layer[idx].attention.self.value = new_layer.to(module.weight.device)

                    assert torch.allclose(module.weight, self.bert.encoder.layer[idx].attention.self.value.weight)

    def forward(self, x, mask, masking):
        bert_out = self.bert(x, attention_mask=mask, masking=masking)
        o = bert_out.last_hidden_state[:, 0, :]
        o = self.do(o)
        o = self.relu(o)
        o = self.l1(o)
        o = self.sigmoid(o)
        return o


class LoRABert(transformers.PreTrainedModel):
    """
    Classic LoRA implementation applied to query, key, and value layers of the self-attention mechanism.
    """
    def __init__(self, bert, rank=8, alpha=8):
        super().__init__(config=BertConfig.from_pretrained('google-bert/bert-base-uncased'))
        self.bert = bert
        self.l1 = nn.Linear(bert.config.hidden_size, 1)
        self.do = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Add LoRA layers to the BERT model
        for i, (name, module) in enumerate(self.bert.named_modules()):
            if isinstance(module, nn.Linear) and "encoder" and "attention" in name:
                idx = int(name.split(".")[2])
                new_layer = loralib.Linear(module.in_features,
                                           module.out_features,
                                           r=rank, lora_alpha=alpha,
                                           lora_dropout=0.)
                if "query" in name:
                    new_layer.load_state_dict(module.state_dict(), strict=False)
                    self.bert.encoder.layer[idx].attention.self.query = new_layer.to(module.weight.device)
                elif "key" in name:
                    new_layer.load_state_dict(module.state_dict(), strict=False)
                    self.bert.encoder.layer[idx].attention.self.key = new_layer.to(module.weight.device)
                elif "value" in name:
                    new_layer.load_state_dict(module.state_dict(), strict=False)
                    self.bert.encoder.layer[idx].attention.self.value = new_layer.to(module.weight.device)

    def reset_adapter(self, rank, alpha):
        for i, (name, module) in enumerate(self.bert.named_modules()):
            if isinstance(module, MultiLinear):
                idx = int(name.split(".")[2])
                new_layer = loralib.Linear(module.in_features,
                                           module.out_features,
                                           r=rank, lora_alpha=alpha,
                                           lora_dropout=0.)
                if "query" in name:
                    new_layer.weight = module.weight
                    new_layer.bias = module.bias

                    self.bert.encoder.layer[idx].attention.self.query = new_layer.to(module.weight.device)
                elif "key" in name:
                    new_layer.weight = module.weight
                    new_layer.bias = module.bias

                    self.bert.encoder.layer[idx].attention.self.key = new_layer.to(module.weight.device)
                elif "value" in name:
                    new_layer.weight = module.weight
                    new_layer.bias = module.bias

                    self.bert.encoder.layer[idx].attention.self.value = new_layer.to(module.weight.device)

    def forward(self, x, mask):
        bert_out = self.bert(x, attention_mask=mask)
        o = bert_out.last_hidden_state[:, 0, :]
        o = self.do(o)
        o = self.relu(o)
        o = self.l1(o)
        o = self.sigmoid(o)
        return o