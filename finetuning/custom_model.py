import math

import loralib
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import BertConfig


class Adapter(nn.Module, loralib.LoRALayer):
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
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Module.__init__(self)
        loralib.LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                   merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.weight = nn.Parameter()
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

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
        if self.r > 0 and not self.merged:
            result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return None


class MultiLinear(nn.Linear):
    def __init__(self, linear, in_features, out_features, r, lora_alpha, lora_dropout, num_adapters=2,
                 merge_weights=True, **kwargs):
        self.num_adapters = num_adapters
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.linear = linear
        self.adapters = nn.ModuleList(Adapter(in_features, out_features, r, lora_alpha, lora_dropout, merge_weights) for _ in
                                      range(num_adapters))
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def forward(self, x, masking):
        result = self.linear(x)

        for i in range(self.num_adapters):
            if self.adapters[i].r > 0:
                result += self.adapters[i](x) * masking[:, i].view(-1, 1, 1)

        return result


class CustomBert(transformers.PreTrainedModel):
    def __init__(self, bert, num_adapters=2):
        super(CustomBert, self).__init__(config=BertConfig.from_pretrained('google/bert_uncased_L-2_H-128_A-2'))
        self.bert = bert
        self.l1 = nn.Linear(128, 1)
        self.do = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Add LoRA layers to the BERT model
        for i, (name, module) in enumerate(self.bert.named_modules()):
            if isinstance(module, nn.Linear) and "encoder" and "attention" in name:
                idx = int(name.split(".")[2])
                if "query" in name:
                    self.bert.encoder.layer[idx].attention.self.query = MultiLinear(module,
                                                                                    module.in_features,
                                                                                    module.out_features, r=8,
                                                                                    lora_alpha=32, lora_dropout=0.1,
                                                                                    num_adapters=num_adapters
                                                                                    )

                    assert torch.allclose(module.weight,
                                          self.bert.encoder.layer[idx].attention.self.query.linear.weight)

                elif "key" in name:
                    self.bert.encoder.layer[idx].attention.self.key = MultiLinear(module,
                                                                                  module.in_features,
                                                                                  module.out_features, r=8,
                                                                                  lora_alpha=32, lora_dropout=0.1,
                                                                                  num_adapters=num_adapters
                                                                                  )

                    assert torch.allclose(module.weight, self.bert.encoder.layer[idx].attention.self.key.linear.weight)
                elif "value" in name:
                    self.bert.encoder.layer[idx].attention.self.value = MultiLinear(module,
                                                                                    module.in_features,
                                                                                    module.out_features, r=8,
                                                                                    lora_alpha=32, lora_dropout=0.1,
                                                                                    num_adapters=num_adapters
                                                                                    )

                    assert torch.allclose(module.weight,
                                          self.bert.encoder.layer[idx].attention.self.value.linear.weight)

    def forward(self, x, mask, masking):
        bert_out = self.bert(x, attention_mask=mask, masking=masking)
        o = bert_out.last_hidden_state[:, 0, :]
        o = self.do(o)
        o = self.relu(o)
        o = self.l1(o)
        o = self.sigmoid(o)
        return o


class LoRABert(transformers.PreTrainedModel):
    def __init__(self, bert, num_adapters=2):
        super(LoRABert, self).__init__(config=BertConfig.from_pretrained('google/bert_uncased_L-2_H-128_A-2'))
        self.bert = bert
        self.l1 = nn.Linear(128, 1)
        self.do = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Add LoRA layers to the BERT model
        for i, (name, module) in enumerate(self.bert.named_modules()):
            if isinstance(module, nn.Linear) and "encoder" and "attention" in name:
                idx = int(name.split(".")[2])
                if "query" in name:
                    self.bert.encoder.layer[idx].attention.self.query = loralib.Linear(module.in_features,
                                                                                        module.out_features,
                                                                                        r=8, lora_alpha=32,
                                                                                        lora_dropout=0.1)


                elif "key" in name:
                    self.bert.encoder.layer[idx].attention.self.key = loralib.Linear(module.in_features,
                                                                                        module.out_features,
                                                                                        r=8, lora_alpha=32,
                                                                                        lora_dropout=0.1)

                elif "value" in name:
                    self.bert.encoder.layer[idx].attention.self.value = loralib.Linear(module.in_features,
                                                                                        module.out_features,
                                                                                        r=8, lora_alpha=32,
                                                                                        lora_dropout=0.1)


    def forward(self, x, mask):
        bert_out = self.bert(x, attention_mask=mask)
        o = bert_out.last_hidden_state[:, 0, :]
        o = self.do(o)
        o = self.relu(o)
        o = self.l1(o)
        o = self.sigmoid(o)
        return o