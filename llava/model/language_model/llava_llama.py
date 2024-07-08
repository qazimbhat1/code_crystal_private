#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_codellama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        # print(self.model)
        # Print sub-layer names for all layers
        # from transformers import AutoModel, AutoTokenizer
        # for name, layer in self.model.named_modules():
            # print(name)
        # # Print module names and their corresponding values for modules containing 'rotary_emb.inv_freq'
        # for name, module in self.model.named_modules():
        #     # print(name)
        #     if 'rotary_emb' in name:
        #         # print("JJ")
        #         for param_name, param in module.named_modules():
        #             print(type(param))
        #             # print(param.inv_freq)
        #             print(param.dim)
        #             print(param.base)
        #             print(f"Module: {name}, Parameter: {param_name}")#, Value: {param.data}")
        #             break
        # print("------------")
        # raise
        # # # Replace 'your_model_name' with the name of the Hugging Face model
        # hf_model_name = "codellama/CodeLlama-7b-hf"
        

        # # Load the Hugging Face model and tokenizer
        # # hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        # hf_model = AutoModel.from_pretrained(hf_model_name)

        # Print layer names and their corresponding weights
        # for name, param in hf_model.named_modules():
        #     print(f"Layer: {name}")

        # No fun in testing this to above since both use same transformer versions.
        # for name, module in hf_model.named_modules():
        #     # print(name)
        #     if 'rotary_emb' in name:
        #         # print("JJ")
        #         for param_name, param in module.named_modules():
        #             # print(type(param))
        #             # print(param.inv_freq)
        #             print(param.dim)
        #             print(param.base)
        #             print(f"Module CKPT: {name}, Parameter: {param_name}")#, Value: {param.data}")
        #             break

        # raise
        # # Get the names of layers in each model
        # hf_layers = set(name for name, _ in hf_model.named_modules())
        # your_layers = set(name for name, _ in self.model.named_modules())
        # # Find layers that exist in one model but not in the other
        # missing_in_your_model = hf_layers - your_layers
        # missing_in_hf_model = your_layers - hf_layers

        # # Print the results
        # print("-----------------Layers missing in your model:")
        # print(missing_in_your_model)

        # print("\n----------------Layers missing in Hugging Face model:")
        # print(missing_in_hf_model)
        # raise
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("llava_codellama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)




#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# from typing import List, Optional, Tuple, Union

# import torch
# import torch.nn as nn

# from transformers import AutoConfig, AutoModelForCausalLM, \
#                          LlamaConfig, LlamaModel, LlamaForCausalLM

# from .crystal_coder.modeling_crystalcoder import CrystalCoderModel, CrystalCoderLMHeadModel, CrystalCoderConfig

# from transformers.modeling_outputs import CausalLMOutputWithPast

# from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


# class LlavaCrystalConfig(CrystalCoderConfig):
#     model_type = "llava_crystal"


# class LlavaCrystalModel(LlavaMetaModel, CrystalCoderModel):
#     config_class = LlavaCrystalConfig

#     def __init__(self, config: CrystalCoderConfig):
#         #TODO: check with mpt if anything needed here
#         super(LlavaCrystalModel, self).__init__(config)
#     def embed_tokens(self, x):
#         return self.wte(x)

# class LlavaCrystalForCausalLM(CrystalCoderLMHeadModel, LlavaMetaForCausalLM):
#     config_class = LlavaCrystalConfig

#     def __init__(self, config):
#         super(CrystalCoderLMHeadModel, self).__init__(config)
#         # print(config)
#         self.transformer = LlavaCrystalModel(config)


#         self.output_logits_scale = config.mup_output_alpha * config.mup_width_scale
#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None
#         # print(self.transformer)
#         # print("embedding", self.transformer.get_input_embeddings())
#         # self.pretraining_tp = config.pretraining_tp

        
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_model(self):
#         return self.transformer
    
#     def get_input_embeddings(self):
#         return self.transformer.wte

#     def set_input_embeddings(self, new_embeddings):
#         self.transformer.wte = new_embeddings

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         images: Optional[torch.FloatTensor] = None,
#         return_dict: Optional[bool] = None,
#         token_type_ids: Optional[torch.LongTensor] = None, #TODO: check these 4 below added from crystal coder class
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None
#     ) -> Union[Tuple, CausalLMOutputWithPast]:

#         if inputs_embeds is None:
#             (
#                 input_ids,
#                 position_ids,
#                 attention_mask,
#                 past_key_values,
#                 inputs_embeds,
#                 labels
#             ) = self.prepare_inputs_labels_for_multimodal(
#                 input_ids,
#                 position_ids,
#                 attention_mask,
#                 past_key_values,
#                 labels,
#                 images
#             )

#         return super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict
#         )

#     def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
#         images = kwargs.pop("images", None)
#         _inputs = super().prepare_inputs_for_generation(
#             input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
#         )
#         if images is not None:
#             _inputs['images'] = images
#         return _inputs

# AutoConfig.register("llava_crystal", LlavaCrystalConfig)
# AutoModelForCausalLM.register(LlavaCrystalConfig, LlavaCrystalForCausalLM)


# # from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
# # from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
# # from configuration_crystalcoder import CrystalCoderConfig


# # AutoConfig.register("crystalcoder", CrystalCoderConfig)
# # AutoModel.register(CrystalCoderConfig, CrystalCoderModel)
# # AutoModelForCausalLM.register(CrystalCoderConfig, CrystalCoderLMHeadModel)
# # AutoTokenizer.register(CrystalCoderConfig, fast_tokenizer_class=PreTrainedTokenizerFast)