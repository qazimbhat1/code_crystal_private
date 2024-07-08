from transformers import AutoModel, AutoTokenizer
# for name, layer in self.model.named_modules():
    # print(name)
# # Print module names and their corresponding values for modules containing 'rotary_emb.inv_freq'
# for name, module in self.model.named_modules():
#     # print(name)
#     if 'rotary_emb' in name:
#         # print("JJ")
#         for param_name, param in module.named_modules():
#             print(type(param))
#             print(param.inv_freq)
#             print(f"Module: {name}, Parameter: {param_name}")#, Value: {param.data}")
#             break
# print("------------")
# # Replace 'your_model_name' with the name of the Hugging Face model
hf_model_name = "codellama/CodeLlama-7b-hf"


# Load the Hugging Face model and tokenizer
# hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
hf_model = AutoModel.from_pretrained(hf_model_name)

# Print layer names and their corresponding weights
# for name, param in hf_model.named_modules():
#     print(f"Layer: {name}")


for name, module in hf_model.named_modules():
    # print(name)
    if 'rotary_emb' in name:
        # print("JJ")
        for param_name, param in module.named_modules():
            print(type(param))
            print(param.inv_freq)
            print(param.dim)
            print(param.base)
            print(f"Module CKPT: {name}, Parameter: {param_name}")#, Value: {param.data}")
            break