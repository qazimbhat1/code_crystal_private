import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_images_from_directory(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(directory, filename)
            image = load_image(file_path)
            images.append(image)
    return images

def save_output_to_html(output, output_directory, input_image_filename):
    base_name, ext = os.path.splitext(os.path.basename(input_image_filename))
    output_file = os.path.join(output_directory, f"{base_name}_output.html")
    with open(output_file, "w") as f:
        f.write(output)

def main(args):
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    # # Model
    # disable_torch_init()

    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    # if 'llama-2' in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"

    # if args.conv_mode is not None and conv_mode != args.conv_mode:
    #     print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    # else:
    #     args.conv_mode = conv_mode

    # conv = conv_templates[args.conv_mode].copy()
    # if "mpt" in model_name.lower():
    #     roles = ('user', 'assistant')
    # else:
    #     roles = conv.roles

    # image = load_image(args.image_file)
    # # Similar operation in model_worker.py
    # image_tensor = process_images([image], image_processor, model.config)
    # if type(image_tensor) is list:
    #     image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    # else:
    #     image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    # 
    images = load_images_from_directory(args.image_directory)

    for image, image_filename in zip(images, sorted(os.listdir(args.image_directory))):
        print(f"--------------PROCESSING NEW IMAGE: {image_filename}-------------------")
        # Model
        disable_torch_init()

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        print(f"--------------PROCESSING NEW IMAGE: {image_filename}-------------------")


        while True:
            try:
                inp = input(f"{roles[0]}: ")
            except EOFError:
                inp = ""
            if not inp:
                print("exit...")
                break
            #inp = "Please generate html code for the given image"
            print(f"{roles[1]}: ", end="")

            # print(f"{roles[1]}: ", end="")

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            # conv.messages[-1][-1] = outputs

            # if args.debug:
            #     print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

            output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = output

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": output}, "\n")

            save_output_to_html(output, args.output_directory, image_filename)  # Save the output to HTML file
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-directory", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output-directory", type=str, required=True, help="Directory to save the output HTML file")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)

    # if __name__ == "__main__":
    # parser.add_argument("--image-directory", type=str, required=True, help="Directory containing input images")
    # parser.add_argument("--output-directory", type=str, required=True, help="Directory to save the output HTML file")
    # args = parser.parse_args()
    # main(args)
