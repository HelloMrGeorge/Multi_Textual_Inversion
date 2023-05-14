import os
import lpips
import torch
from tqdm import tqdm, trange
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer


def update_from_embedding(model_path, placeholder_token, embedding, save_path):
    '''
    model_path = "/root/sd-v1-5"
    placeholder_token = "<liukanshan>"
    embedding = torch.load('embeds/liukanshan.bin')
    save_path = "/root/project/encoder_402"
    '''

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    tokenizer.add_tokens([placeholder_token])
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data

    text_encoder.get_input_embeddings().weight[placeholder_token_id]

    if embedding[placeholder_token].shape == text_encoder.get_input_embeddings().weight[placeholder_token_id].shape:
        token_embeds[placeholder_token_id] = embedding[placeholder_token]
        text_encoder.save_pretrained(f'{save_path}/text_encoder')
        tokenizer.save_pretrained(f'{save_path}/tokenizer')
        print(f'done in {save_path}')

def load_MTI_embeds(pipe, model_id, token, number=1, weight_dir="/root/annotated_textual_inversion/output", name="learned_embeds"):
    
    orig_token = token.replace("<", "").replace(">", "")
    new_tokens = []
    for i in range(number):
        weight_name = os.path.join(weight_dir, f"{name}-token{i}.bin")
        pipe.load_textual_inversion(model_id, f"<{orig_token}-{i}>", weight_name=weight_name)
        new_tokens.append(f"<{orig_token}-{i}>")
    return " ".join(new_tokens)

def generate_sample(pipe, prompt, output_dir, num_samples=10, step=50, guidance_scale=8, generator=None):

    pipe.set_progress_bar_config(disable=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"generating samples ...")
    for i in trange(num_samples):
        sample = pipe(prompt, num_inference_steps=step, guidance_scale=guidance_scale, num_images_per_prompt=1, generator=generator).images[0]
        sample.save(f"{output_dir}/sample-{i}.png")

    pipe.set_progress_bar_config(disable=False)

def lpips_validate_pair(loss_fn, sample, target):

    sample = lpips.im2tensor(lpips.load_image(sample)).cuda()
    target = lpips.im2tensor(lpips.load_image(target)).cuda()
    return loss_fn(sample, target).item()

def lpips_validate_dir(loss_fn, sample_dir, target_dir):

    sample_paths = sorted(os.listdir(sample_dir))
    target_paths = sorted(os.listdir(target_dir))

    losses = {}
    for idx, target_path in enumerate(target_paths):
        print(f"validating {idx} - {target_path} ...")
        losses[target_path] = []
        for sample_path in tqdm(sample_paths):
            losses[target_path].append(lpips_validate_pair(loss_fn, os.path.join(sample_dir, sample_path), os.path.join(target_dir, target_path)))

    return losses

def clip_validate_image_pair(model, processor, sample, target, device="cuda"):
    
    sample = processor(images=Image.open(sample), return_tensors="pt")
    sample_features = model.get_image_features(sample.pixel_values.to(device))

    target = processor(images=Image.open(target), return_tensors="pt")
    target_features = model.get_image_features(target.pixel_values.to(device))

    # normalized features
    sample_features = sample_features / sample_features.norm(p=2, dim=-1, keepdim=True)
    target_features = target_features / target_features.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity as logits
    logits = torch.matmul(sample_features, target_features.t())
    return logits.item()

def clip_validate_image_dir(model, processor, sample_dir, target_dir, device="cuda"):

    sample_paths = sorted(os.listdir(sample_dir))
    target_paths = sorted(os.listdir(target_dir))

    losses = {}
    for idx, target_path in enumerate(target_paths):
        print(f"validating {idx} - {target_path} ...")
        losses[target_path] = []
        for sample_path in tqdm(sample_paths):
            losses[target_path].append(clip_validate_image_pair(model, processor, os.path.join(sample_dir, sample_path), os.path.join(target_dir, target_path), device=device))

    return losses

def clip_validate_text_image_pair(model, processor, text, image, device="cuda"):
    
    text = processor(text=text, return_tensors="pt")
    text_features = model.get_text_features(text.input_ids.to(device), text.attention_mask.to(device))

    image = processor(images=Image.open(image), return_tensors="pt")
    image_features = model.get_image_features(image.pixel_values.to(device))

    # normalized features
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity as logits
    logits = torch.matmul(text_features, image_features.t())
    return logits.item()

def clip_validate_pair(model, processor, text, image, device="cuda"):
    
    image = Image.open(image)
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    outputs = model(input_ids=inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device), pixel_values=inputs.pixel_values.to(device))
    return outputs.logits_per_image.item()  # this is the image-text similarity score

def clip_validate_dir(model, processor, text, image_dir, device="cuda"):

    image_paths = sorted(os.listdir(image_dir))

    losses = []
    for image_path in tqdm(image_paths):
        losses.append(clip_validate_pair(model, processor, text, os.path.join(image_dir, image_path), device=device))

    return losses
