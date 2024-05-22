import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, StableDiffusionXLImg2ImgPipeline
import datetime
import json
import os

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_MODEL_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
LORA_WEIGHTS = 'sdxl-lego-city-model/pytorch_lora_weights.safetensors'

def get_default_pipeline():
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.float16, use_safetensors=True)

    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    pipe.to("cuda")

    return pipe

def get_lora_pipeline():
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.float16, use_safetensors=True)
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    pipe.load_lora_weights(LORA_WEIGHTS)
    pipe.to("cuda")
    
    return pipe

def get_refiner(base_model):
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_MODEL_ID,
        text_encoder_2=base_model.text_encoder_2,
        vae=base_model.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

    return refiner

def generate_image(pipe, prompt, i, current_time, prefix, output_dir, refiner = None):
    n_steps = 20
    high_noise_frac = 0.8

    if refiner is None:
        image = pipe(prompt, num_inference_steps=n_steps).images[0]
    else:
        image = pipe(prompt, num_inference_steps=n_steps, denoising_end=high_noise_frac, output_type="latent").images[0]
        image = refiner(prompt=prompt, num_inference_steps=n_steps, denoising_start=high_noise_frac, image=image).images[0]

    file_name = os.path.join(output_dir, f"{current_time}_{prefix}_img_{i}")

    info_json = {
        "prompt": prompt,
        "model_id": MODEL_ID,
    }
    with open(f"{file_name}.json", "w") as json_file:
        json.dump(info_json, json_file)
    
    image.save(f"{file_name}.png")

def generate_images(pipe, prefix, refiner):
    prompts = [
        "majestic lion in the jungle, looking at the camera.",
        "friends hanging out on a beautiful summer evening, beach bar.",
        "teacher explaining, in fun and entertaining way, physics to a group of interested kids.",
        "crowd at a concert, enjoying the music and the atmosphere.",
        "a small cactus with a happy face in the Sahara desert.",
        "professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.",
        "cute dragon creature",
        "a very beautiful and colorful bird",
        "a cute little puppy",
        "a big football stadium",
        "a house in a modern city on a sunny day",
        "pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail."
    ]

    output_dir = "./generated/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    for i, prompt in enumerate(prompts):
        generate_image(pipe, prompt, i, current_time, prefix, output_dir)
        generate_image(pipe, prompt, i, current_time, f"{prefix}_w_refiner", output_dir, refiner)

        # prompt = f"Image in the style of simpsons cartoons, {prompt}"
        prompt = f"Image in lego city adventures style, {prompt}"

        generate_image(pipe, prompt, i, current_time, f"{prefix}_w_trigger", output_dir)
        generate_image(pipe, prompt, i, current_time, f"{prefix}_w_trigger_and_refiner", output_dir, refiner)

if __name__ == "__main__":
    if True:
        pipe = get_default_pipeline()
        refiner = get_refiner(pipe)

        generate_images(pipe, "default", refiner)

        del pipe
        del refiner
        torch.cuda.empty_cache()

    pipe = get_lora_pipeline()
    refiner = get_refiner(pipe)
    generate_images(pipe, "lora", refiner)