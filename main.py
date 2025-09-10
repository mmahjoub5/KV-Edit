import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from PIL import ExifTags, Image
import torch
import gradio as gr
import numpy as np
from flux.sampling import prepare
from flux.util import (configs, load_ae, load_clip, load_t5)
from models.kv_edit import Flux_kv_edit

from typing import Optional, Dict, Any
import numpy as np
from PIL import Image

def make_canvas(
    image: Image.Image,
    mask: Optional[Image.Image] = None,
    as_numpy: bool = True
) -> Dict[str, Any]:
    """
    Build a Gradio ImageEditor-like payload:
      - background: RGBA image (np array or PIL)
      - layers: list with a single RGBA mask layer (alpha channel = mask), or []
      - composite: same size RGBA (usually same as background)
    mask: white(=edit)/black(=preserve). Only alpha channel is read by your code.
    """
    bg = image.convert("RGBA")
    comp = bg.copy()

    if as_numpy:
        bg_arr = np.array(bg)
        comp_arr = np.array(comp)
        layers = []
        if mask is not None:
            m = mask.convert("L")
            # Build an RGBA layer: red for visibility (RGB), alpha = mask
            layer_rgba = np.zeros((*bg_arr.shape[:2], 4), dtype=np.uint8)
            layer_rgba[..., 0] = 255  # red channel (just cosmetic)
            layer_rgba[..., 3] = np.array(m)  # alpha carries mask
            layers = [layer_rgba]
        return {"background": bg_arr, "layers": layers, "composite": comp_arr}
    else:
        layers = []
        if mask is not None:
            m = mask.convert("L")
            layer = Image.new("RGBA", bg.size, (255, 0, 0, 0))
            layer.putalpha(m)
            layers = [layer]
        return {"background": bg, "layers": layers, "composite": comp}



@dataclass
class SamplingOptions:
    source_prompt: str = ''
    target_prompt: str = ''
    width: int = 1366
    height: int = 768
    inversion_num_steps: int = 0
    denoise_num_steps: int = 0
    skip_step: int = 0
    inversion_guidance: float = 1.0
    denoise_guidance: float = 1.0
    seed: int = 42
    re_init: bool = False
    attn_mask: bool = False
    attn_scale: float = 1.0



class FluxEditor_kv_demo:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.offload = args.offload

        self.name = args.name
        self.is_schnell = args.name == "flux-schnell"

        self.output_dir = 'regress_result'

        self.t5 = load_t5(self.device, max_length=256 if self.name == "flux-schnell" else 512)
        self.clip = load_clip(self.device)
        self.model = Flux_kv_edit(device="cpu" if self.offload else self.device, name=self.name)
        self.ae = load_ae(self.name, device="cpu" if self.offload else self.device)

        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()
        self.info = {}
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.encoder.to(self.device)
        
    @torch.inference_mode()
    def inverse(self, brush_canvas,
             source_prompt, target_prompt, 
             inversion_num_steps, denoise_num_steps, 
             skip_step, 
             inversion_guidance, denoise_guidance,seed,
             re_init, attn_mask
             ):
        self.z0 = None
        self.zt = None
        # self.info = {}
        # gc.collect()
        if 'feature' in self.info:
            key_list = list(self.info['feature'].keys())
            for key in key_list:
                del self.info['feature'][key]
        self.info = {}
        
        rgba_init_image = brush_canvas["background"]
        init_image = rgba_init_image[:,:,:3]
        shape = init_image.shape        
        height = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        width = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image = init_image[:height, :width, :]
        rgba_init_image = rgba_init_image[:height, :width, :]

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps,
            skip_step=0,# no skip step in inverse leads chance to adjest skip_step in edit
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask,
            attn_scale=1.0
        )
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        torch.cuda.empty_cache()
        
        if opts.attn_mask:
            rgba_mask = brush_canvas["layers"][0][:height, :width, :]
            mask = rgba_mask[:,:,3]/255
            mask = mask.astype(int)
        
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device)
        else:
            mask = None
        
        self.init_image = self.encode(init_image, self.device).to(self.device)

        t0 = time.perf_counter()

        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        with torch.no_grad():
            inp = prepare(self.t5, self.clip,self.init_image, prompt=opts.source_prompt)
        
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)
        self.z0,self.zt,self.info = self.model.inverse(inp,mask,opts)
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            
        t1 = time.perf_counter()
        print(f"inversion Done in {t1 - t0:.1f}s.")
        return None

        
        
    @torch.inference_mode()
    def edit(self, brush_canvas,
             source_prompt, target_prompt, 
             inversion_num_steps, denoise_num_steps, 
             skip_step, 
             inversion_guidance, denoise_guidance,seed,
             re_init, attn_mask,attn_scale
             ):
        
        torch.cuda.empty_cache()
        
        rgba_init_image = brush_canvas["background"]
        init_image = rgba_init_image[:,:,:3]
        shape = init_image.shape        
        height = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        width = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image = init_image[:height, :width, :]
        rgba_init_image = rgba_init_image[:height, :width, :]

        rgba_mask = brush_canvas["layers"][0][:height, :width, :]
        mask = rgba_mask[:,:,3]/255
        mask = mask.astype(int)
        
        rgba_mask[:,:,3] = rgba_mask[:,:,3]//2
        masked_image = Image.alpha_composite(Image.fromarray(rgba_init_image, 'RGBA'), Image.fromarray(rgba_mask, 'RGBA'))
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device)
        
        seed = int(seed)
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps,
            skip_step=skip_step,
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask,
            attn_scale=attn_scale
        )
        if self.offload:
            
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)

        t0 = time.perf_counter()

        with torch.no_grad():
            inp_target = prepare(self.t5, self.clip, self.init_image, prompt=opts.target_prompt)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)
            
        x = self.model.denoise(self.z0,self.zt,inp_target,mask,opts,self.info)
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)
            
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x.to(self.device))
        
        x = x.clamp(-1, 1)
        x = x.float().cpu()
        x = rearrange(x[0], "c h w -> h w c")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        output_name = os.path.join(self.output_dir, "img_{idx}.jpg")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            idx = 0
        else:
            fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
            if len(fns) > 0:
                idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
            else:
                idx = 0
        
        fn = output_name.format(idx=idx)
    
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
    
        exif_data[ExifTags.Base.ImageDescription] = target_prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)
        masked_image.save(fn.replace(".jpg", "_mask.png"),  format='PNG')
        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
        
        print("End Edit")
        return img

    
    @torch.inference_mode()
    def encode(self,init_image, torch_device):
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0) 
        init_image = init_image.to(torch_device)
        self.ae.encoder.to(torch_device)
        
        init_image = self.ae.encode(init_image).to(torch.bfloat16)
        return init_image
    
from PIL import Image

class FluxEditorRunner:
    def __init__(self, args):
        self.editor = FluxEditor_kv_demo(args)
        self.is_schnell = args.name == "flux-schnell"

    def _full_white_mask(self, img: Image.Image) -> Image.Image:
        return Image.new("L", img.size, color=255)

    def inverse(
        self,
        brush_canvas: Image.Image,
        source_prompt: str,
        target_prompt: str,
        inversion_num_steps: int = 28,
        denoise_num_steps: int = 28,
        skip_step: int = 4,
        inversion_guidance: float = 1.5,
        denoise_guidance: float = 5.5,
        seed: int = 0,
        re_init: bool = False,
        attn_mask: bool = False,
        mask: Optional[Image.Image] = None,
    ):
        # Build canvas; only needed if your inverse() uses brush_canvas["background"]
        # If you set attn_mask=True, supply a mask; otherwise leave it None.
        canvas = make_canvas(brush_canvas, mask if attn_mask else None, as_numpy=True)
        return self.editor.inverse(
            canvas,
            source_prompt,
            target_prompt,
            inversion_num_steps,
            denoise_num_steps,
            skip_step,
            inversion_guidance,
            denoise_guidance,
            seed,
            re_init,
            attn_mask,
        )

    def edit(
        self,
        brush_canvas: Image.Image,
        source_prompt: str,
        target_prompt: str,
        inversion_num_steps: int = 28,
        denoise_num_steps: int = 28,
        skip_step: int = 4,
        inversion_guidance: float = 1.5,
        denoise_guidance: float = 5.5,
        seed: int = 0,
        re_init: bool = False,
        attn_mask: bool = False,
        attn_scale: float = 1.0,
        mask: Optional[Image.Image] = None,
        full_image_if_no_mask: bool = True,
    ):
        # For localized edits, pass a real mask; for global edits,
        # let it default to a full-white mask.
        if mask is None and full_image_if_no_mask:
            mask = self._full_white_mask(brush_canvas)

        canvas = make_canvas(brush_canvas, mask if attn_mask or mask is not None else None, as_numpy=True)
        return self.editor.edit(
            canvas,
            source_prompt,
            target_prompt,
            inversion_num_steps,
            denoise_num_steps,
            skip_step,
            inversion_guidance,
            denoise_guidance,
            seed,
            re_init,
            attn_mask,
            attn_scale,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--port", type=int, default=41032)
    args = parser.parse_args()

    runner = FluxEditorRunner(args)
    # load your input image
    input_img = Image.open("frankfurt_000000_003025_leftImg8bit_foggy_beta_0.01.png")
    # run inversion
    inv_img = runner.inverse(
        brush_canvas=input_img,
        source_prompt="a photo of a cat",
        target_prompt="a photo of a tiger",
        inversion_num_steps=1,
        denoise_num_steps=1,
        
    )
    
    
    # run edit
    edited_img = runner.edit(
        brush_canvas=input_img,
        source_prompt="a cat sitting on a chair",
        target_prompt="a tiger sitting on a chair",
        denoise_guidance=7.0,
        attn_scale=1.5,
        denoise_num_steps=1
    )
    
    edited_img.save("edited.png")
   