# python script for surface normal estimation from video samples
import argparse
import os
import logging
import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image

from models.geowizard_pipeline import DepthNormalEstimationPipeline
from utils.seed_all import seed_all
from diffusers import DiffusionPipeline, DDIMScheduler, AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# Set up logging
logging.basicConfig(level=logging.INFO)

def process_video(video_path, output_dir, pipe, args):
    """Process a video to estimate surface normals for each frame."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    output_dir_normal_color = os.path.join(output_dir, "normal_colored")
    os.makedirs(output_dir_normal_color, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer for saving the output video
    output_video_path = os.path.join(output_dir, "output_normal_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame
    for frame_idx in tqdm(range(total_frames), desc="Processing video frames"):
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(frame_rgb)

        # Estimate surface normals
        with torch.no_grad():
            pipe_out = pipe(
                input_image,
                denoising_steps=args.denoise_steps,
                ensemble_size=args.ensemble_size,
                processing_res=args.processing_res,
                match_input_res=not args.output_processing_res,
                domain=args.domain,
                color_map=args.color_map,
                show_progress_bar=False,
            )
            normal_colored: Image.Image = pipe_out.normal_colored

        # Save the normal map as an image
        normal_colored_save_path = os.path.join(output_dir_normal_color, f"frame_{frame_idx:04d}_normal_colored.png")
        normal_colored.save(normal_colored_save_path)

        # Convert the normal map to a format suitable for video writing
        normal_colored_cv2 = cv2.cvtColor(np.array(normal_colored), cv2.COLOR_RGB2BGR)
        out.write(normal_colored_cv2)

    # Release video capture and writer
    cap.release()
    out.release()
    logging.info(f"Video processing complete. Output saved to {output_video_path}")

def main():
    """Main function to handle video processing."""
    parser = argparse.ArgumentParser(
        description="Run MonoDepthNormal Estimation on a video using Stable Diffusion."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default='lemonaddie/geowizard',
        help="Pretrained model path from Hugging Face or local directory.",
    )
    parser.add_argument(
        "--input_video", type=str, required=True, help="Path to the input video."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results."
    )
    parser.add_argument(
        "--domain",
        type=str,
        default='indoor',
        required=True,
        help="Domain for prediction (e.g., indoor, outdoor).",
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=10,
        help="Diffusion denoising steps. More steps improve accuracy but slow down inference.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to ensemble. More predictions improve results but slow down inference.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float). May lead to suboptimal results.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution for processing. Set to 0 to use input resolution.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="Output results at the processing resolution instead of input resolution.",
    )
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap for rendering depth predictions.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    # Set random seed
    if args.seed is None:
        import time
        args.seed = int(time.time())
    seed_all(args.seed)

    # Load the model pipeline
    if args.half_precision:
        dtype = torch.float16
    else:
        dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder='vae')
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder='scheduler')
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_path, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_path, subfolder="feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

    pipe = DepthNormalEstimationPipeline(
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        unet=unet,
        scheduler=scheduler,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Process the video
    process_video(args.input_video, args.output_dir, pipe, args)

if __name__ == "__main__":
    main()
