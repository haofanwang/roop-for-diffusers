import os
import cv2
import copy
import argparse
import logging
import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for caching models
face_analyser = None
face_swapper = None


def getFaceSwapModel(model_path: str):
    """Load the face swap model."""
    model = insightface.model_zoo.get_model(model_path)
    return model


def getFaceAnalyser(model_path: str, providers, det_size=(320, 320)):
    """Load the face analyser model."""
    global face_analyser
    if face_analyser is None:
        face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
        face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser, frame: np.ndarray):
    """Get a single face from the frame."""
    faces = face_analyser.get(frame)
    try:
        return min(faces, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyser, frame: np.ndarray):
    """Get all faces in the frame, sorted from left to right."""
    try:
        faces = face_analyser.get(frame)
        return sorted(faces, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    """Swap the face from source to target image."""
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


def parse_indexes(index_str, num_faces):
    """Parse face indexes from a string with optional ranges (e.g., '0,1,2-4')."""
    indexes = []
    for part in index_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            indexes.extend(range(start, end + 1))
        else:
            indexes.append(int(part))
    return [i for i in indexes if i < num_faces]


def process(source_img: Union[Image.Image, List], target_img: Image.Image, source_indexes: str, target_indexes: str, model: str):
    """Main function to perform face swapping."""
    # Load providers and models
    providers = onnxruntime.get_available_providers()
    face_analyser = getFaceAnalyser(model, providers)

    # Load face swapper model
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = getFaceSwapModel(model_path)

    # Convert target image to numpy array
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_faces = get_many_faces(face_analyser, target_img)

    # Check if faces are found in the target image
    if not target_faces:
        logging.error("No target faces found!")
        return None

    temp_frame = copy.deepcopy(target_img)

    # Process the source image(s)
    if isinstance(source_img, list):
        num_source_images = len(source_img)
        num_target_faces = len(target_faces)

        if num_source_images != num_target_faces:
            logging.warning("The number of source images does not match the number of target faces. Proceeding with face-by-face swap.")

        for i in range(num_target_faces):
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
            if source_faces is None:
                raise ValueError(f"No faces found in source image {i}.")

            temp_frame = swap_face(face_swapper, source_faces, target_faces, i, i, temp_frame)
    else:
        num_source_faces = len(get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)))
        target_indexes = parse_indexes(target_indexes, len(target_faces))
        source_indexes = parse_indexes(source_indexes, num_source_faces)

        for source_index, target_index in zip(source_indexes, target_indexes):
            temp_frame = swap_face(face_swapper, get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)),
                                   target_faces, source_index, target_index, temp_frame)

    # Convert the result back to an image
    result_image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
    return result_image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face swap with optional restoration and enhancement.")
    parser.add_argument("--source_img", type=str, required=True, help="Path(s) to source image(s), separated by semicolons.")
    parser.add_argument("--target_img", type=str, required=True, help="Path to the target image.")
    parser.add_argument("--output_img", type=str, default="result.png", help="Path to save the output image.")
    parser.add_argument("--source_indexes", type=str, default="-1", help="Comma-separated list of source face indexes to use.")
    parser.add_argument("--target_indexes", type=str, default="-1", help="Comma-separated list of target face indexes to swap.")
    parser.add_argument("--face_restore", action="store_true", help="Enable face restoration after swapping.")
    parser.add_argument("--background_enhance", action="store_true", help="Enable background enhancement.")
    parser.add_argument("--face_upsample", action="store_true", help="Enable face upsampling.")
    parser.add_argument("--upscale", type=int, default=1, help="Upscale factor, up to 4.")
    parser.add_argument("--codeformer_fidelity", type=float, default=0.5, help="CodeFormer fidelity parameter.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Split the source image paths
    source_img_paths = args.source_img.split(';')
    logging.info(f"Source image paths: {source_img_paths}")
    target_img_path = args.target_img

    # Open images
    source_img = [Image.open(img_path) for img_path in source_img_paths]
    target_img = Image.open(target_img_path)

    # Load the pre-trained face swap model
    model = "./checkpoints/inswapper_128.onnx"
    result_image = process(source_img, target_img, args.source_indexes, args.target_indexes, model)

    if result_image is None:
        logging.error("Face swap failed.")
        return

    # Optional face restoration
    if args.face_restore:
        from restoration import *
        check_ckpts()  # Ensure the restoration models are available
        upsampler = set_realesrgan()
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9).to(device)
        checkpoint = torch.load("CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth")["params_ema"]
        codeformer_net.load_state_dict(checkpoint)
        codeformer_net.eval()

        result_image = np.array(result_image)
        result_image = face_restoration(result_image, args.background_enhance, args.face_upsample, args.upscale, args.codeformer_fidelity,
                                       upsampler, codeformer_net, device)
        result_image = Image.fromarray(result_image)

    # Save the result
    result_image.save(args.output_img)
    logging.info(f"Result saved to: {args.output_img}")


if __name__ == "__main__":
    main()
