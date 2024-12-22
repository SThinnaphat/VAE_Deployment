import os
import zipfile
import gdown
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from vae import VAE  # Ensure this points to your VAE model definition

# Check for CUDA and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Download model weights from Google Drive
def download_model_weights():
    model_url = "https://drive.google.com/uc?export=download&id=1cJUq3xdB_6gPdXnPUKVSryH9Sr1tAhhC"  # Replace YOUR_FILE_ID with your actual file ID
    model_path = "vae_finetuned2.ckpt"  # Set the file name for the model weights
    
    # Use gdown to download the model
    gdown.download(model_url, model_path, quiet=False)
    
    st.success("Model weights downloaded successfully.")

# Initialize the VAE model
def load_model():
    # Load model weights
    checkpoint_path = 'vae_finetuned2.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hyperparams = checkpoint['hyper_parameters']

    vae = VAE(
        input_res=hyperparams['input_res'],
        enc_block_str=hyperparams['enc_block_str'],
        dec_block_str=hyperparams['dec_block_str'],
        enc_channel_str=hyperparams['enc_channel_str'],
        dec_channel_str=hyperparams['dec_channel_str'],
    ).to(device)

    state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
    vae.load_state_dict(state_dict, strict=False)
    vae.eval()
    
    return vae

# Image Preprocessing
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to a fixed size
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    return img, img_tensor

# Image merging and fine-tuning
def merge_latents_and_reconstruct(image_path1, image_path2, alpha, vae):
    img1, img_tensor1 = load_and_preprocess_image(image_path1)
    img2, img_tensor2 = load_and_preprocess_image(image_path2)
    
    with torch.no_grad():
        mu1, logvar1 = vae.enc(img_tensor1)
        z1 = vae.reparameterize(mu1, logvar1)

        mu2, logvar2 = vae.enc(img_tensor2)
        z2 = vae.reparameterize(mu2, logvar2)

        merged_z = (1 - alpha) * z1 + alpha * z2
        recon_img = vae.decode(merged_z).squeeze(0).cpu()  # Ensure the result is moved to the CPU for visualization
    
    return img1, img2, recon_img

# Streamlit interface
def main():
    st.title("VAE Image Merger")

    # Download model weights from Google Drive
    download_model_weights()

    # Load the model
    vae = load_model()

    # Check if user has uploaded images
    uploaded_image1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
    uploaded_image2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

    # If no images uploaded, use demo images
    if uploaded_image1 is None or uploaded_image2 is None:
        # Show demo images
        demo_image1_path = '1.jpg'  # Replace with your demo image path
        demo_image2_path = '20.jpg'  # Replace with your demo image path

        demo_image1 = Image.open(demo_image1_path)
        demo_image2 = Image.open(demo_image2_path)

        st.image(demo_image1, caption="Demo Image 1", use_column_width=True)
        st.image(demo_image2, caption="Demo Image 2", use_column_width=True)

        # Merge demo images without fine-tuning
        alpha = 50 / 100  # Default alpha for demo images
        img1, img2, recon_img = merge_latents_and_reconstruct(demo_image1_path, demo_image2_path, alpha, vae)

        recon_pil = transforms.ToPILImage()(recon_img)
        st.image(recon_pil, caption="Merged Demo Image", use_column_width=True)
    else:
        # Slider for alpha blending
        alpha = st.slider("Silde to observe changes", min_value=30, max_value=70, value=50)

        fine_tune_button = st.button("Generate Merged Image")
        if fine_tune_button:
            # Show waiting message
            st.info("Fine-tuning in progress, please wait...")
            
            # Fine-tune and merge
            img1, img2, recon_img = merge_latents_and_reconstruct(uploaded_image1, uploaded_image2, alpha / 100, vae)

            # Display the original and merged images
            st.image(img1, caption="Uploaded Image 1", use_column_width=True)
            st.image(img2, caption="Uploaded Image 2", use_column_width=True)

            # Display the reconstructed (merged) image
            recon_pil = transforms.ToPILImage()(recon_img)
            st.image(recon_pil, caption="Merged Image", use_column_width=True)

if __name__ == "__main__":
    main()
