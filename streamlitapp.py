import os
from PIL import Image
import numpy as np
import streamlit as st
import config
from model import load_model
import torch
from matplotlib import pyplot as plt

st.set_page_config(layout='wide')
with st.sidebar:
    st.image(
        'https://imageio.forbes.com/specials-images/imageserve/5f51c38ba72e09805e578c53/3-Predictions-For-The-Role-Of'
        '-Artificial-Intelligence-In-Art-And-Design/960x0.jpg?format=jpg&width=960')
    st.title("Generative Adversarial Networks")
    st.info(
        "This is an image to image model that changes the characteristics of an image. In this case this is able to color the drawings of anime characters.This can also change the satellite view to roadmap view")
    model_type = st.selectbox("Choose a model", options=["anime", "maps"])

st.title("Image-to-Image translation with Pix2Pix")

tab1,tab2 = st.tabs(["Demonstration","About"])
with tab1:
    if model_type == "anime":
        options = os.listdir(os.path.join('anime', 'images'))
        selected_photo = st.selectbox("Choose a photo", options=options)

        img_path = os.path.join('anime', 'images', selected_photo)
        image = np.array(Image.open(img_path))

        input_image = image[:512, 512:, :3]
        target_image = image[:512, 512:, :3]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        target_image = config.transform_only_mask(image=target_image)["image"]
        input_image = input_image[None, :, :, :]
        target_image = target_image[None, :, :, :]
        path = "./anime_model/gen.pth.tar"
        model = load_model(path)
        model.eval()

        col1, col2 = st.columns(2)
        if options:
            with col1:
                img_path = os.path.join('anime', 'images', selected_photo)
                image = np.array(Image.open(img_path))

                input_image = image[:512, 512:, :3]
                input_image = input_image[None, :, :, :]
                st.image(input_image)

            with col2:
                with torch.no_grad():
                    y_fake = model(target_image)
                    y_fake = y_fake * 0.5 + 0.5
                    y = torch.squeeze(y_fake).permute(1, 2, 0)
                    final_arr = np.array(y.cpu())
                    final_arr = final_arr[:,:,:3]
                    final_image = Image.fromarray((final_arr*255).astype(np.uint8))
                    final_image = final_image.resize((512, 512))
                    st.image(final_image)

    elif model_type == "maps":
        options = os.listdir(os.path.join('maps', 'images'))
        selected_photo = st.selectbox("Choose a photo", options=options)

        img_path = os.path.join('maps', 'images', selected_photo)
        image = np.array(Image.open(img_path))

        input_image = image[:600, :600, :3]
        target_image = image[:600, :600, :3]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        target_image = config.transform_only_mask(image=target_image)["image"]
        input_image = input_image[None, :, :, :]
        target_image = target_image[None, :, :, :]
        path = "./maps_model/gen.pth.tar"
        model = load_model(path)
        model.eval()

        col1, col2 = st.columns(2)
        if options:
            with col1:
                img_path = os.path.join('maps', 'images', selected_photo)
                image = np.array(Image.open(img_path))

                input_image = image[:600, :600, :3]
                input_image = input_image[None, :, :, :]
                st.image(input_image)

            with col2:
                with torch.no_grad():
                    y_fake = model(target_image)
                    y_fake = y_fake * 0.5 + 0.5
                    y = torch.squeeze(y_fake).permute(1, 2, 0)
                    final_arr = np.array(y.cpu())
                    final_arr = final_arr[:, :, :3]
                    final_image = Image.fromarray((final_arr * 255).astype(np.uint8))
                    final_image = final_image.resize((512, 512))
                    st.image(final_image)

with tab2:
    st.markdown("""
                <p style="font-size:20px; text-align:center">Pix2Pix GAN is an architecture designed for image-to-image translation tasks, where the goal is to convert an input image from one domain to another.The generator network in Pix2Pix GAN consists of an encoder-decoder structure. The encoder takes the input image and progressively reduces its spatial dimensions, capturing high-level features. The decoder then takes the encoded representation and progressively upsamples it, generating the output image. The discriminator network in Pix2Pix GAN is similar to the discriminator in traditional GANs. </p>
                <img style="display: block; margin-left: auto; margin-right: auto;padding: 5px;" src="https://neurohive.io/wp-content/uploads/2018/11/Capture-8.jpg" />
                """,unsafe_allow_html=True)