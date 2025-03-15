###pip install streamlit
###Run: streamlit run frontend.py    or     python -m streamlit run frontend.py
import streamlit as st
import torch
from gensim.models import Word2Vec
import torch.nn as nn
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.init as init
import re

# ------------------------------
# 1) Load or define your Word2Vec
# ------------------------------
@st.cache_resource
def load_w2v_model():
    # Adjust the path to your "bare_minimum" model if needed
    w2v = Word2Vec.load("bare_minimum")
    w2vjp = Word2Vec.load("bare_minimumjp")
    return w2v, w2vjp

# ------------------------------
# 2) Define your Encoder/Decoder
# ------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(16, 128, bias=False)
        self.dense2 = nn.Linear(128,768,bias=False)
        # Try upscaling more linearly, but make sure it is a square number that can be upscaled by 2 to 128
        
        self.upsample1 = nn.ConvTranspose2d(3, 3, kernel_size = 4, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3,3,3,padding=1, bias=False)
        self.conv12 = nn.Conv2d(3,3,5,padding=2, bias=False)
        self.upsample2 = nn.ConvTranspose2d(3, 3, kernel_size = 4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(3,3,3,padding=1, bias=False)
        self.conv22 = nn.Conv2d(3,3,5,padding=2, bias=False)
        self.upsample3 = nn.ConvTranspose2d(3, 3, kernel_size = 4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(3,3,3,padding=1, bias=False)
        self.conv32 = nn.Conv2d(3,3,5,padding=2, bias=False)
        # Try modifying the in/out channels and kernel sizes here. 
        # Can also add normal Conv layers between these, just make sure padding = kernel_size // 2
        
        self.mesh = nn.Conv2d(6, 3, 3, padding=1, bias=False)
        # Try adding additional layers here, since this is the place where
        # the current sentence image is merged with the word
        # NEW: Skip connection layers to merge low- and high-level features
        self.skip1 = nn.Conv2d(3, 3, 1, bias=False)  # Merges features after first upsample
        self.skip2 = nn.Conv2d(3, 3, 1, bias=False)  # Merges features after second upsample
        self.skip3 = nn.Conv2d(3, 3, 1, bias=False)  # Merges features after third upsample

        # NEW: Edge enhancement layers to reduce blur
        self.sharp1 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.sharp2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        
    def forward(self, word_vec, simage):
        
        x = self.dense1(word_vec)
        x = nn.functional.relu(x)
        
        x = self.dense2(x)
        x = nn.functional.relu(x)
        
        x = x.reshape((-1,3,16,16))
        
        x = self.upsample1(x)
        x = self.sharp1(x) 
        x = nn.functional.relu(x)
        skip_x1 = self.skip1(x)  # NEW: Capture details
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv12(x)
        x = nn.functional.relu(x)

        x = self.upsample2(x)
        x = self.sharp2(x) 
        x = nn.functional.relu(x)
        skip_x2 = self.skip2(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv22(x)
        x = nn.functional.relu(x)

        x = self.upsample3(x)
        x = nn.functional.relu(x) # <- This activation function should result in something image-like, relu isn't great
        skip_x3 = self.skip3(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv32(x)
        x = nn.functional.relu(x)

        # Make modifications here too
        # Upsample skip_x1 and skip_x2 to match x's dimensions
        skip_x1_resized = nn.functional.interpolate(skip_x1, size=x.shape[2:], mode="bilinear", align_corners=True)
        skip_x2_resized = nn.functional.interpolate(skip_x2, size=x.shape[2:], mode="bilinear", align_corners=True)

        # Merge the resized skip connections with x
        x = x + skip_x1_resized + skip_x2_resized + skip_x3
        
        x = torch.concat((simage, x), dim=1)
        return self.mesh(x)
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        
        self.pool1 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(3,6,3,padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2)
        self.conv6 = nn.Conv2d(6,1,3,padding=1, bias=False)
        
        self.flatten = nn.Flatten()
        self.classify1 = nn.Linear(266,128, bias=False)
        self.classify2 = nn.Linear(128,64, bias=False)
        self.classify3 = nn.Linear(64, 10, bias=False)
        
        self.next1 = nn.Conv2d(3,16,3,padding=1, bias=False)
        self.next2 = nn.Conv2d(16,16,3,padding=1, bias=False)
        self.next3 = nn.Conv2d(16, 3, 1, bias=False)
        
        
    def forward(self, input, context):
        
        x = self.conv1(input)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        
        c = self.pool1(x)
        c = self.conv4(c)
        c = nn.functional.relu(c)
        c = self.pool2(c)
        c = self.conv5(c)
        c = nn.functional.relu(c)
        c = self.pool3(c)
        c = self.conv6(c)
        c = nn.functional.relu(c)

        c = self.flatten(c)
        c = torch.concat((c, context), dim=1)
        c = self.classify1(c)
        c = nn.functional.relu(c)
        c = self.classify2(c)
        c = nn.functional.relu(c)
        c = self.classify3(c)
        
        n = self.next1(x)
        n = nn.functional.relu(n)
        n = self.next2(n)
        n = nn.functional.relu(n)
        n = self.next3(n)
        n = nn.functional.sigmoid(n)
        
        return c, n

# ------------------------------
# 3) Initialize (or load) Models
# ------------------------------
@st.cache_resource
def init_models():
    torch.manual_seed(42)
    enc = torch.load("encoder1",map_location=torch.device('cpu'), weights_only=False)
    dec = torch.load("decoder1",map_location=torch.device('cpu'), weights_only=False)
    encjp = torch.load("encoder1jp",map_location=torch.device('cpu'), weights_only=False)
    
    return enc, dec, encjp

# ------------------------------
# 5) Streamlit UI
# ------------------------------
def main():
    st.title("🔎 Found In Translation")

    # Language selection toggle
    language = st.radio("Choose Language:", ["English", "Japanese"])

    # Load resources
    w2v, w2vjp = load_w2v_model()
    enc, dec, encjp = init_models()

    # Select appropriate encoder & Word2Vec model based on language
    if language == "English":
        encoder = enc
        w2v_model = w2v
    else:
        encoder = encjp
        w2v_model = w2vjp

    # User input
    user_sentence = st.text_input(f"Enter a sentence in {language}:")

    if st.button("Generate Image"):
        # Convert user sentence into a list of words
        words = re.findall(r'\b\w+\b', user_sentence.lower())
        if not words:
            st.error("Please enter a valid sentence.")
            return

        # Generate the image
        canvas = torch.zeros(1, 3, 128, 128)
        for w in words:
            if w not in w2v_model.wv.key_to_index:
                st.warning(f"Word '{w}' not in vocabulary. Using zero vector.")
                w_vec = torch.zeros((1, 16))
            else:
                w_vec = torch.from_numpy(w2v_model.wv[w]).float().unsqueeze(0)
            canvas = encoder(w_vec, canvas)

        # Decode sentence
        pic = canvas.clone()
        context = torch.zeros((1, 10))
        caption = []
        for w in words:
            word, canvas = dec(canvas, context)
            context = word
            caption.append(word)

        # Convert to numpy and show
        output_np = pic.detach().clone().squeeze().permute(1, 2, 0).numpy()
        
        # Display output image with decoded sentence
        decoded_sentence =" ".join([w2v_model.wv.index_to_key[w.argmax()] for w in caption])
        decoded_caption = "Decoded sentence:"+" ".join([w2v.wv.index_to_key[w.argmax()] for w in caption])
        st.image(output_np[:, :, ::-1], clamp=True, use_container_width=True)
        st.write(f"<div style='text-align:center;'><p style='font-size:24px;'><b>{decoded_caption}</b></p><div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()
