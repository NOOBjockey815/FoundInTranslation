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

# ------------------------------
# 1) Load or define your Word2Vec
# ------------------------------
@st.cache_resource
def load_w2v_model():
    # Adjust the path to your "bare_minimum" model if needed
    w2v = Word2Vec.load("bare_minimum")
    return w2v

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
    enc = torch.load("encoder1", weights_only=False)
    dec = torch.load("decoder1", weights_only=False)
    
    return enc, dec

# ------------------------------
# 5) Streamlit UI
# ------------------------------
def main():
    st.title("Simple Text-to-Image Demo (Trained on 3 Sentences)")

    # 1) Load resources
    w2v = load_w2v_model()
    enc, dec = init_models()

    # 3) Let user input one of the three sentences
    st.write("Try one of these sentences:")
    st.write("- `the bear is eating honey`")
    st.write("- `where is the hospital`")
    st.write("- `students are striking`")
    user_sentence = st.text_input("Enter one of the trained sentences here:")

    if st.button("Generate Image"):
        # Convert user sentence into a list of words
        words = user_sentence.strip().split()
        if not words:
            st.error("Please enter a non-empty sentence.")
            return
        
        # 4) Generate the image
        canvas = torch.zeros(1,3,128,128)
        for w in words:
            if w not in w2v.wv.key_to_index:
                st.warning(f"Word '{w}' not in vocabulary. Using zero vector.")
                w_vec = torch.zeros((1,16))
            else:
                w_vec = torch.from_numpy(w2v.wv[w]).float().unsqueeze(0)
            canvas = enc(w_vec, canvas)
        
        # 5) Convert to numpy and show
        # Optionally clamp or apply sigmoid if needed
        output_np = canvas.detach().clone()
        # If you want a [0,1] range:
        output_np = torch.sigmoid(output_np)
        output_np = output_np.squeeze().permute(1,2,0).numpy()
        print(output_np)
        # Show with Streamlit
        st.image(output_np, caption="Generated Image", clamp=True)

if __name__ == "__main__":
    main()
