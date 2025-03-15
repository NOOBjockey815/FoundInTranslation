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
        
        self.upsample1 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3,3,3,padding=1, bias=False)
        self.conv12 = nn.Conv2d(3,3,5,padding=2, bias=False)
        self.upsample2 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(3,3,3,padding=1, bias=False)
        self.conv22 = nn.Conv2d(3,3,5,padding=2, bias=False)
        self.upsample3 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(3,3,3,padding=1, bias=False)
        self.conv32 = nn.Conv2d(3,3,5,padding=2, bias=False)

        self.mesh = nn.Conv2d(6, 3, 3, padding=1, bias=False)

    def forward(self, word_vec, simage):
        x = self.dense1(word_vec)
        x = nn.functional.relu(x)
        x = self.dense2(x)
        x = nn.functional.relu(x)
        
        x = x.reshape((-1,3,16,16))
        
        x = self.upsample1(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv12(x)
        x = nn.functional.relu(x)
        x = self.upsample2(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv22(x)
        x = nn.functional.relu(x)
        x = self.upsample3(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv32(x)
        x = nn.functional.relu(x)

        x = torch.concat((simage, x), dim=1)
        x = self.mesh(x)
        # No final activation => output can be any range
        return x

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
    enc = Encoder()
    dec = Decoder()
    s = nn.ModuleList([enc, dec])
    
    # Optional: Initialize weights (Xavier)
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            init.xavier_uniform_(m.weight)
    s.apply(weights_init)
    
    return enc, dec

# ------------------------------
# 4) (Optional) Train the Models
#    - In a real app, you'd load pretrained weights instead.
# ------------------------------
def train_models(enc, dec, w2v):
    # Just as in your code, we do minimal training on the 3 images + sentences
    # for demonstration. This can be time-consuming in practice.

    # Prepare images
    bear = cv2.imread("bear.jpg")
    bear = cv2.resize(bear,(128,128))
    hospital = cv2.imread("hos.png")
    hospital = cv2.resize(hospital,(128,128))
    hospital[100:,:,:] = 255
    strike = cv2.imread("strike.jpg")
    strike = cv2.resize(strike,(128,128))

    imgs = np.array([(i.reshape(1,128,128,3) / 256).astype(np.float32) for i in [bear, hospital, strike]])
    images = torch.tensor(imgs).permute(0,1,4,2,3)

    sentences = [
        ["the", "bear", "is", "eating", "honey"],
        ["where", "is", "the", "hospital"],
        ["students", "are", "striking"]
    ]

    crit_words = nn.CrossEntropyLoss()
    crit_image = nn.MSELoss()
    optim = torch.optim.Adam(nn.ModuleList([enc, dec]).parameters())

    # Quick "light" training
    # Feel free to reduce or remove for demonstration
    for epoch in range(10):
        total_loss = 0
        for img, sent in zip(images, sentences):
            optim.zero_grad()
            canvas = torch.zeros(1,3,128,128)
            # Pass each word
            for w in sent:
                w_vec = torch.from_numpy(w2v.wv[w]).float().unsqueeze(0)
                canvas = enc(w_vec, canvas)
            # Image loss
            loss = crit_image(img, canvas)
            
            # Word classification
            context = torch.zeros(1,10)
            for w in sent:
                w_vec, canvas = dec(canvas, context)
                context = w_vec
                loss += crit_words(w_vec, torch.tensor([w2v.wv.key_to_index[w]]))
            
            loss.backward()
            optim.step()
            total_loss += loss.item()
        # st.write(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    return enc, dec

# ------------------------------
# 5) Streamlit UI
# ------------------------------
def main():
    st.title("Simple Text-to-Image Demo (Trained on 3 Sentences)")

    # 1) Load resources
    w2v = load_w2v_model()
    enc, dec = init_models()

    # 2) (Optional) Train or load pretrained
    st.write("Training models for demonstration (only 10 epochs). This might take a bit...")
    enc, dec = train_models(enc, dec, w2v)
    st.write("Training complete!")

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

        # Show with Streamlit
        st.image(output_np, caption="Generated Image", clamp=True)

if __name__ == "__main__":
    main()
