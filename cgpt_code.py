import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2
import numpy as np

# ---------------------------
# 1. Image Preprocessing
# ---------------------------
def load_image(path, size=(224, 224)):
    """
    Loads an image from a file, resizes it, converts BGR to RGB,
    normalizes it with ImageNet statistics, and returns a float32 tensor.
    """
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # Ensure mean and std are float32
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # Convert from HWC to CHW format and add batch dimension
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)  # Remains as float32
    return img

# ---------------------------
# 2. Dataset Preparation
# ---------------------------
captions = [
    "the bear is eating honey",
    "where is the hospital",
    "students are striking"
]
image_paths = ["bear.jpg", "hos.png", "strike.jpg"]

# Build a simple vocabulary from the captions; include special tokens.
special_tokens = ['<pad>', '<start>', '<end>']
all_words = []
for cap in captions:
    all_words.extend(cap.split())
vocab = special_tokens + sorted(list(set(all_words)))
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

# ---------------------------
# 3. Model Definitions
# ---------------------------
# CNN Encoder: uses a pretrained ResNet18 to extract image features.
class CNN_Encoder(nn.Module):
    def __init__(self, embed_size):
        super(CNN_Encoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove final fully connected layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        # Replaced BatchNorm1d with LayerNorm to avoid issues with batch size 1.
        self.ln = nn.LayerNorm(embed_size)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)  # shape: (batch, 512, 1, 1)
        features = features.view(features.size(0), -1)  # shape: (batch, 512)
        features = self.linear(features)               # shape: (batch, embed_size)
        features = self.ln(features)
        return features

# RNN Decoder: an LSTM that generates captions.
class RNN_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(RNN_Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        """
        Expects captions as tensor of shape (batch, seq_length).
        Prepend image features to the embedded captions.
        """
        embeddings = self.embed(captions)              # (batch, seq_length, embed_size)
        features = features.unsqueeze(1)               # (batch, 1, embed_size)
        embeddings = torch.cat((features, embeddings), dim=1)  # (batch, seq_length+1, embed_size)
        outputs, _ = self.lstm(embeddings)
        outputs = self.linear(outputs)                 # (batch, seq_length+1, vocab_size)
        return outputs

    def sample(self, features, max_len=20):
        """
        Greedy decoding for inference.
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)       # (batch, 1, embed_size)
        states = None
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))      # (batch, vocab_size)
            _, predicted = outputs.max(1)                  # (batch)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)                 # (batch, embed_size)
            inputs = inputs.unsqueeze(1)
            if predicted.item() == word2idx['<end>']:
                break
        return sampled_ids

# ---------------------------
# 4. Training Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = 256
hidden_size = 512

encoder = CNN_Encoder(embed_size).to(device)
decoder = RNN_Decoder(embed_size, hidden_size, vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
# Optimize only the decoder and the encoder's linear and normalization layers.
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.ln.parameters())
optimizer = optim.Adam(params, lr=0.001)

# ---------------------------
# 5. Training Loop
# ---------------------------
num_epochs = 15
for epoch in range(num_epochs):
    total_loss = 0.0
    for img_path, cap in zip(image_paths, captions):
        # Load and preprocess image
        image = load_image(img_path, size=(224,224)).to(device)
        # Tokenize caption and add <start> and <end> tokens
        tokens = ['<start>'] + cap.split() + ['<end>']
        caption_idx = [word2idx[word] for word in tokens]
        caption_tensor = torch.tensor(caption_idx).unsqueeze(0).to(device)  # (1, seq_length)
        
        optimizer.zero_grad()
        features = encoder(image)
        # Use teacher forcing: feed caption (excluding the last token) to the decoder
        outputs = decoder(features, caption_tensor[:, :-1])
        # Compare outputs with actual caption tokens (shifted by one)
        loss = criterion(outputs.reshape(-1, vocab_size), caption_tensor.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# ---------------------------
# 6. Inference
# ---------------------------
encoder.eval()
decoder.eval()
for img_path in image_paths:
    image = load_image(img_path, size=(224,224)).to(device)
    features = encoder(image)
    sampled_ids = decoder.sample(features, max_len=10)
    sampled_caption = []
    for word_id in sampled_ids:
        word = idx2word[word_id]
        if word == '<end>':
            break
        sampled_caption.append(word)
    sentence = ' '.join(sampled_caption)
    print(f"Generated caption for {img_path}: {sentence}")
