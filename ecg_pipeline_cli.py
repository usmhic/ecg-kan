!pip install ipywidgets

import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.svm import OneClassSVM
import ipywidgets as widgets
from IPython.display import display, clear_output
from kan import KAN
from io import BytesIO
import zipfile

# Non-Technical Explanation:
# This interactive script lets you pick a person (subject) from a dataset of heartbeats, choose a model to "learn" that person's unique heartbeat pattern, and then test if it can recognize new heartbeats as belonging to that person or someone else.
# Easy Steps:
# 1. Choose data source: Use the built-in ECG dataset or upload your own (as a ZIP of folders with images).
# 2. Select the person to "train" on.
# 3. Pick a model (like a smart checker for patterns).
# 4. Click "Train Model" - this teaches the model the person's pattern.
# 5. Click "Test Random Beat" - it picks a random heartbeat (from the person or someone else) and says if it's recognized.
# You can repeat testing as many times as you want.

# Technical Explanation:
# - Data Loading: From ECG-ID WTMM images or uploaded ZIP (extracted to folders like 'subject_1' with PNG images).
# - Models: OneClassSVM (flattened images), OneClassCNN (CNN autoencoder), OneClassKAN (KAN autoencoder).
# - Training: On selected subject's images (80% train, 20% test for genuine).
# - Testing: Random selection from test genuine or impostor (other subjects), using anomaly score vs threshold.
# - Interactivity: Using ipywidgets for dropdowns, buttons, file upload.

# User settings (adjust if needed)
IMAGE_SIZE = 64
BATCH_SIZE = 4
EPOCHS = 20  # Reduced for faster training
LR = 0.001
EXISTING_DATA_DIR = './ecg_wtmm_images'  # Existing ECG-ID data

transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        return self.transform(image) if self.transform else image

# One-Class CNN (Convolutional Autoencoder)
class OneClassCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        flatten_dim = 128 * (IMAGE_SIZE // 8) ** 2
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, IMAGE_SIZE // 8, IMAGE_SIZE // 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# One-Class KAN (KAN Autoencoder using pykan)
class OneClassKAN(nn.Module):
    def __init__(self):
        super().__init__()
        flatten_dim = IMAGE_SIZE * IMAGE_SIZE
        self.encoder = KAN(width=[flatten_dim, 256, 64], grid=3, k=2)  # Reduced complexity for speed
        self.decoder = KAN(width=[64, 256, flatten_dim], grid=3, k=2)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)
        return decoded.view(x.shape).sigmoid()

def train_model(model, dataloader, epochs=EPOCHS, lr=LR):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            output = model(data)
            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}')
    return model

def get_ae_scores(model, dataloader):
    criterion = nn.MSELoss(reduction='none')
    scores = []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            output = model(data)
            loss = criterion(output, data).mean([1, 2, 3])
            scores.extend(loss.tolist())
    return np.array(scores)

def get_svm_scores(model, loader):
    features = np.concatenate([data.view(len(data), -1).numpy() for data in loader], axis=0)
    return -model.decision_function(features)

# Interactive Widgets Setup
data_source = widgets.Dropdown(options=['ECG-ID Dataset', 'Upload Own Dataset'], description='Data Source:')
upload_widget = widgets.FileUpload(accept='.zip', multiple=False, description='Upload ZIP')
subject_dropdown = widgets.Dropdown(description='Subject:')
model_dropdown = widgets.Dropdown(options=['OneClassSVM', 'OneClassCNN', 'OneClassKAN'], description='Model:')
train_button = widgets.Button(description='Train Model')
test_button = widgets.Button(description='Test Random Beat')
output = widgets.Output()

# Global variables for data and models
all_images = {}
trained_model = None
model_type = None
threshold = None
test_genuine_images = []
impostor_images = []

def load_existing_data():
    global all_images
    all_images = {}
    for subj_id in range(1, 11):  # Assuming 10 subjects
        subj_dir = os.path.join(IMAGE_DIR, f'subject_{subj_id}')
        if os.path.exists(subj_dir):
            images = [os.path.join(subj_dir, f) for f in os.listdir(subj_dir) if f.endswith('.png')]
            all_images[subj_id] = images
    subject_dropdown.options = [f'Subject {k}' for k in all_images.keys()]

def extract_zip(file_data):
    with zipfile.ZipFile(BytesIO(file_data), 'r') as zip_ref:
        zip_ref.extractall(IMAGE_DIR)
    load_existing_data()

def on_data_source_change(change):
    if change['new'] == 'ECG-ID Dataset':
        upload_widget.layout.display = 'none'
        load_existing_data()
    else:
        upload_widget.layout.display = 'block'

def on_upload_change(change):
    if upload_widget.value:
        file_data = list(upload_widget.value.values())[0]['content']
        extract_zip(file_data)
        output.clear_output()
        with output:
            print("Dataset uploaded and extracted.")

data_source.observe(on_data_source_change, names='value')

def on_train(button):
    global trained_model, model_type, threshold, test_genuine_images, impostor_images
    output.clear_output()
    with output:
        print("Starting training...")
    subject_str = subject_dropdown.value
    subject_id = int(subject_str.split()[1])
    genuine_images = all_images.get(subject_id, [])
    if not genuine_images:
        with output:
            print("No data for selected subject.")
        return
    np.random.shuffle(genuine_images)
    split = int(0.8 * len(genuine_images))
    train_images = genuine_images[:split]
    test_genuine_images = genuine_images[split:]
    impostor_images = [img for s_id, imgs in all_images.items() if s_id != subject_id for img in imgs]
    train_loader = DataLoader(ImageDataset(train_images, transform), batch_size=BATCH_SIZE, shuffle=True)
    model_name = model_dropdown.value
    if model_name == 'OneClassSVM':
        train_features = np.concatenate([data.view(len(data), -1).numpy() for data in train_loader], axis=0)
        model = OneClassSVM(kernel='rbf', nu=0.1).fit(train_features)
        train_scores = -model.decision_function(train_features)
        model_type = 'SVM'
    elif model_name == 'OneClassCNN':
        model = OneClassCNN()
        model = train_model(model, train_loader)
        train_scores = get_ae_scores(model, train_loader)
        model_type = 'AE'
    elif model_name == 'OneClassKAN':
        model = OneClassKAN()
        model = train_model(model, train_loader)
        train_scores = get_ae_scores(model, train_loader)
        model_type = 'AE'
    threshold = np.mean(train_scores) + 3 * np.std(train_scores)
    trained_model = model
    with output:
        print("Training complete. Ready for testing.")

def on_test(button):
    output.clear_output()
    if trained_model is None:
        with output:
            print("Please train the model first.")
        return
    is_genuine = np.random.choice([True, False])
    if is_genuine:
        if not test_genuine_images:
            with output:
                print("No more genuine test beats.")
            return
        test_path = np.random.choice(test_genuine_images)
        label = "Genuine"
    else:
        if not impostor_images:
            with output:
                print("No impostor beats.")
            return
        test_path = np.random.choice(impostor_images)
        label = "Impostor"
    test_loader = DataLoader(ImageDataset([test_path], transform), batch_size=1)
    if model_type == 'SVM':
        features = np.concatenate([data.view(len(data), -1).numpy() for data in test_loader], axis=0)
        score = -trained_model.decision_function(features)[0]
    else:
        score = get_ae_scores(trained_model, test_loader)[0]
    recognized = score < threshold
    result = "Recognized as Genuine" if recognized else "Rejected as Impostor"
    with output:
        print(f"Test Beat: {os.path.basename(test_path)}")
        print(f"Actual: {label}")
        print(f"Score: {score:.4f} (Threshold: {threshold:.4f})")
        print(f"Result: {result}")
        print(f"Non-Technical: The model checked this heartbeat and decided it's {'the same person' if recognized else 'someone else'}.")

# Display widgets
display(data_source)
display(upload_widget)
display(subject_dropdown)
display(model_dropdown)
display(train_button)
display(test_button)
display(output)

train_button.on_click(on_train)
test_button.on_click(on_test)
upload_widget.observe(on_upload_change, names='value')
on_data_source_change({'new': data_source.value})