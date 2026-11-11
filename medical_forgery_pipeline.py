"""
Medical Image Forgery Detection and Localization
Research prototype based on the supplied TransGAN + Swin + MedSAM diagram.

What this script does:
1. Reads authentic images from CQ500 and/or MIMIC-CXR folders.
2. Creates controlled synthetic copy-move/splicing/inpainting-style forgeries.
3. Saves paired forged images and binary pixel masks.
4. Trains a Swin-Tiny multi-scale feature model for image classification and mask prediction.
5. Includes a lightweight transformer GAN generator/discriminator as an auxiliary adversarial module.
6. Optionally refines predicted masks with the official MedSAM predictor at inference time.

Important:
- Synthetic masks are generated labels, not naturally occurring clinical forgery annotations.
- This is a research prototype, not a clinical diagnostic system.
- MedSAM refinement is optional and requires the official MedSAM repository and checkpoint.
"""

# ------------------------------ IMPORTS ---------------------------------------
import argparse                         # Reads command-line options.
import csv                              # Writes and reads dataset manifests.
import random                           # Samples random forgery locations and augmentation choices.
from pathlib import Path                # Handles Windows and Linux paths safely.

import numpy as np                      # Performs image and mask array operations.
from PIL import Image, ImageFilter      # Loads, resizes, pastes, and saves images.
import torch                            # Provides tensors and GPU computation.
import torch.nn as nn                   # Provides neural-network layers.
import torch.nn.functional as F          # Provides differentiable functions.
from torch.utils.data import Dataset, DataLoader  # Provides dataset and batching utilities.
from torchvision import transforms      # Provides image transformations.
from torchvision.transforms import InterpolationMode  # Controls resize interpolation.

# Optional medical-image reader. Install pydicom to read DICOM files.
try:
    import pydicom                      # Reads DICOM medical image files.
except ImportError:
    pydicom = None                      # Keeps JPG/PNG workflows usable without pydicom.

# Swin is supplied by the timm model library.
try:
    import timm                          # Provides pretrained Swin Transformer backbones.
except ImportError:
    timm = None                          # Gives a clear error when model creation is attempted.


# ------------------------------ CONFIGURATION --------------------------------
ROOT = Path(r"D:\Medical_Forgery_Project")  # Main project directory on the D: drive.
RAW_CQ500 = ROOT / "datasets" / "CQ500" / "raw"  # Original CQ500 files.
RAW_MIMIC = ROOT / "datasets" / "MIMIC_CXR" / "raw"  # Original MIMIC-CXR files.
DATA_ROOT = ROOT / "datasets" / "Forgery_Dataset"  # Generated train/val/test dataset.
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"  # Trained model checkpoints.
OUTPUT_DIR = ROOT / "outputs"  # Predictions, masks, heatmaps, and metrics.
IMAGE_SIZE = 224                        # Model input height and width.
BATCH_SIZE = 4                          # Small batch size for an 8-GB GPU.
EPOCHS = 30                             # Initial training duration; increase after validation.
LEARNING_RATE = 1e-4                    # Initial AdamW learning rate.
NUM_WORKERS = 0                         # Safe default for Windows notebooks and scripts.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Selects GPU when available.

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}  # Supported image formats.


# ------------------------------ IMAGE UTILITIES ------------------------------
def read_medical_image(path):
    """Load a DICOM or common image file and return a normalized grayscale PIL image."""
    path = Path(path)                   # Converts the input to a Path object.
    if path.suffix.lower() == ".dcm":   # Checks whether the file is DICOM.
        if pydicom is None:             # Checks whether the DICOM library is installed.
            raise ImportError("Install pydicom: pip install pydicom")
        dicom = pydicom.dcmread(str(path))  # Reads DICOM metadata and pixel data.
        pixels = dicom.pixel_array.astype(np.float32)  # Converts pixels to floating point.
        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = pixels.max() - pixels  # Inverts MONOCHROME1 images for display.
        low, high = np.percentile(pixels, [1, 99])  # Robustly estimates display intensity limits.
        pixels = np.clip(pixels, low, high)  # Removes extreme intensity outliers.
        pixels = (pixels - low) / max(high - low, 1e-6)  # Scales intensities to [0, 1].
        pixels = (pixels * 255).astype(np.uint8)  # Converts normalized values to 8-bit pixels.
        return Image.fromarray(pixels).convert("L")  # Returns grayscale PIL image.
    return Image.open(path).convert("L")  # Loads ordinary image formats as grayscale.


def make_forgery(image):
    """
    Create a controlled synthetic manipulation and its binary ground-truth mask.
    The returned mask is 255 inside the manipulated region and 0 elsewhere.
    """
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))  # Standardizes image size.
    forged = image.copy()               # Creates a separate image to manipulate.
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)  # Initializes an all-black mask.
    width = random.randint(35, 85)       # Samples manipulation width.
    height = random.randint(35, 85)      # Samples manipulation height.
    x = random.randint(0, IMAGE_SIZE - width)  # Samples a valid horizontal position.
    y = random.randint(0, IMAGE_SIZE - height)  # Samples a valid vertical position.
    box = (x, y, x + width, y + height)  # Defines the manipulated rectangular region.
    mode = random.choice(["copy_move", "splice", "inpaint_like"])  # Chooses manipulation type.

    if mode == "copy_move":              # Copy-move duplicates a region from the same image.
        sx = random.randint(0, IMAGE_SIZE - width)  # Samples source horizontal position.
        sy = random.randint(0, IMAGE_SIZE - height)  # Samples source vertical position.
        patch = image.crop((sx, sy, sx + width, sy + height))  # Extracts the source patch.
        forged.paste(patch, (x, y))      # Pastes the patch at the target position.
    elif mode == "splice":               # Splicing inserts a patch from another image.
        patch = image.crop((0, 0, width, height)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        forged.paste(patch, (x, y))      # Inserts a transformed patch as a simple splice proxy.
    else:                                # Inpainting-like manipulation removes local texture.
        patch = image.crop(box).filter(ImageFilter.GaussianBlur(radius=8))
        forged.paste(patch, (x, y))      # Replaces the region with a blurred approximation.

    mask.paste(255, box)                 # Marks the manipulated region in the ground-truth mask.
    return forged, mask                 # Returns the synthetic forgery and its mask.


def discover_images(folder):
    """Recursively find supported image files inside a folder."""
    folder = Path(folder)                # Converts the folder to a Path object.
    if not folder.exists():              # Checks whether the folder exists.
        return []                        # Returns an empty list for a missing folder.
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def prepare_dataset():
    """Create train/val/test folders containing authentic and synthetic forged pairs."""
    sources = discover_images(RAW_CQ500) + discover_images(RAW_MIMIC)  # Combines both source datasets.
    if not sources:                      # Stops if no source images were found.
        raise FileNotFoundError(
            f"No images found. Put CQ500 files in {RAW_CQ500} and MIMIC-CXR files in {RAW_MIMIC}."
        )

    random.seed(42)                      # Makes the split reproducible.
    random.shuffle(sources)              # Randomizes source order before splitting.
    n = len(sources)                     # Counts source images.
    train_end = int(0.70 * n)            # Uses 70 percent for training.
    val_end = int(0.85 * n)              # Uses 15 percent for validation.
    splits = {
        "train": sources[:train_end],    # Assigns training source images.
        "val": sources[train_end:val_end],  # Assigns validation source images.
        "test": sources[val_end:],       # Assigns test source images.
    }

    for split_name, split_sources in splits.items():  # Processes each dataset split.
        image_dir = DATA_ROOT / split_name / "images"  # Defines image output folder.
        mask_dir = DATA_ROOT / split_name / "masks"    # Defines mask output folder.
        image_dir.mkdir(parents=True, exist_ok=True)   # Creates image folder.
        mask_dir.mkdir(parents=True, exist_ok=True)    # Creates mask folder.
        manifest_path = DATA_ROOT / split_name / "labels.csv"  # Defines split manifest.
        with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)  # Creates CSV writer.
            writer.writerow(["image", "mask", "label", "source"])  # Writes column names.
            for index, source_path in enumerate(split_sources):  # Processes each authentic source.
                try:
                    original = read_medical_image(source_path)  # Loads the original medical image.
                    original = original.resize((IMAGE_SIZE, IMAGE_SIZE))  # Standardizes image size.
                    real_name = f"real_{index:06d}.png"  # Creates a unique authentic filename.
                    real_path = image_dir / real_name  # Defines authentic image destination.
                    zero_mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)  # Creates authentic mask.
                    zero_mask_path = mask_dir / real_name  # Defines authentic mask destination.
                    original.save(real_path)  # Saves authentic image.
                    zero_mask.save(zero_mask_path)  # Saves its all-zero mask.
                    writer.writerow([real_name, real_name, 0, str(source_path)])  # Records authentic sample.

                    forged, forged_mask = make_forgery(original)  # Generates a synthetic forgery.
                    fake_name = f"forged_{index:06d}.png"  # Creates a unique forged filename.
                    forged.save(image_dir / fake_name)  # Saves forged image.
                    forged_mask.save(mask_dir / fake_name)  # Saves pixel-level target mask.
                    writer.writerow([fake_name, fake_name, 1, str(source_path)])  # Records forged sample.
                except Exception as error:  # Handles unreadable or unsupported source files.
                    print(f"Skipped {source_path}: {error}")  # Reports the skipped file.

    print(f"Dataset preparation completed: {DATA_ROOT}")  # Reports generated dataset location.


# ------------------------------ DATASET ---------------------------------------
class ForgeryDataset(Dataset):
    """Loads image, binary mask, and image-level authenticity label from a split."""

    def __init__(self, split):
        self.folder = DATA_ROOT / split  # Stores the selected split directory.
        with (self.folder / "labels.csv").open("r", encoding="utf-8") as csv_file:
            self.rows = list(csv.DictReader(csv_file))  # Loads sample metadata from CSV.
        self.image_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),       # Converts image to [C,H,W] float tensor in [0,1].
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # Normalizes each RGB channel to [-1,1].
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),       # Converts binary mask to a one-channel tensor.
        ])

    def __len__(self):
        return len(self.rows)            # Returns number of samples in this split.

    def __getitem__(self, index):
        row = self.rows[index]           # Gets the requested CSV row.
        image = Image.open(self.folder / "images" / row["image"]).convert("RGB")  # Loads image.
        mask = Image.open(self.folder / "masks" / row["mask"]).convert("L")  # Loads mask.
        image = self.image_transform(image)  # Converts and normalizes image.
        mask = self.mask_transform(mask)  # Converts mask to tensor.
        label = torch.tensor(float(row["label"]), dtype=torch.float32)  # Converts label to float.
        return image, mask, label         # Returns model inputs and targets.


# ------------------------------ TRANSFORMER GAN ------------------------------
class TransformerGenerator(nn.Module):
    """Small transformer-based image-to-image generator for the GAN component."""

    def __init__(self, image_size=IMAGE_SIZE, patch_size=16, dim=256, depth=4):
        super().__init__()               # Initializes the parent neural-network module.
        self.patch_size = patch_size     # Stores patch side length.
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)  # Patch embedding.
        token_count = (image_size // patch_size) ** 2  # Computes number of image tokens.
        self.position = nn.Parameter(torch.zeros(1, token_count, dim))  # Learnable position embedding.
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True)  # Transformer block.
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)  # Stacks transformer blocks.
        self.to_pixels = nn.Sequential(
            nn.ConvTranspose2d(dim, 128, kernel_size=4, stride=4),  # Upsamples feature grid.
            nn.GELU(),                   # Adds nonlinear activation.
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Refines generated features.
            nn.GELU(),                   # Adds nonlinear activation.
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Predicts RGB residual.
            nn.Tanh(),                   # Bounds generated residual to [-1,1].
        )

    def forward(self, image):
        tokens = self.patch_embed(image).flatten(2).transpose(1, 2)  # Converts patches to tokens.
        tokens = tokens + self.position[:, :tokens.size(1)]  # Adds positional information.
        tokens = self.transformer(tokens)  # Models global relationships between image patches.
        side = int(tokens.size(1) ** 0.5)  # Recovers square feature-map side length.
        feature_map = tokens.transpose(1, 2).reshape(image.size(0), -1, side, side)  # Restores grid.
        residual = self.to_pixels(feature_map)  # Generates an image residual.
        residual = F.interpolate(residual, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return torch.clamp(image + 0.15 * residual, -1, 1)  # Returns a bounded generated image.


class TransformerDiscriminator(nn.Module):
    """Transformer discriminator that predicts whether an image is authentic or generated."""

    def __init__(self, patch_size=16, dim=192, depth=3):
        super().__init__()               # Initializes the parent module.
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)  # Patch tokens.
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=6, batch_first=True)  # Transformer layer.
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)  # Discriminator encoder.
        self.classifier = nn.Linear(dim, 1)  # Predicts real/fake logit.

    def forward(self, image):
        tokens = self.patch_embed(image).flatten(2).transpose(1, 2)  # Extracts patch tokens.
        tokens = self.transformer(tokens)  # Models relationships between patches.
        pooled = tokens.mean(dim=1)        # Computes global token average.
        return self.classifier(pooled).squeeze(1)  # Returns one real/fake logit per image.


# ------------------------------ SWIN + LOCALIZATION MODEL ---------------------
class SwinForgeryModel(nn.Module):
    """Swin feature extractor with a binary classifier and pixel-level mask decoder."""

    def __init__(self, pretrained=True):
        super().__init__()                # Initializes the parent module.
        if timm is None:                  # Checks whether timm is installed.
            raise ImportError("Install timm: pip install timm")
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )                                  # Loads hierarchical Swin feature maps.
        channels = self.backbone.feature_info.channels()  # Reads feature channel dimensions.
        self.fusion = nn.Sequential(
            nn.Conv2d(sum(channels), 256, kernel_size=1),  # Projects concatenated features.
            nn.BatchNorm2d(256),          # Stabilizes feature distributions.
            nn.GELU(),                    # Adds nonlinear activation.
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Refines fused features.
            nn.BatchNorm2d(128),          # Normalizes decoder activations.
            nn.GELU(),                    # Adds nonlinear activation.
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Reduces decoder channels.
            nn.GELU(),                    # Adds nonlinear activation.
            nn.Conv2d(64, 1, kernel_size=1),  # Predicts one forgery-mask logit per pixel.
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 256),  # Projects pooled deepest Swin features.
            nn.GELU(),                    # Adds nonlinear activation.
            nn.Dropout(0.3),              # Reduces overfitting.
            nn.Linear(256, 1),            # Predicts image-level forgery logit.
        )

    def forward(self, image):
        feature_maps = self.backbone(image)  # Extracts four hierarchical Swin feature maps.
        target_size = feature_maps[0].shape[-2:]  # Uses highest-resolution feature-map size.
        aligned = [
            F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            for feature in feature_maps
        ]                                  # Resizes feature maps to a common spatial resolution.
        fused = self.fusion(torch.cat(aligned, dim=1))  # Concatenates and fuses multi-scale features.
        mask_logits = self.decoder(fused)  # Predicts coarse localization logits.
        mask_logits = F.interpolate(mask_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        deepest = F.adaptive_avg_pool2d(feature_maps[-1], 1).flatten(1)  # Pools deepest features.
        class_logits = self.classifier(deepest).squeeze(1)  # Predicts image-level class.
        return class_logits, mask_logits  # Returns classification and localization predictions.


# ------------------------------ HYBRID LOSS -----------------------------------
def dice_loss(logits, targets, smooth=1.0):
    """Computes soft Dice loss for binary pixel-level segmentation."""
    probabilities = torch.sigmoid(logits)  # Converts logits to probabilities.
    probabilities = probabilities.flatten(1)  # Flattens each predicted mask.
    targets = targets.flatten(1)  # Flattens each ground-truth mask.
    intersection = (probabilities * targets).sum(dim=1)  # Computes overlap.
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)  # Computes total foreground mass.
    dice = (2 * intersection + smooth) / (denominator + smooth)  # Computes smoothed Dice score.
    return 1 - dice.mean()  # Returns average Dice loss.


def semantic_consistency_loss(class_logits, mask_logits):
    """Encourages image-level classification to agree with predicted foreground area."""
    image_probability = torch.sigmoid(class_logits)  # Computes image-level forgery probability.
    mask_probability = torch.sigmoid(mask_logits).mean(dim=(1, 2, 3))  # Computes average mask probability.
    return F.mse_loss(image_probability, mask_probability)  # Penalizes disagreement.


def hybrid_loss(class_logits, mask_logits, labels, masks):
    """Combines classification, pixel-level, and semantic-consistency objectives."""
    classification = F.binary_cross_entropy_with_logits(class_logits, labels)  # Binary classification.
    segmentation = F.binary_cross_entropy_with_logits(mask_logits, masks)  # Pixel-level BCE.
    dice = dice_loss(mask_logits, masks)  # Region-overlap loss.
    consistency = semantic_consistency_loss(class_logits, mask_logits)  # Cross-task consistency.
    total = classification + segmentation + dice + 0.2 * consistency  # Weighted total objective.
    return total, {
        "classification": classification.detach(),
        "segmentation_bce": segmentation.detach(),
        "dice": dice.detach(),
        "consistency": consistency.detach(),
    }                                  # Returns total loss and detached logging values.


# ------------------------------ TRAINING --------------------------------------
def train_one_epoch(model, loader, optimizer, scaler):
    """Runs one supervised epoch with automatic mixed precision when CUDA is available."""
    model.train()                        # Enables training behavior such as dropout.
    running_loss = 0.0                    # Initializes epoch loss accumulator.
    for images, masks, labels in loader:  # Iterates over mini-batches.
        images = images.to(DEVICE)        # Moves images to GPU or CPU.
        masks = masks.to(DEVICE)          # Moves pixel masks to the selected device.
        labels = labels.to(DEVICE)        # Moves image labels to the selected device.
        optimizer.zero_grad(set_to_none=True)  # Clears gradients efficiently.
        with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            class_logits, mask_logits = model(images)  # Runs forward pass.
            loss, _ = hybrid_loss(class_logits, mask_logits, labels, masks)  # Computes hybrid loss.
        scaler.scale(loss).backward()     # Computes scaled gradients for mixed precision.
        scaler.step(optimizer)            # Updates model parameters.
        scaler.update()                   # Updates gradient scaler state.
        running_loss += loss.item()       # Adds batch loss to epoch accumulator.
    return running_loss / max(len(loader), 1)  # Returns average epoch loss.


@torch.no_grad()
def evaluate(model, loader):
    """Evaluates classification and segmentation metrics on a validation/test split."""
    model.eval()                          # Disables dropout and training-specific behavior.
    correct = 0                           # Counts correct image classifications.
    total = 0                             # Counts all evaluated images.
    tp = fp = fn = 0                      # Initializes pixel-level confusion counts.
    for images, masks, labels in loader:  # Iterates over evaluation batches.
        images = images.to(DEVICE)        # Moves images to selected device.
        masks = masks.to(DEVICE)          # Moves masks to selected device.
        labels = labels.to(DEVICE)        # Moves labels to selected device.
        class_logits, mask_logits = model(images)  # Predicts classes and masks.
        predictions = (torch.sigmoid(class_logits) >= 0.5).float()  # Converts class probabilities to labels.
        correct += (predictions == labels).sum().item()  # Counts correct classifications.
        total += labels.numel()           # Updates number of evaluated images.
        predicted_masks = (torch.sigmoid(mask_logits) >= 0.5).float()  # Binarizes predicted masks.
        tp += (predicted_masks * masks).sum().item()  # Counts correctly predicted forged pixels.
        fp += (predicted_masks * (1 - masks)).sum().item()  # Counts false-positive pixels.
        fn += ((1 - predicted_masks) * masks).sum().item()  # Counts missed forged pixels.
    accuracy = correct / max(total, 1)    # Computes image-level accuracy.
    precision = tp / max(tp + fp, 1e-8)  # Computes pixel-level precision.
    recall = tp / max(tp + fn, 1e-8)     # Computes pixel-level recall.
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)  # Computes pixel-level F1.
    dice = 2 * tp / max(2 * tp + fp + fn, 1e-8)  # Computes global pixel-level Dice.
    iou = tp / max(tp + fp + fn, 1e-8)   # Computes global pixel-level IoU.
    return {
        "accuracy": accuracy,            # Image-level classification accuracy.
        "pixel_precision": precision,     # Pixel-level precision.
        "pixel_recall": recall,           # Pixel-level recall.
        "pixel_f1": f1,                   # Pixel-level F1 score.
        "dice": dice,                     # Pixel-level Dice coefficient.
        "iou": iou,                       # Pixel-level intersection over union.
    }


def train_model():
    """Creates data loaders, initializes Swin model, trains, validates, and saves weights."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)  # Creates checkpoint directory.
    train_data = ForgeryDataset("train")  # Loads training data.
    val_data = ForgeryDataset("val")      # Loads validation data.
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model = SwinForgeryModel(pretrained=True).to(DEVICE)  # Initializes pretrained Swin model.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # Optimizer.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)  # LR scheduler.
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))  # Mixed precision scaler.
    best_dice = -1.0                      # Tracks best validation Dice score.

    for epoch in range(EPOCHS):           # Loops over training epochs.
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)  # Trains one epoch.
        metrics = evaluate(model, val_loader)  # Evaluates validation split.
        scheduler.step()                  # Updates learning rate.
        print(f"Epoch {epoch + 1}/{EPOCHS} | loss={train_loss:.4f} | metrics={metrics}")  # Logs results.
        if metrics["dice"] > best_dice:   # Checks whether validation Dice improved.
            best_dice = metrics["dice"]   # Updates best score.
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_swin_forgery.pth")  # Saves best model.
        torch.save(model.state_dict(), CHECKPOINT_DIR / "last_swin_forgery.pth")  # Saves latest model.

    print(f"Training finished. Best validation Dice: {best_dice:.4f}")  # Reports best validation score.


# ------------------------------ INFERENCE -------------------------------------
@torch.no_grad()
def predict_image(image_path, checkpoint=None):
    """Predicts image-level authenticity and a pixel-level forgery probability map."""
    model = SwinForgeryModel(pretrained=False).to(DEVICE)  # Builds model architecture without downloading weights.
    checkpoint = checkpoint or (CHECKPOINT_DIR / "best_swin_forgery.pth")  # Selects checkpoint path.
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))  # Loads trained weights.
    model.eval()                          # Sets inference mode.
    image = read_medical_image(image_path).convert("RGB")  # Reads input image.
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resizes image to network input size.
        transforms.ToTensor(),          # Converts image to tensor.
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # Applies training normalization.
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # Adds batch dimension and moves to device.
    class_logits, mask_logits = model(tensor)  # Predicts classification and mask logits.
    probability = torch.sigmoid(class_logits).item()  # Converts class logit to probability.
    mask = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()  # Converts predicted mask to NumPy array.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Creates output directory.
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # Converts mask to grayscale image.
    mask_image.save(OUTPUT_DIR / "predicted_probability_map.png")  # Saves probability map.
    binary = Image.fromarray(((mask >= 0.5) * 255).astype(np.uint8))  # Creates binary predicted mask.
    binary.save(OUTPUT_DIR / "predicted_binary_mask.png")  # Saves binary mask.
    label = "FORGED" if probability >= 0.5 else "AUTHENTIC"  # Converts probability to class label.
    print(f"Prediction: {label} | forgery probability: {probability:.4f}")  # Prints image-level result.
    print(f"Saved masks in: {OUTPUT_DIR}")  # Reports output location.
    return label, probability, mask  # Returns predictions for further analysis.


# ------------------------------ MEDSAM REFINEMENT -----------------------------
def refine_with_medsam(image_path, coarse_mask_path, medsam_checkpoint, output_path=None):
    """
    Optional MedSAM inference adapter.

    Install and configure the official MedSAM repository first. The checkpoint and
    model API can vary by release; this function uses the official SamPredictor
    interface. It converts the coarse Swin mask into a bounding-box prompt.
    """
    try:
        from segment_anything import sam_model_registry, SamPredictor  # Official SAM predictor API.
    except ImportError as error:
        raise ImportError(
            "MedSAM is optional. Clone the official MedSAM repository and install its requirements first."
        ) from error

    image = np.array(read_medical_image(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    coarse = np.array(Image.open(coarse_mask_path).convert("L"))  # Reads Swin probability map.
    ys, xs = np.where(coarse >= 128)       # Finds coarse foreground coordinates.
    if len(xs) == 0:                       # Handles empty coarse prediction.
        raise ValueError("Coarse mask is empty; MedSAM box prompt cannot be generated.")
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])  # Creates bounding-box prompt.
    sam = sam_model_registry["vit_b"](checkpoint=str(medsam_checkpoint))  # Loads MedSAM ViT-B checkpoint.
    sam.to(device=DEVICE)                  # Moves MedSAM to GPU or CPU.
    predictor = SamPredictor(sam)          # Creates interactive segmentation predictor.
    predictor.set_image(image)             # Encodes input image.
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)  # Predicts prompt-conditioned masks.
    best = masks[int(np.argmax(scores))]   # Selects the mask with highest predicted score.
    output_path = Path(output_path or OUTPUT_DIR / "medsam_refined_mask.png")  # Sets output path.
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Creates output folder.
    Image.fromarray((best * 255).astype(np.uint8)).save(output_path)  # Saves refined binary mask.
    print(f"MedSAM-refined mask saved to: {output_path}")  # Reports saved mask.
    return best                            # Returns refined mask array.


# ------------------------------ COMMAND-LINE ENTRY ----------------------------
def main():
    """Parses command and runs dataset preparation, training, or inference."""
    parser = argparse.ArgumentParser(description="Medical image forgery detection and localization")
    parser.add_argument("--prepare", action="store_true", help="Create synthetic train/val/test data")
    parser.add_argument("--train", action="store_true", help="Train Swin classification/localization model")
    parser.add_argument("--predict", type=str, default=None, help="Predict one image path")
    args = parser.parse_args()              # Reads command-line arguments.
    if args.prepare:                        # Checks dataset-preparation request.
        prepare_dataset()                   # Builds the paired synthetic dataset.
    if args.train:                           # Checks training request.
        train_model()                       # Starts model training.
    if args.predict:                         # Checks inference request.
        predict_image(args.predict)          # Predicts the requested image.


if __name__ == "__main__":                  # Runs only when called as a script.
    main()                                   # Starts the selected operation.
