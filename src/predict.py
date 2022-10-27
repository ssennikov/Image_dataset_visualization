import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import shutil


def move_predictions(model, train_dir: str, test_dir: str, dataset, device):
    for path in Path(test_dir).glob('*.png'):
        img = Image.open(path)
        transform_norm = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img_normalized = transform_norm(img).float()
        img_normalized = img_normalized.unsqueeze_(0)
        img_normalized = img_normalized.to(device)
        with torch.no_grad():
            model.eval()
            output = model(img_normalized)
            index = output.data.cpu().numpy().argmax()
            classes = dataset.classes
            label = classes[index]

        shutil.move(str(path), train_dir / Path(label))
