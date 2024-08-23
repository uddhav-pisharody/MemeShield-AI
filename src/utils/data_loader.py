from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
directory2 = 'Enter directory here'
# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_dict, text_dict, tokenizer, max_length, labels, transform=None):
        self.image_dict = image_dict
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.labels = labels
        self.image_ids = list(image_dict.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.image_dict[image_id]
        image = Image.open(directory2 + img_path).convert('RGB')
        text = self.text_dict[image_id]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        if self.transform:
            image = self.transform(image)
        return image, input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

images_dict = {} # your image dict
text_dict = {} # your text dict
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
labels = [] # your labels

def get_data_loaders(images_dict, text_dict, tokenizer, max_length=128, labels=labels, transform=transform):
    dataset = CustomDataset(images_dict, text_dict, tokenizer, max_length=128, labels=labels, transform=transform)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, pin_memory=True,shuffle=False)