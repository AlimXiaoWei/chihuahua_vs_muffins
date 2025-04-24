import yaml, os, torch, time
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from pathlib import Path

with open('../config/config.yaml') as f:
    cfg = yaml.safe_load(f)

IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4


train_dir = Path(cfg['data']['path']) / 'train'
test_dir  = Path(cfg['data']['path'])  / 'test'


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])


test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

full_train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
test_ds       = datasets.ImageFolder(test_dir,  transform=test_tf)

train_len = int(0.8*len(full_train_ds))
val_len   = len(full_train_ds) - train_len
train_ds, val_ds = random_split(full_train_ds, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

history = {'train_loss':[], 'val_loss':[]}

for epoch in range(1, EPOCHS+1):
    model.train(); running = 0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x); loss = criterion(out, y)
        loss.backward(); optimizer.step()
        running += loss.item()*x.size(0)
    train_loss = running/len(train_loader.dataset)


    model.eval(); running = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x); loss = criterion(out, y)
            running += loss.item()*x.size(0)
    val_loss = running/len(val_loader.dataset)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    print(f"Epoch {epoch}/{EPOCHS} — train {train_loss:.4f} · val {val_loss:.4f}")

torch.save(model.state_dict(), '../assets/muffin_vs_chihuahua.pth')