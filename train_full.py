import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm # 기획자를 위한 예쁜 진행률 막대 도구!

# 1. 제한 해제! 40,000장 전체를 긁어모으는 데이터셋 로직
class KFoodDatasetFull(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.img_paths = []
        self.labels = []
        
        sub_folders = glob.glob(os.path.join(root_dir, "*", "*"))
        self.classes = sorted([os.path.basename(f) for f in sub_folders])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print("🔍 4만 장의 데이터를 스캔하고 있습니다. 잠시만 기다려주세요...")
        for folder in sub_folders:
            cls_name = os.path.basename(folder)
            images = glob.glob(os.path.join(folder, "*.jpg")) + \
                     glob.glob(os.path.join(folder, "*.png")) + \
                     glob.glob(os.path.join(folder, "*.jpeg"))
            
            # 100장 제한(limit) 삭제! 전체 이미지를 모두 담습니다.
            for img in images:
                self.img_paths.append(img)
                self.labels.append(class_to_idx[cls_name])
                
        print(f"✅ 스캔 완료! 총 {len(self.classes)}개 클래스, {len(self.img_paths)}장의 이미지를 장전했습니다.")

    def __len__(self): 
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img, self.labels[idx]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    DATA_DIR = "C:/food/kfood"
    
    dataset = KFoodDatasetFull(root_dir=DATA_DIR, transform=transform)
    
    # 램 32GB와 RTX 4060의 힘을 믿고 한 번에 64장씩(Batch Size) 고속 처리! 
    # (일반 노트북은 16장도 버겁지만 우리는 거뜬합니다)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 40)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 에포크를 10번으로 설정 (충분히 똑똑해지는 횟수)
    epochs = 10 
    print(f"\n🚀 [최종 학습 시작] 그래픽카드({device}) 풀가동을 시작합니다!")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # tqdm을 씌워서 진행 상황을 % 막대로 예쁘게 보여줍니다.
        progress_bar = tqdm(dataloader, desc=f"[에포크 {epoch+1}/{epochs}]", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'오답률(Loss)': f"{loss.item():.4f}"})
            
        print(f"🏁 에포크 {epoch+1} 완료! (평균 오답률: {running_loss/len(dataloader):.4f})")
        
                
    print("\n🎉 드디어 4만 장의 지옥 훈련이 모두 끝났습니다! AI가 한식 마스터가 되었습니다.")
    
    # 최종 버전을 새로운 이름으로 안전하게 저장
    torch.save(model.state_dict(), 'kfood_model_full.pth')
    print("💾 최종 AI 뇌(kfood_model_full.pth) 저장 완료! 이제 편히 쉬십시오.")