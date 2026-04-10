import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# 1. 기획 단계의 핵심: 2단 폴더 구조에서 100장만 쏙 빼오는 커스텀 데이터셋 로직
class KFoodDataset(Dataset):
    def __init__(self, root_dir, limit=100, transform=None):
        self.transform = transform
        self.img_paths = []
        self.labels = []
        
        # kfood 폴더 안의 '대분류 -> 소분류' 폴더 경로를 모두 찾음
        sub_folders = glob.glob(os.path.join(root_dir, "*", "*"))
        
        # 40개 음식 이름(소분류 폴더명)을 가나다순으로 정렬하여 클래스 생성
        self.classes = sorted([os.path.basename(f) for f in sub_folders])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for folder in sub_folders:
            cls_name = os.path.basename(folder)
            # jpg, png, jpeg 이미지 모두 찾기
            images = glob.glob(os.path.join(folder, "*.jpg")) + \
                     glob.glob(os.path.join(folder, "*.png")) + \
                     glob.glob(os.path.join(folder, "*.jpeg"))
            
            # 여기서 딱 100장만 자릅니다! (limit=100)
            images = images[:limit] 
            
            for img in images:
                self.img_paths.append(img)
                self.labels.append(class_to_idx[cls_name])
                
        print(f"✅ 총 {len(self.classes)}개 음식 클래스, 총 {len(self.img_paths)}장의 이미지를 성공적으로 불러왔습니다.")

    def __len__(self): 
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # 이미지를 열고 AI가 인식할 수 있게 RGB로 변환
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img, self.labels[idx]


# 2. 이미지 크기 맞추기 및 텐서 변환 (AI가 먹기 좋게 가공)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 3. 메인 실행 블록
if __name__ == '__main__':
    # 내 PC의 데이터 경로 설정
    DATA_DIR = "C:/food/kfood"
    
    # 데이터셋 및 데이터로더 생성 (한 번에 32장씩 가볍게 처리)
    dataset = KFoodDataset(root_dir=DATA_DIR, limit=100, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 가장 가볍고 빠른 ResNet18 모델 불러오기
    model = models.resnet18(pretrained=True)
    
    # 마지막 출력층을 우리 목적에 맞게 40개로 수정
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 40)
    
    # 오류 수정 기준(Loss)과 최적화 방법(Optimizer) 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # CPU 또는 GPU 설정 (노트북 사양에 맞춰 자동 할당)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"🚀 학습을 시작합니다! (사용 장비: {device})")
    
    # PoC 테스트이므로 딱 3번(Epochs=3)만 가볍게 반복 학습
    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 진행 상황 출력
            if (i+1) % 10 == 0:
                print(f"[에포크 {epoch+1}/{epochs}] {i+1}번째 배치 완료... (Loss: {loss.item():.4f})")
                
    # (기존 코드의 맨 마지막 줄)
    print("🎉 100장씩 맛보기 학습(PoC)이 완벽하게 끝났습니다!")
    
    # AI의 기억(가중치)을 파일로 저장하는 마법의 한 줄
    torch.save(model.state_dict(), 'kfood_model_poc.pth')