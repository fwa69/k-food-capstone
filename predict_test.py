import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# 1. 정답지 세팅
DATA_DIR = "C:/food/kfood"
sub_folders = glob.glob(os.path.join(DATA_DIR, "*", "*"))
classes = sorted([os.path.basename(f) for f in sub_folders])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델 불러오기 (★ 이제 'full' 버전을 불러옵니다!)
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 40)

# ★ 4만 장을 학습한 최종 모델 파일명으로 변경!
model.load_state_dict(torch.load('kfood_model_full.pth', weights_only=True)) 
model = model.to(device)
model.eval()

# 3. 이미지 가공
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. 예측 함수 (기획자님의 50% Fallback 로직 적용 완료)
def predict_image(image_path, threshold=50.0):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probabilities = F.softmax(outputs, dim=1)[0] * 100 
        max_prob, predicted = torch.max(probabilities, 0) 
        
        confidence = max_prob.item()
        result_class = classes[predicted.item()]

        if confidence < threshold:
            return False, confidence, result_class
        else:
            return True, confidence, result_class

# =====================================================================

if __name__ == '__main__':
    # 테스트할 이미지 경로 (어제 틀렸던 짜장면이나 칼국수 사진을 넣어보세요!)
    test_image_path = "C:/food/test.jpg" 
    
    try:
        print("🤔 4만 장을 마스터한 AI가 사진을 분석하고 있습니다...")
        is_success, confidence, result = predict_image(test_image_path, threshold=50.0)
        
        if is_success:
            print(f"🎉 AI의 예측: 이 음식은 [ {result} ] 입니다! (확신도: {confidence:.1f}%)")
        else:
            print(f"⚠️ 앗! 이 이미지는 우리 음식 리스트에 없거나 AI가 너무 헷갈려 합니다.")
            print(f"   (AI의 변명: {result} 같긴 한데... 확신도가 {confidence:.1f}% 밖에 안 돼서 포기할게요 ㅠㅠ)")
            
    except FileNotFoundError:
        print(f"❌ 에러: {test_image_path} 사진을 찾을 수 없습니다.")