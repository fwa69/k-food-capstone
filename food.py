import urllib.request
import urllib.parse
import json
import os
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. 네이버 API 엔진 (K-Food 철벽 방어막 유지)
# ---------------------------------------------------------
def get_naver_restaurants(query, food_name):
    # 코드에 직접 적지 않고, 금고에서 꺼내오기!
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    encText = urllib.parse.quote(query)
    # 넉넉하게 15개를 가져와서 한식 아닌 것들을 전부 쳐냅니다.
    url = f"https://openapi.naver.com/v1/search/local.json?query={encText}&display=15&sort=comment"
    
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    
    try:
        response = urllib.request.urlopen(request)
        if response.getcode() == 200:
            data = json.loads(response.read().decode('utf-8'))
            
            filtered_items = []
            for item in data['items']:
                category = item['category']
                
                # ★ K-Food 무적의 철벽 방어막 ★
                if "이탈리아" in category or "양식" in category or "카페" in category or "돈가스" in category or "주점" in category or "맥주" in category or "치킨" in category or "피자" in category or "베이커리" in category:
                    continue
                        
                filtered_items.append(item)
                
                if len(filtered_items) == 5:
                    break
                    
            return filtered_items
            
    except Exception as e:
        print(f"❌ API 에러: {e}")
    return []

# ---------------------------------------------------------
# 2. ★업그레이드★ '스마트 키워드 조립기' (VOC 데이터 반영)
# ---------------------------------------------------------
# situation 파라미터 추가 ("혼밥", "모임", None 중 선택)
def make_smart_query(location, food_name, situation=None):
    # [기본] 음식 종류별 꼬리표 달기
    if food_name in ["삼겹살", "목살"] or food_name.endswith("구이"):
        suffix = "고깃집"
    elif food_name[-1] in ["탕", "찜", "국", "밥", "전"] or food_name.endswith("볶음") or food_name.endswith("찌개") or food_name.endswith("김치"):
        suffix = "한식 맛집"
    elif food_name.endswith("면") or food_name.endswith("국수"):
        suffix = "전문점"
    else:
        suffix = "한식 맛집"
        
    # ★ [핵심] 타겟 유저 인터뷰(VOC) 기반 키워드 동적 추가 ★
    if situation == "혼밥":
        # 외삼촌 니즈 반영: 혼자 먹기 좋은 곳
        suffix += " 혼밥" 
    elif situation == "모임":
        # 친구 니즈 반영: 전화로 물어볼 필요 없게 아예 '단체석' 있는 곳을 긁어옴
        suffix += " 단체석" 
        
    return f"{location} {food_name} {suffix}"

# ---------------------------------------------------------
# 3. 스마트 폴백 추천 로직 (situation 파라미터 전달 추가)
# ---------------------------------------------------------
def smart_recommend(address, food_name, situation=None):
    parts = address.split()
    if len(parts) >= 3:
        broad_loc = parts[1] 
        narrow_loc = f"{parts[1]} {parts[2]}" 
    else:
        broad_loc = parts[0]
        narrow_loc = address

    # 출력 화면을 예쁘게 꾸며줍니다.
    sit_text = f" [{situation} 맞춤]" if situation else ""
    print("\n" + "="*60)
    print(f"🔎 검색 조건: {address} | {food_name}{sit_text}")
    print("="*60)
    
    # 1단계 검색
    query_1 = make_smart_query(narrow_loc, food_name, situation)
    print(f"🤖 [1단계] 도보권 찐맛집 검색 👉 '{query_1}'")
    
    results = get_naver_restaurants(query_1, food_name)
    
    # 결과가 부족하면 2단계 폴백 발동
    if len(results) < 3:
        print("\n⚠️ 앗! 동네에 식당이 부족합니다. [스마트 폴백] 가동!")
        query_2 = make_smart_query(broad_loc, food_name, situation)
        print(f"🤖 [2단계] 지역구 단위 검색 👉 '{query_2}'")
        results = get_naver_restaurants(query_2, food_name)
    else:
        print("\n✅ 동네에 맛집이 충분히 많습니다! (폴백 미가동)")
        
    print("-" * 60)
    if not results:
        print("😭 조건에 맞는 식당을 찾지 못했습니다.")
    else:
        for i, item in enumerate(results):
            title = item['title'].replace('<b>', '').replace('</b>', '')
            addr = item['roadAddress'] if item['roadAddress'] else item['address']
            category_name = item['category'].split('>')[-1] 
            print(f"{i+1}. 🍽️ {title} (📍 {addr}) [분류: {category_name}]")
    print("="*60 + "\n")

# =====================================================================
# 🚀 인터뷰 결과가 반영된 최종 테스트 블록
# =====================================================================
if __name__ == '__main__':
    # 📸 보고서 캡처용 1: 외삼촌의 "혼밥, 도보권" 니즈 완벽 구현!
    print("▶️ 테스트 1: 점심시간, 동네에서 제육볶음 혼밥하기")
    smart_recommend("서울 강남구 역삼동", "제육볶음", situation="혼밥")
    
    # 📸 보고서 캡처용 2: 친구의 "단체 예약 가능 여부 확인" 니즈 완벽 구현!
    print("▶️ 테스트 2: 팀 회식, 단체석이 있는 고깃집 찾기")
    smart_recommend("전북 군산시 수송동", "삼겹살", situation="모임")
    
    # (참고) 옵션 없이 그냥 메뉴만 검색할 때 (기본 기능 유지)
    # smart_recommend("전북 군산시 수송동", "김치찌개") 