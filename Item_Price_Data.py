import pandas as pd
import requests
import json
from lostark_api_token import Token

headers1 = {
    'accept': 'application/json',
    'authorization': Token,
    'Content-Type': 'application/json',
}

headers2 = {
    'accept': 'application/json',
    'authorization': Token
}

json_data = {
    'Sort': 'RecentPrice',
    'CategoryCode': 50010,
    'CharacterClass': None,
    'ItemTier': None,
    'ItemGrade': None,
    'ItemName': '명예의 파편 주머니(대)',
    'PageNo': 1,
    'SortCondition': 'DSC',
}

#url1= 'https://developer-lostark.game.onstove.com/markets/items'
url2= 'https://developer-lostark.game.onstove.com/markets/items/66130133'
#response1 = requests.post(url1, headers=headers1, json = json_data)
response2 = requests.get(url2, headers=headers2)
# headers의 정보를 통해 접근(로그인) 하여 url의 Data를 받아옴.
# 다른 아이템의 정보를 가져오고 싶다면 코드를 알아내서 url2의 마지막 숫자 수정

#item_list_data = response1.json()
item_data = response2.json()
data = pd.read_csv('./loa_chart.csv') #차트를 로드
data_date_list = []
data_price_list = []

#로드한 데이터를 리스트에 추가함.
for i in range(len(data['Date'])):
    data_date_list.append(data.loc[i,'Date'])
    data_price_list.append(data.loc[i,'item1'])

#API로부터 받아온 정보를 리스트에 추가함.
recent_date_list = []
recent_price_list = []
for i in range(14):
    recent_date_list.append(item_data[0]['Stats'][i]['Date'])
    recent_price_list.append(item_data[0]['Stats'][i]['AvgPrice'])

#로아 API로부터 받아온 정보는 최신 -> 이전 순서라 역순 정렬이 필요함.
recent_date_list.reverse()
recent_price_list.reverse()

#로드한 날짜 데이터의 마지막 날짜와 API로부터 받아온 날짜를 비교함.
#비교 후 최신화 하는 과정.
last_date_ddl = data_date_list[-1]
first_date_rdl = recent_date_list[0]

if last_date_ddl < first_date_rdl:
    data_date_list.extend(recent_date_list)
    data_price_list.extend(recent_price_list)
else:
    overlap_index = None
    for i in range(len(data_date_list)):
        if data_date_list[i] == first_date_rdl:
            overlap_index = i
            break
    if overlap_index is not None:
        data_date_list = data_date_list[:overlap_index] + recent_date_list
        data_price_list = data_price_list[:overlap_index] + recent_price_list

#최신화된 정보를 기록후 파일로 저장.
new_data = pd.DataFrame({'Date': data_date_list,
                         'item1': data_price_list})

new_data.to_csv("loa_chart.csv", index=False)
print("Data Updated")


