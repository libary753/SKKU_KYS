import requests
from bs4 import BeautifulSoup
from datetime import datetime

file_path = datetime.strftime(datetime.now(),"./scrap_%Y-%m-%d-%H_%M_%S.txt")
print(file_path)
f = open(file_path,'w',encoding='utf-8')
url = "http://okky.kr/articles/questions" # url 설정
response = requests.get(url) # 소스파일 받아옴

soup = BeautifulSoup(response.text,'html.parser') # bs 생성
lists = soup.select('li.list-group-item') #soup에서 item을 가져
data = ''
for li in lists:
    a = li.find('h5').find('a') #<h5> <a> ~~~~~~~ </a> </h5> 를 찾음
    link = a['href'] #href 값을 받아와서 link에 저장 /artice/389983
    title = a.text # 게시글 내용
    list_id = link.split('/')[-1] # 게시글 ID 389983
    print(link, list_id,title)
    data += 'link : %s, id : %s, title : %s\n' %(link,list_id,title)
f.write(data)
f.close()