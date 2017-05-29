#-*- coding: utf-8 -*-
import os
import flickrapi
import json
import urllib.request

api_key = '444fe6a8f8c758d2ce5df259af88514c'
api_secret = '695f43d9acf4fda7'
photoset_id = '72157629082469698'
flickr = flickrapi.FlickrAPI(api_key, api_secret,)

def scrapPhotoset(photoset_id):
    doc = flickr.photosets_getPhotos(
               photoset_id=photoset_id,
               extras='original_format',
               format='json',
               nojsoncallback="1",
              )
    photoset = json.loads(doc.decode())
    count = 0
    directory = './images/'+photoset_id
    if not os.path.isdir(directory): os.mkdir(directory)
    for photo in photoset['photoset']['photo']:
        url = "http://farm%s.static.flickr.com/%s/%s_%s_o.jpg" % (photo['farm'], photo['server'], photo['id'], photo['originalsecret'])
        fileName = './images/'+photoset_id+'/'+str(count)+'.jpg'
        
        image_from = urllib.request.urlopen(url)
        image_to = open(fileName, 'wb')
        image_to.write(image_from.read())
        image_from.close()
        image_to.close()
        
        print (fileName)
        count += 1

"""
여기부터 photoset_id를 자동으로 탐색
"""
def getMembers(group_id):
    print(1)
    
def getGroups():
    flickr.flickr_group.members.getlist()

def login():
    #reauest token 받아오기
    flickr.flickr_oauth.get_request_token()
    flickr.flickr_oauth.auth_via_console()
    flickr.fl