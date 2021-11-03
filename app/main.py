# 範例二 網頁執行預測
## 檔案路徑
import os
## 檔案上傳 Library
import shutil

## FASTAPI Library
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, Form
## FASTAPI Library 網頁框架 
from fastapi.templating import Jinja2Templates
## FASTAPI Library 外掛資料夾
from fastapi.staticfiles import StaticFiles

## FASTAI Library
from fastai.vision.all import *

## 服務器
#import nest_asyncio
#from pyngrok import ngrok
#import uvicorn


# 載入 FASTAPI
app = FastAPI()

# 網頁框架
templates = Jinja2Templates(directory='templates/')

# 掛載資料夾
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 模型檔案
myPath='models'
myModel=myPath+'/nodule.pkl'
learn = load_learner(myModel)

# 檔案上傳 WEB
@app.get('/nodule')
def form_get(request: Request):
    result = ''
    return templates.TemplateResponse('uploadFile.html', context={'request': request, 'result': result})

# 檔案預測 WEB
@app.post('/nodule')
def form_post(request: Request, file: UploadFile = File(...)):
    upload_image = 'uploads/'+file.filename
    html_upload_image = '/uploads/'+file.filename

    with open(upload_image, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = learn.predict(upload_image) 
    result = '圖片預測結果: ' + prediction[0]
    return templates.TemplateResponse('uploadFile.html', context={'request': request, 'result': result, 'upload_image': html_upload_image})
  
