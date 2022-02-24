from io import BytesIO
from flask import jsonify, Blueprint, request
import cv2
from libs.utils import *
from libs.imageProcessing import *
from libs.drawLine import *
from libs.painting import *
from PIL import Image
from uuid import uuid4
from s3_connect import s3_connection
from aws_config import AWS_S3_BUCKET_NAME


from multiprocessing import Process, Manager

views = Blueprint("server", __name__)

def reduce_color_process(idx, paintingTool, img, cluster, result, colorNames, colors):
    idx = str(idx)


    print(f'{idx}번 프로세스 컬러 군집화 시작')
    clusteredImage = paintingTool.colorClustering( img, cluster = cluster )
    
    print(f'{idx}번 프로세스 이미지 확장')
    expandedImage = imageExpand(clusteredImage, guessSize = False, size=3)
    
    print(f'{idx}번 프로세스 컬러 매칭 시작')
    paintingMap = paintingTool.expandImageColorMatch(expandedImage)
    

    print(f'{idx}번 프로세스 컬러 추출 시작')
    colorNames_, colors_ = getColorFromImage(paintingMap)

    print(f'{idx}번 프로세스 컬러 {len(colorNames_)}개')

    colorNames[idx] = colorNames_
    colors[idx] = colors_

    result.put(paintingMap)
    return


def s3_upload(result_img, s3):
    file_name = uuid4().hex
    image = result_img.astype('uint8')
    image = Image.fromarray(image)
    buffer = BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)
    s3.upload_fileobj(buffer, AWS_S3_BUCKET_NAME, f"result_image/{file_name}.png", ExtraArgs={'ACL':'public-read'})
    print('업로드 완료')
    return str(file_name)

colorNames = {}
colors = {}
img_lab = None
lab = None
img_list = []

@views.route("/pipo", methods=["POST"])
def convert():
    global colorNames
    global colors

    global img_lab
    global lab

    global img_list

    url = request.json['url']
    paintingTool = Painting(url)
    img_list = []

    print(f'블러 시작')

    # 색 단순화 + 블러 처리
    blurImage = paintingTool.blurring(
        div = 8, 
        radius = 10, 
        sigmaColor =20, 
        medianValue=3
    )  

    clusters = [32]
    manager = Manager()
    result_list = manager.Queue()
    colorNames_ = manager.dict()
    colors_ = manager.dict()
    processes = []

    for idx, cluster in enumerate(clusters):
        process = Process(target=reduce_color_process, args=(idx+1, paintingTool, blurImage, cluster, result_list, colorNames_, colors_))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    painting_map_1 = result_list.get()
    img_list.append(painting_map_1)
    colorNames = dict(colorNames_)
    colors = dict(colors_)
    paintingMap = img_list[0]


    # 선 그리기
    print(f'선 그리기 시작')
    drawLineTool = DrawLine(paintingMap)
    lined_image = drawLineTool.getDrawLine()
    lined_image = drawLineTool.drawOutline(lined_image)

    # 레이블 추출
    img_lab, lab = getImgLabelFromImage(colors['1'], paintingMap)
    lined_image = cv2.convertScaleAbs(lined_image)

    # contour, hierarchy 추출
    print(f'컨투어 추출 시작')
    contours, hierarchy, thresh = getContoursFromImage(lined_image.copy())


    # 결과 이미지 백지화
    result_img = makeWhiteFromImage(paintingMap)
    result_img = setBackgroundAlpha(paintingMap, result_img)

    # 결과이미지 렌더링
    # image를 넣으면 원본이미지에 그려주고, result_img에 넣으면 백지에 그려줌
    print(f'넘버링 시작')
    result_img = setColorNumberFromContours(result_img, thresh, contours, hierarchy, img_lab, lab, colorNames['1'])

    print(f'컬러 레이블링 시작')
    result_img2, hex_list = setColorLabel(result_img.copy(), colorNames['1'], colors['1'])

    print(f'작업 완료')

    s3 = s3_connection()
    img = s3_upload(result_img, s3)
    label_img = s3_upload(result_img2, s3)

    return jsonify(img=img, label_img=label_img, hex_list=hex_list)
