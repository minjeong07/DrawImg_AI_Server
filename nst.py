from uuid import uuid4
from matplotlib.cbook import maxdict
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from s3_connect import s3_connection
from aws_config import AWS_S3_BUCKET_NAME


def load_style(style_path, max_dim):

    img = tf.io.read_file(style_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # 전체 이미지의 비율을 유지하면서, 원하는 크기로 변환
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def upload_tensor_img(tensor):
    tensor = np.array(tensor*255 ,dtype=np.uint8)
    image = Image.fromarray(tensor[0])
    print(type(image))
    buffer = BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)
    s3 = s3_connection()
    file_name = uuid4().hex
    # s3.upload_fileobj(buffer, AWS_S3_BUCKET_NAME, f"style_trans_image/{file_name}", ExtraArgs={'ACL':'public-read'})
    s3.put_object(
        Bucket = AWS_S3_BUCKET_NAME,
        Body = buffer,
        Key = f"style_trans_image/{file_name}",
        ACL = 'public-read'
    )
    print('업로드 완료')
    return file_name

def nst_apply(url, hub_module) :
    # style_path = tf.keras.utils.get_file('testing.jpg',
    #                                      'https://images.velog.io/images/aopd48/post/ba2a6e23-28e2-45aa-a1fc-1d43cf21da45/image.png') # 고흐
    # style_path = tf.keras.utils.get_file('mone.jpg',
    #                                      'https://images.velog.io/images/aopd48/post/e7ed2e7c-0be6-4bf8-8428-a7d107539a2f/image.png') # 고흐
    style_path = tf.keras.utils.get_file('w22a2ve.jpg',
                                        'https://images.velog.io/images/aopd48/post/46607f41-6e83-430f-bfe6-ebcb6ad35721/image.png') # 고흐

    style_image = load_style(style_path, 512)

    res = requests.get(url)
    img = Image.open(BytesIO(res.content))

    content_image = tf.keras.preprocessing.image.img_to_array(img)    
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    content_image = tf.image.resize(content_image, (512, 512))
    
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    image_url = upload_tensor_img(stylized_image)
    return image_url