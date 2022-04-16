from io import BytesIO
from uuid import uuid4
import numpy as np
from spade.model import Pix2PixModel
from spade.dataset import get_transform
from torchvision.transforms import ToPILImage
from PIL import Image
from s3_connect import s3_connection
from aws_config import AWS_S3_BUCKET_NAME
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.path.join(BASE_DIR, 'pretrained')



def evaluate(labelmap):
    opt = {
        'label_nc': 182, # num classes in coco model
        'crop_size': 512,
        'load_size': 512,
        'aspect_ratio': 1.0,
        'isTrain': False,
        'checkpoints_dir': MODEL_DIR,
        'which_epoch': 'latest',
        'use_gpu': False
    }

    model = Pix2PixModel(opt)
    model.eval()

    image = Image.fromarray(np.array(labelmap).astype(np.uint8))

    transform_label = get_transform(opt, method=Image.NEAREST, normalize=False)
    # transforms.ToTensor in transform_label rescales image from [0,255] to [0.0,1.0]
    # lets rescale it back to [0,255] to match our label ids
    label_tensor = transform_label(image) * 255.0
    label_tensor[label_tensor == 255] = opt['label_nc'] # 'unknown' is opt.label_nc
    print("label_tensor:", label_tensor.shape)

    # not using encoder, so creating a blank image...
    transform_image = get_transform(opt)
    image_tensor = transform_image(Image.new('RGB', (500, 500)))

    data = {
        'label': label_tensor.unsqueeze(0),
        'instance': label_tensor.unsqueeze(0),
        'image': image_tensor.unsqueeze(0)
    }
    generated = model(data, mode='inference')
    print("generated_image:", generated.shape)
    s3 = s3_connection()
    file_name = uuid4().hex
    image = to_image(generated)
    buffer = BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)
    s3.upload_fileobj(buffer, AWS_S3_BUCKET_NAME, f"gau/{file_name}.png", ExtraArgs={'ACL':'public-read'})
    print('업로드 완료')
    url = f'https://{AWS_S3_BUCKET_NAME}.s3.ap-northeast-2.amazonaws.com/gau/{file_name}.png'
    return url

def to_image(generated):
    to_img = ToPILImage()
    normalized_img = ((generated.reshape([3, 512, 512]) + 1) / 2.0) * 255.0
    return to_img(normalized_img.byte().cpu())

