from app import process_image
import os

if __name__ == '__main__':
    img='dataset/processed/images/2007_000032.jpg'
    if not os.path.exists(img):
        print('image not found')
    else:
        result=process_image(img,'awgn',10.0,0.8,'test123')
        print('done', result['metrics'])
