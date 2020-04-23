import wget

def download_imagenet_classes():
    res = wget.download('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
    print(res)
    #, 'imagenet_class_index.json')

download_imagenet_classes()