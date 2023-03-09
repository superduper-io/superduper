import random


def captioning_splitter(r):
    index = random.randrange(len(r['captions']))
    target = {}
    target['caption'] = r['captions'][index]
    r['caption'] = r['captions'][index]
    return r, target


def retrieval_splitter(r):
    return {'img': r['img']}, {'captions': [r['captions'][0]]}

