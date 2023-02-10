import random


def captioning_splitter(r):
    index = random.randrange(len(r['captions']))
    target = {}
    target['caption'] = r['captions'][index]
    r['caption'] = r['captions'][index]
    return r, target

