import random


def captioning_splitter(r):
    index = random.randrange(len(r['captions']))
    caption = '<s> ' + r['captions'][index].strip()
    target = {}
    target['caption'] = caption
    return r, target

