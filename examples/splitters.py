import random


def captioning_splitter(r):
    index = random.randrange(len(r['captions']))
    caption = r['captions'][index].strip()
    target = {}
    target['caption'] = caption
    r['caption'] = '<s> ' + caption
    return r, target

