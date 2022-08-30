from sddb.client import SddbClient

c = SddbClient()
docs = c.coco.documents
docs.single_thread = False
result = docs.find_one({
    'captions': {
        '$like': {
            'document': ['many people eating birthday cake'],
            'n': 10
        }
    }
})