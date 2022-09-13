import bs4


class Dummy:
    def preprocess(self, r):
        return {
            '_content': {
                'url': 'https://cdn.pixabay.com/photo/2013/07/12/17/47/'
                       'test-pattern-152459_1280.png',
                'converter': 'sddb.models.converters.PILImage'
            }
        }

    def eval(self):
        pass


class Parser:
    def preprocess(self, page):
        soup = bs4.BeautifulSoup(page)
        mynext = lambda x: list(x)[0]
        try:
            title = mynext(soup.findAll('h1', attrs={'class': 'product-title'})).text
        except IndexError:
            title = ''
        try:
            link = mynext(soup.findAll('a', attrs={'class': 'vi-image-gallery__enlarge-link'}))
            img = mynext(link.findAll('img'))['src']
            img = {'_content': {
                'url': img,
                'converter': 'sddb.models.converters.PILImage'
            }}
        except IndexError:
            img = {}
        try:
            snippet = mynext(soup.findAll('div', {'class': 'item-snippet lines-3 short'})).text
        except IndexError:
            snippet = ''
        try:
            price = mynext(soup.findAll('span', {'class': 'item-price'})).text
        except IndexError:
            price = ''
        return {
            'title': title,
            'image': img,
            'snippet': snippet,
            'price': price
        }

    def eval(self):
        pass