import urllib
from newspaper import Article
url = 'https://www.theatlantic.com/ideas/archive/2022/02/canada-anti-vaccine-trucker-protests/622060/'

def meta_extract(url):
    article = Article(url)
    article.download()
    article.parse()
    article.download('punkt')
    article.nlp()

    return article.authors, article.publish_date, article.top_image, article.images, article.title, article.summary
    
    # print(f"Author: {str(article.authors)}\nDate: {str(article.publish_date.strftime('%m/%d/%y'))}")
    # print(f"Top Image: {article.top_image}\n")
    # print(f"All Images: {[image for image in article.images]}\n")

    # print(f"Summary: {article.title}")
    # print(f"\t {article.summary}")