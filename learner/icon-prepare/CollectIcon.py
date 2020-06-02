
import lxml.html
from selenium import webdriver
import urllib.request

for page in range(22,34):
    target_url = 'https://iconmonstr.com/page/' + str(page)
    driver = webdriver.PhantomJS()
    driver.get(target_url)
    root = lxml.html.fromstring(driver.page_source)
    images = root.cssselect('.container-content img')
    for image in images:
        url = image.get('data-src')
        print(url)
        urllib.request.urlretrieve(url, "icons/" + url.split("/")[-1])
