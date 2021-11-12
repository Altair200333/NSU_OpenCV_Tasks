from cv2 import cv2 as cv
import numpy as np

from dota_match_time import *
import requests
from bs4 import BeautifulSoup
import urllib
from dota_match_time import *


def url_to_image(url):
    page = requests.get(url)
    image = np.asarray(bytearray(page.content), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image


def match_time_url(url):
    page_url = "https://en.game-tournaments.com/dota-2/" + url
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, "html.parser")

    job_elements = soup.find_all("a", class_="g-rezults")

    address = job_elements[0]["href"]
    image_url = 'https://en.game-tournaments.com' + address

    image = url_to_image(image_url)

    return math_time(image), image


# как такого id мачта там нема, ближайшее шо есть это число в адресной строке, которое вроде уникально для кажой игры, с виду это чуть ли не порядковый номер
# поиска по этому номеру тоже штатного нет, но методом тактического тыка было обнаружено, что при замене id в адресной строке, сайт перенаправит сам на нужный матч
# то есть нужный механизм там есть, но его достать только таким котылём можно
def match_time_id(id):
    page_url = "https://game-tournaments.com/dota-2/bts-pro-series-season-9/sea/boom-vs-nigma-galaxy-sea-" + str(id)
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, "html.parser")

    job_elements = soup.find_all("a", class_="g-rezults")

    address = job_elements[0]["href"]
    image_url = 'https://en.game-tournaments.com' + address

    image = url_to_image(image_url)

    return math_time(image), image
