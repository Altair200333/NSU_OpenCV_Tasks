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
