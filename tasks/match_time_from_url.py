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


def match_time_id(id, game_num=1):
    page_url = "https://game-tournaments.com/"

    # mimic ajax request
    headers = {'content-type': 'application/x-www-form-urlencoded; charset=UTF-8', 'x-requested-with': 'XMLHttpRequest'}
    raw_data = 'game=&rid=matches&ajax=block_video&data%5Bid%5D=' + str(id) + '&data%5Bnum%5D=' + str(game_num)

    page = requests.post(page_url, headers=headers, data=raw_data)

    soup = BeautifulSoup(page.content, "html.parser")

    job_elements = soup.find_all("a", class_="g-rezults")
    address = job_elements[0]["href"]
    image_url = 'https://en.game-tournaments.com' + address

    image = url_to_image(image_url)

    return math_time(image), image

    # old method;
    page_url = "https://game-tournaments.com/dota-2/bts-pro-series-season-9/sea/boom-vs-nigma-galaxy-sea-" + str(id)
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, "html.parser")

    job_elements = soup.find_all("a", class_="g-rezults")

    address = job_elements[0]["href"]
    image_url = 'https://en.game-tournaments.com' + address

    image = url_to_image(image_url)

    return math_time(image), image
