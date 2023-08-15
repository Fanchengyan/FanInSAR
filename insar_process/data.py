from pathlib import Path
import re


class Sentinel1:
    def __init__(self) -> None:
        pass

    def parse_url_from_asf(self, url):
        """
        Parse the url from asf to get the product id
        """
        return url.split('/')[-1].split('.')[0]


class ALOS2:
    def __init__(self) -> None:
        pass
