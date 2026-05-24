import time
import os
import requests
import logging
from typing import Optional
from dotenv import load_dotenv 


load_dotenv()
logger = logging.getLogger(__name__) #logger for this modulue

NASA_LIBRARY_URL = "https://images-api.nasa.gov/search"
NASA_APOD_URL    = "https://api.nasa.gov/planetary/apod"


class NASAClient:
    def __init__(self):
        self.api_key = os.getenv("NASA_API_KEY","DEMO_KEY")
        self.session = requests.Session()
        self.session.headers.update({"Accept":"application/json"}) #we need explicity json
        

    def search_images(self,query:str,page:int=1,page_size: int =100) -> list[dict]:
        """search nasa image library,return raw item list """
        params = {
            "q":query,
            "media_type":"image",
            "page":page,
            "page_size":page_size,
        }

        #example-https://images-api.nasa.gov/search?q=mars&media_type=image
        try:
            resp = self.session.get(NASA_LIBRARY_URL,params=params,timeout=30)
            resp.raise_for_status()
            return resp.json()["collection"]["items"]
        except requests.RequestException as e:
            logger.error(f"search_images failed for '{query}' : {e}")
            return []


    def search_all_pages(self,query:str,max_images:int=300)->list[dict]:
        """Paginate through result untill max_image is reached """
        results, page = [],1
        while len(results) < max_images:
            batch = self.search_images(query,page=page)
            if not batch:
                break
            results.extend(batch)
            page+=1
            time.sleep(0.5) #dont want to put too much load on NASA servers :)
        return results[:max_images]


    @staticmethod
    def get_preview_url(item:dict)->Optional[str]:
        """Extract preview image URL directly from search result item. """
        for link in item.get("links",[]):
            if link.get("rel") == "preview" and link.get("render") == "image":
                return link["href"]

        return None

    @staticmethod 
    def get_nasa_id(item:dict) -> Optional[str]:
        try:
            return item["data"][0]["nasa_id"]
        except (KeyError,IndexError):
            return None 

    
    def get_apod(self, date: Optional[str] = None) -> Optional[dict]:
        """
        Fetch Astronomy Picture of the Day.
        date format: 'YYYY-MM-DD'  (omit for today)
        """
        params = {"api_key": self.api_key}
        if date:
            params["date"] = date
        try:
            resp = self.session.get(NASA_APOD_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # APOD sometimes returns a video; skip those
            if data.get("media_type") != "image":
                logger.info(f"APOD for {date} is not an image (media_type={data.get('media_type')}), skipping.")
                return None
            return data
        except requests.RequestException as e:
            logger.error(f"get_apod failed: {e}")
            return None
