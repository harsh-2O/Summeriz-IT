from bs4 import BeautifulSoup
import json
import numpy as np
import requests
from requests.models import MissingSchema
# import spacy
import trafilatura

# Collecting the Html Content from the Website.
urls = ['https://understandingdata.com/', 'https://sempioneer.com/', ]

data = {}

for url in urls:
    # 1. Obtain the response.
    resp = requests.get(url)

    # 2. If the response content is 200 - Status Ok, Save The HTML Content:
    if resp.status_code == 200:
        data[url] = resp.text


#  Extracting the text from a single webpage.
def beautifulsoup_extract_text_fallback(response_content):
    '''
    This is a fallback function, so that we can always return a value for text content.
    Even for when both Trafilatura and BeautifulSoup are unable to extract the text from a
    single URL.
    '''

    # Create the beautifulsoup object:
    soup = BeautifulSoup(response_content, 'html.parser')

    # Finding the text:
    text = soup.find_all(text=True)

    # Remove unwanted tag elements:
    cleaned_text = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        'style', ]

    # Then we will loop over every item in the extract text and make sure that the beautifulsoup4 tag
    # is NOT in the blacklist
    for item in text:
        if item.parent.name not in blacklist:
            cleaned_text += '{} '.format(item)

    # Remove any tab separation and strip the text:
    cleaned_text = cleaned_text.replace('\t', '')
    return cleaned_text.strip()


def extract_text_from_single_web_page(url):
    downloaded_url = trafilatura.fetch_url(url)
    try:
        a = trafilatura.extract(downloaded_url, with_metadata=True, include_comments=False,
                                date_extraction_params={'extensive_search': True, 'original_date': True},
                                output_format='json')
    except AttributeError:
        a = trafilatura.extract(downloaded_url, with_metadata=True,
                                date_extraction_params={'extensive_search': True, 'original_date': True},
                                output_format='json')
    if a:
        json_output = json.loads(a)
        return json_output['text']
    else:
        try:
            resp = requests.get(url)
            # We will only extract the text from successful requests:
            if resp.status_code == 200:
                return beautifulsoup_extract_text_fallback(resp.content)
            else:
                # This line will handle for any failures in both the Trafilature and BeautifulSoup4 functions:
                return np.nan
        # Handling for any URLs that don't have the correct protocol
        except MissingSchema:
            return np.nan


single_url = 'https://medium.com/@devkosal/switching-java-jdk-versions-on-macos-80bc868e686a'
text = extract_text_from_single_web_page(url=single_url)
print(text)



text_file = open("test.txt", "w")
n = text_file.write(text)
text_file.close()