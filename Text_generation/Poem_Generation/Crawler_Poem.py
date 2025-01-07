import os
import pandas as pd
import requests
import time
import random
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


WEBDRIVER_DELAY_TIME_INT = 10
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.headless = True

driver = webdriver.Chrome(options=chrome_options)
driver.implicitly_wait(5)
wait = WebDriverWait(driver, WEBDRIVER_DELAY_TIME_INT)


def crawl_data():
    datasets = []
    deletion_script = 'arguments[0].parentNode.removeChild(arguments[0]);'

    datasets = []
    deletion_script = 'arguments[0].parentNode.removeChild(arguments[0]);'
    for page_idx in tqdm(range(1, 11)):
        main_url = f'https://www.thivien.net/searchpoem.php?PoemType=16&ViewType=1&Country=2&Age[]=3&Page={page_idx}'
        driver.get(main_url)

        content_tags_xpath = '//*[@class="page-content container"]/div[2]/div/div[@class="list-item"]'
        content_tags = driver.find_elements(By.XPATH, content_tags_xpath)
        for idx in range(len(content_tags)):
            content_tag_xpath = f'/html/body/div[4]/div[2]/div/div[{2+idx}]'
            content_title_xpath = f'/html/body/div[4]/div[2]/div/div[{2+idx}]/h4/a'
            content_tag = wait.until(EC.presence_of_element_located((By.XPATH, content_tag_xpath)))
            poem_title = wait.until(EC.presence_of_element_located((By.XPATH, content_title_xpath))).text
            poem_url = wait.until(EC.presence_of_element_located((By.XPATH, content_title_xpath))).get_attribute('href')

            try:
                driver.get(poem_url)

                poem_src_xpath = '//div[@class="small"]'
                poem_content_tag = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'poem-content')))

                try:
                    poem_content_i_tag = poem_content_tag.find_element(By.TAG_NAME, 'i')
                    driver.execute_script(deletion_script, poem_content_i_tag)
                except:
                    pass

                try:
                    poem_content_b_tag = poem_content_tag.find_element(By.TAG_NAME, 'b')
                    driver.execute_script(deletion_script, poem_content_b_tag)
                except:
                    pass

                poem_content = poem_content_tag.text

                try:
                    poem_src_tag = wait.until(EC.presence_of_element_located((By.XPATH, poem_src_xpath)))
                    poem_src = poem_src_tag.text
                except:
                    poem_src = ''

                poem_info = {
                    'title': poem_title,
                    'content': poem_content,
                    'source': poem_src,
                    'url': poem_url
                }

                datasets.append(poem_info)

                driver.back()
            except Exception as e:
                print(e)
                print(poem_url)
    return datasets


def save_to_csv():
    datasets = crawl_data()
    df = pd.DataFrame(datasets)

    df.to_csv('poem-datasets.csv', index=True)
    return df


if __name__ == "__main__":
    df = save_to_csv()