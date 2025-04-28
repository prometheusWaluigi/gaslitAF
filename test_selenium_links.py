import os
import unittest
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestAllLinks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        # Allow custom Chrome binary via env var
        chrome_bin = os.getenv('CHROME_BINARY')
        if chrome_bin:
            options.binary_location = chrome_bin
        try:
            service = Service(ChromeDriverManager().install())
            cls.driver = webdriver.Chrome(service=service, options=options)
        except Exception as e:
            raise unittest.SkipTest(f"Skipping Selenium tests: {e}")
        base_url = os.getenv('LOCAL_DOCS_URL')
        if not base_url:
            base_url = 'file://' + os.path.abspath('docs/index.html')
        cls.base_url = base_url

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def test_theory_links(self):
        self.driver.get(self.base_url)
        links = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'ul li a'))
        )
        self.assertGreater(len(links), 0)
        for link in links:
            href = link.get_attribute('href')
            self.assertTrue(href.startswith('http') or href.startswith('file://'))
            self.driver.get(href)
            header = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'h1'))
            )
            self.assertTrue(header.text)
            # Return to index
            self.driver.get(self.base_url)

    def test_audio_elements(self):
        self.driver.get(self.base_url)
        audios = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, 'audio'))
        )
        self.assertGreater(len(audios), 0)
        seen = set()
        for audio in audios:
            src = audio.get_attribute('src')
            self.assertTrue(src.endswith('.mp3'))
            self.assertNotIn(src, seen)
            seen.add(src)

if __name__ == '__main__':
    unittest.main()
