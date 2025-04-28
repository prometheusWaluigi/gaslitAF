import os
import unittest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class TestAllLinks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Skip local runs without proper driver
        if not os.getenv('GITHUB_ACTIONS'):
            raise unittest.SkipTest('Skipping Selenium tests outside CI')
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        # Allow custom Chrome binary via env var
        chrome_bin = os.getenv('CHROME_BINARY')
        if chrome_bin and os.path.exists(chrome_bin):
            options.binary_location = chrome_bin
        # Try Selenium Manager first, then fallback to webdriver-manager
        try:
            cls.driver = webdriver.Chrome(options=options)
        except Exception:
            service = Service(ChromeDriverManager().install())
            cls.driver = webdriver.Chrome(service=service, options=options)
        base_url = os.getenv('LOCAL_DOCS_URL')
        if not base_url:
            base_url = 'file://' + os.path.abspath('docs/index.html')
        cls.base_url = base_url

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def test_theory_links(self):
        self.driver.get(self.base_url)
        elems = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'ul li a'))
        )
        hrefs = [e.get_attribute('href') for e in elems]
        self.assertGreater(len(hrefs), 0)
        for href in hrefs:
            self.assertTrue(href.startswith('http') or href.startswith('file://'))
            self.driver.get(href)
            if 'github.com' in href:
                # GitHub folder page: verify non-empty title
                self.assertTrue(self.driver.title)
            else:
                header = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'h1'))
                )
                self.assertTrue(header.text)
            # Return to index
            self.driver.get(self.base_url)

    def test_audio_elements(self):
        self.driver.get(self.base_url)
        sources = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'audio source'))
        )
        self.assertGreater(len(sources), 0)
        seen = set()
        for source in sources:
            src = source.get_attribute('src')
            self.assertTrue(src.endswith('.mp3'))
            self.assertNotIn(src, seen)
            seen.add(src)

if __name__ == '__main__':
    unittest.main()
