import os
import sys
import chromedriver_autoinstaller
import unittest
from selenium import webdriver
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
        if chrome_bin and os.path.exists(chrome_bin):
            options.binary_location = chrome_bin
        else:
            if sys.platform.startswith('linux'):
                for bin_path in ["/usr/bin/google-chrome-stable","/usr/bin/google-chrome","/usr/bin/chromium-browser","/usr/bin/chromium"]:
                    if os.path.exists(bin_path):
                        options.binary_location = bin_path
                        break
            elif sys.platform == 'win32':
                default_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
                if os.path.exists(default_path):
                    options.binary_location = default_path
        # Auto-install matching chromedriver
        driver_path = chromedriver_autoinstaller.install()
        service = Service(driver_path)
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
        links = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'ul li a'))
        )
        self.assertGreater(len(links), 0)
        for link in links:
            href = link.get_attribute('href')
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
