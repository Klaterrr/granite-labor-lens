import os
import re
import nltk
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup, Tag
from datetime import datetime
import time
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.warning("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

class LegalTextProcessor:
    def __init__(self):
        self.segmenter = self._init_segmenter()
        self.regex_patterns = self._init_regex_patterns()

    def _init_segmenter(self):
        try:
            return nltk.data.load('tokenizers/punkt/english.pickle')
        except Exception as e:
            logger.warning(f"Failed to load NLTK tokenizer: {e}")
            return None

    def _init_regex_patterns(self):
        return {
            'LEGAL_ENTITY': [
                r'(?:Convention|Protocol|Recommendation)\s+No\.\s*\d+',
                r'(?:Article|Section)\s+\d+[a-z]*',
                r'(?:paragraph|para\.)\s+\d+',
                r'C\d+|P\d+|R\d+'
            ]
        }

    def segment_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        if self.segmenter:
            try:
                return self.segmenter.tokenize(text)
            except Exception as e:
                logger.warning(f"Fallback tokenizer failed: {e}")
        return [text]

    def preprocess_text(self, text: str) -> str:
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(Art\.|Sec\.|Para\.|Ref\.) (\d+)', r'\1_PRESERVED_\2', text)
        text = re.sub(r'(\w+)\s+v\.\s+(\w+)', r'\1 v_PRESERVED_ \2', text)
        return text.strip()

    def postprocess_text(self, text: str) -> str:
        text = text.replace('_PRESERVED_', ' ')
        text = text.replace('v_PRESERVED_', 'v.')
        return text.strip()

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities = {'LEGAL_ENTITY': []}
        for pattern in self.regex_patterns['LEGAL_ENTITY']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities['LEGAL_ENTITY'].append(match.group())
        for key in entities:
            entities[key] = sorted(set(entities[key]))
        return entities

class ILOCategorizedScraper:
    """Specialized scraper for ILO protocols and categorized conventions."""
    
    def __init__(self, output_dir: str = "ilo_data"):
        """
        Initialize the scraper.
        
        Args:
            output_dir: Directory to save scraped data
        """
        self.base_url = "https://normlex.ilo.org"
        self.protocols_url = "https://normlex.ilo.org/dyn/normlex/en/f?p=1000:12005::::::"
        self.conventions_by_subject_url = "https://normlex.ilo.org/dyn/normlex/en/f?p=1000:12030::::::"
        
        # Set up output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize session for persistent connections
        self.session = requests.Session()
        
        # Set request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Initialize legal text processor
        self.text_processor = LegalTextProcessor()
        
        # Track visited URLs to avoid duplicates
        self.visited_urls = set()
        
        # Store category mapping
        self.category_mapping = {}
        
        # Collections to store all scraped data
        self.all_protocols = []
        self.all_conventions = []
        
    def reset_visited_urls(self):
        """Reset the visited URLs cache."""
        self.visited_urls = set()
        logger.info("Reset visited URLs cache")
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a page with improved error handling.
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None if fetching fails
        """
        # Fix URL if it's a relative path with f?p= format
        if url.startswith('f?p='):
            url = f"{self.base_url}/dyn/normlex/en/{url}"
            
        # Check if URL has been visited
        if url in self.visited_urls:
            logger.info(f"Already visited: {url}")
            return None
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Add a delay to avoid overwhelming the server
                time.sleep(1.5 + retry_count)
                
                # Make the request
                logger.info(f"Fetching: {url}")
                response = self.session.get(url, headers=self.headers, timeout=30)
                
                # Check if request was successful
                if response.status_code == 200:
                    # Parse the HTML
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Mark URL as visited
                    self.visited_urls.add(url)
                    
                    return soup
                else:
                    logger.error(f"Failed to fetch {url}. Status code: {response.status_code}")
                    retry_count += 1
                    
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                retry_count += 1
        
        logger.error(f"Failed to fetch {url} after {max_retries} attempts")
        return None
    
    def extract_protocols(self) -> List[Dict[str, Any]]:
        """
        Extract protocols from the protocols page.
        
        Returns:
            List of protocol metadata
        """
        # Fetch protocols page
        soup = self.fetch_page(self.protocols_url)
        if not soup:
            logger.error("Failed to fetch protocols page")
            return []
        
        protocols = []
        
        # Find the protocol content container
        protocol_content = soup.find('div', class_='featureMultiple FM3')
        if not protocol_content:
            logger.error("Protocol content container not found")
            logger.info(f"Available classes: {[div.get('class') for div in soup.find_all('div') if div.get('class')][:10]}")
            return []
        
        # Process both the Fundamental and Technical sections
        content_sections = protocol_content.find_all('div', class_='boxContent')
        
        for section in content_sections:
            # Extract all the protocol links
            for li in section.find_all('li'):
                link = li.find('a')
                if not link:
                    continue
                
                # Extract protocol code and title
                protocol_text = link.get_text(strip=True)
                strong_tag = link.find('strong')
                
                if strong_tag:
                    code = strong_tag.get_text(strip=True)
                    # Remove the code from the full text to get the title
                    title = protocol_text.replace(code, '', 1).strip()
                    if title.startswith('-'):
                        title = title[1:].strip()
                else:
                    # Fallback to regex if strong tag not present
                    code_match = re.match(r'(P\d+)\s*-\s*(.*)', protocol_text)
                    if code_match:
                        code = code_match.group(1)
                        title = code_match.group(2)
                    else:
                        code = ""
                        title = protocol_text
                
                # Extract protocol URL with proper formatting
                protocol_url = link.get('href', '')
                if protocol_url.startswith('f?p='):
                    # These are special ILO relative URLs that need the full path
                    protocol_url = f"{self.base_url}/dyn/normlex/en/{protocol_url}"
                elif not protocol_url.startswith('http'):
                    protocol_url = urljoin(self.base_url, protocol_url)
                
                logger.info(f"Constructed protocol URL: {protocol_url}")
                
                # Extract instrument ID from URL
                instrument_id_match = re.search(r'P12100_INSTRUMENT_ID:(\d+)', protocol_url)
                instrument_id = instrument_id_match.group(1) if instrument_id_match else ""
                
                # Get section name for categorization
                section_heading = section.find_previous('h4')
                category = section_heading.get_text(strip=True) if section_heading else ""
                
                protocols.append({
                    'code': code,
                    'title': title,
                    'url': protocol_url,
                    'instrument_id': instrument_id,
                    'category': category
                })
        
        logger.info(f"Found {len(protocols)} protocols")
        return protocols
    
    def extract_subjects_and_categories(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract subject categories and their conventions.
        
        Returns:
            Dictionary mapping subjects to their conventions
        """
        # Fetch conventions by subject page
        soup = self.fetch_page(self.conventions_by_subject_url)
        if not soup:
            logger.error("Failed to fetch conventions by subject page")
            return {}
        
        subjects = {}
        current_subject = None
        current_subcategory = None
        
        # Look for main content container
        content_container = soup.find('div', id='colMain')
        if not content_container:
            logger.error("Content container not found")
            return {}
        
        # Process all elements
        for element in content_container.find_all(['h4', 'h5', 'div', 'ul', 'ol']):
            # Main subject heading (e.g., "1. Freedom of association")
            if element.name == 'h4':
                current_subject = element.get_text(strip=True)
                current_subcategory = None
                subjects[current_subject] = {
                    'subcategories': {},
                    'conventions': []
                }
            
            # Subcategory heading (e.g., "1.1. Fundamental Conventions")
            elif element.name == 'h5' and current_subject:
                current_subcategory = element.get_text(strip=True)
                subjects[current_subject]['subcategories'][current_subcategory] = []
            
            # Check for up-to-date instrument div
            elif element.name == 'div' and 'titleH7' in element.get('class', []):
                if element.get_text(strip=True) == 'Up-to-date instrument':
                    # Look for lists (ul or ol) that might contain conventions
                    next_list = element.find_next('ul') or element.find_next('ol')
                    if next_list:
                        conventions = self._extract_conventions_from_list(next_list)
                        
                        # Add conventions to current subcategory if exists
                        if current_subject and current_subcategory:
                            subjects[current_subject]['subcategories'][current_subcategory] = conventions
                        # Otherwise add to subject directly
                        elif current_subject:
                            subjects[current_subject]['conventions'].extend(conventions)
                        
                        # Update category mapping
                        for convention in conventions:
                            self.category_mapping[convention['code']] = {
                                'subject': current_subject,
                                'subcategory': current_subcategory
                            }
                    
                    # Also check for tables for backward compatibility
                    next_table = element.find_next('table', class_='report')
                    if next_table:
                        table_conventions = self._extract_conventions_from_table(next_table)
                        
                        # Add conventions to current subcategory if exists
                        if current_subject and current_subcategory:
                            existing = subjects[current_subject]['subcategories'].get(current_subcategory, [])
                            subjects[current_subject]['subcategories'][current_subcategory] = existing + table_conventions
                        # Otherwise add to subject directly
                        elif current_subject:
                            subjects[current_subject]['conventions'].extend(table_conventions)
                        
                        # Update category mapping
                        for convention in table_conventions:
                            self.category_mapping[convention['code']] = {
                                'subject': current_subject,
                                'subcategory': current_subcategory
                            }
        
        # Count total conventions
        total_conventions = sum(
            len(data['conventions']) + sum(len(conventions) for conventions in data['subcategories'].values())
            for data in subjects.values()
        )
        
        logger.info(f"Found {len(subjects)} subjects with {total_conventions} up-to-date conventions")
        return subjects

    def _extract_conventions_from_list(self, list_element: Tag) -> List[Dict[str, Any]]:
        """
        Extract conventions from a list (ul or ol).
        
        Args:
            list_element: BeautifulSoup Tag representing a list
            
        Returns:
            List of convention metadata
        """
        conventions = []
        
        # Extract conventions from list items
        for li in list_element.find_all('li', recursive=False):
            link = li.find('a')
            if not link:
                continue
            
            # Extract convention code and title
            convention_text = link.get_text(strip=True)
            strong_tag = link.find('strong')
            
            if strong_tag:
                code = strong_tag.get_text(strip=True)
                # Remove the code from the full text to get the title
                title = convention_text.replace(code, '', 1).strip()
                if title.startswith('-'):
                    title = title[1:].strip()
            else:
                # Fallback to regex if strong tag not present
                code_match = re.match(r'(C\d+)\s*-\s*(.*)', convention_text)
                if code_match:
                    code = code_match.group(1)
                    title = code_match.group(2)
                else:
                    code = ""
                    title = convention_text
            
            # Extract convention URL
            convention_url = link.get('href', '')
            if convention_url.startswith('f?p='):
                # Fix URL to include proper path
                convention_url = f"{self.base_url}/dyn/normlex/en/{convention_url}"
            elif not convention_url.startswith('http'):
                convention_url = urljoin(self.base_url, convention_url)
            
            # Extract instrument ID from URL
            instrument_id_match = re.search(r'P12100_INSTRUMENT_ID:(\d+)', convention_url)
            instrument_id = instrument_id_match.group(1) if instrument_id_match else ""
            
            conventions.append({
                'code': code,
                'title': title,
                'url': convention_url,
                'instrument_id': instrument_id
            })
        
        return conventions
        
        def _extract_conventions_from_table(self, table: Tag) -> List[Dict[str, Any]]:
            """
            Extract conventions from a table.
            
            Args:
                table: BeautifulSoup Tag representing a table
                
            Returns:
                List of convention metadata
            """
            conventions = []
            
            # Extract conventions from table rows
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if len(cells) >= 2:
                    # Find convention link in the second column
                    link = cells[1].find('a')
                    if not link:
                        continue
                    
                    # Extract convention code and title
                    convention_text = link.get_text(strip=True)
                    code_match = re.match(r'(C\d+)\s*-\s*(.*)', convention_text)
                    
                    if code_match:
                        code = code_match.group(1)
                        title = code_match.group(2)
                    else:
                        code = ""
                        title = convention_text
                    
                    # Extract convention URL
                    convention_url = link['href']
                    if convention_url.startswith('f?p='):
                        # Fix URL to include proper path
                        convention_url = f"{self.base_url}/dyn/normlex/en/{convention_url}"
                    elif not convention_url.startswith('http'):
                        convention_url = urljoin(self.base_url, convention_url)
                    
                    # Extract instrument ID from URL
                    instrument_id_match = re.search(r'P12100_INSTRUMENT_ID:(\d+)', convention_url)
                    instrument_id = instrument_id_match.group(1) if instrument_id_match else ""
                    
                    conventions.append({
                        'code': code,
                        'title': title,
                        'url': convention_url,
                        'instrument_id': instrument_id
                    })
            
            return conventions
        
    def scrape_instrument(self, instrument_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape a specific instrument (protocol or convention).
        
        Args:
            instrument_data: Metadata about the instrument to scrape
            
        Returns:
            Structured data for the instrument
        """
        url = instrument_data['url']
        instrument_id = instrument_data['instrument_id']
        
        logger.info(f"Attempting to scrape instrument: {url}")
        
        # Fetch the instrument page
        soup = self.fetch_page(url)
        if not soup:
            logger.error(f"Failed to fetch instrument: {url}")
            return {}
        
        # Extract structured data
        instrument = {
            'id': instrument_id,
            'code': instrument_data['code'],
            'title': instrument_data['title'],
            'url': url,
            'metadata': {},
            'preamble': '',
            'articles': [],
            'categories': self.category_mapping.get(instrument_data['code'], {}),
            'extracted_date': datetime.now().isoformat()
        }
        
        # If category is provided in the instrument_data, add it
        if 'category' in instrument_data:
            instrument['category'] = instrument_data['category']
        
        # Extract metadata from key information box
        key_info_box = soup.find('div', class_='featureMultiple FM2 boxWithBorder')
        if key_info_box:
            box_content = key_info_box.find('div', class_='boxContent')
            if box_content:
                # Extract key metadata
                for info_line in box_content.get_text().split('\n'):
                    info_line = info_line.strip()
                    if not info_line:
                        continue
                    
                    # Parse adoption info
                    if 'Adoption:' in info_line:
                        instrument['metadata']['adoption'] = info_line.split('Adoption:')[1].strip()
                    
                    # Parse entry into force
                    elif 'Entry into force:' in info_line:
                        instrument['metadata']['entry_into_force'] = info_line.split('Entry into force:')[1].strip()
                    
                    # Parse status
                    elif 'Status:' in info_line:
                        instrument['metadata']['status'] = info_line.split('Status:')[1].strip()
        
        # Find the main content container
        content_container = soup.find('div', class_='textBoxConvention')
        if not content_container:
            logger.error(f"Content container not found for {url}")
            return instrument
        
        # Extract preamble (if available)
        preamble_element = content_container.find('div', class_='frame')
        if preamble_element:
            instrument['preamble'] = preamble_element.get_text(strip=True)
        
        # Extract articles
        articles = []
        article_headers = content_container.find_all('h5')
        
        # If no article headers found, look for anchors that may indicate articles
        if not article_headers:
            logger.info(f"No h5 article headers found for {url}, checking for article anchors")
            
            # Look for anchors (links) that might point to articles
            article_anchors = content_container.find_all('a', attrs={'name': True})
            
            for anchor in article_anchors:
                anchor_name = anchor['name']
                
                # Check if it's an article anchor (typically starts with 'A' followed by a number)
                if re.match(r'A\d+', anchor_name):
                    article_num = anchor_name[1:]  # Remove the 'A' prefix
                    
                    # Try to find the article header or content
                    article_content_container = anchor.parent
                    
                    # Create an article object
                    article = {
                        'number': article_num,
                        'title': f"Article {article_num}",
                        'id': anchor_name,
                        'paragraphs': []
                    }
                    
                    # Extract paragraphs following this anchor
                    current_element = article_content_container.next_sibling
                    
                    while current_element and not (isinstance(current_element, Tag) and current_element.name == 'a' and current_element.get('name', '').startswith('A')):
                        if isinstance(current_element, Tag):
                            if current_element.name == 'p':
                                paragraph_text = current_element.get_text(strip=True)
                                if paragraph_text:
                                    processed_text = self.text_processor.preprocess_text(paragraph_text)
                                    sentences = self.text_processor.segment_sentences(processed_text)
                                    
                                    article['paragraphs'].append({
                                        'type': 'text',
                                        'content': paragraph_text,
                                        'sentences': sentences
                                    })
                            elif current_element.name == 'ol':
                                for li in current_element.find_all('li', recursive=False):
                                    para_content = li.get_text(strip=True)
                                    para_num_match = re.match(r'(\d+)\.', para_content)
                                    para_num = para_num_match.group(1) if para_num_match else ""
                                    
                                    processed_text = self.text_processor.preprocess_text(para_content)
                                    sentences = self.text_processor.segment_sentences(processed_text)
                                    
                                    article['paragraphs'].append({
                                        'type': 'numbered',
                                        'number': para_num,
                                        'content': para_content,
                                        'sentences': sentences
                                    })
                        
                        current_element = current_element.next_sibling
                    
                    # Add article to list if it has paragraphs
                    if article['paragraphs']:
                        articles.append(article)
        else:
            # Process standard article headers
            for article_header in article_headers:
                # Extract article number and title
                article_title = article_header.get_text(strip=True)
                article_num_match = re.match(r'Article\s+(\d+[a-z]*)', article_title)
                
                if not article_num_match:
                    continue
                    
                article_num = article_num_match.group(1)
                
                # Check for article anchor
                article_id = None
                anchor = article_header.find('a', attrs={'name': True})
                if anchor:
                    article_id = anchor['name']
                
                # Initialize article data
                article = {
                    'number': article_num,
                    'title': article_title,
                    'id': article_id,
                    'paragraphs': []
                }
                
                # Extract article content (elements until next article header)
                current_element = article_header.next_sibling
                
                while current_element and (not isinstance(current_element, Tag) or current_element.name != 'h5'):
                    if isinstance(current_element, Tag):
                        # Handle different content types
                        if current_element.name == 'p':
                            # Simple paragraph
                            paragraph_text = current_element.get_text(strip=True)
                            if paragraph_text:
                                # Process paragraph with legal NLP
                                processed_text = self.text_processor.preprocess_text(paragraph_text)
                                sentences = self.text_processor.segment_sentences(processed_text)
                                
                                article['paragraphs'].append({
                                    'type': 'text',
                                    'content': paragraph_text,
                                    'sentences': sentences
                                })
                        
                        elif current_element.name == 'ol':
                            # Numbered paragraphs
                            for li in current_element.find_all('li', recursive=False):
                                # Check for paragraph anchor
                                para_id = None
                                anchor = li.find('a', attrs={'name': True})
                                if anchor:
                                    para_id = anchor['name']
                                
                                # Extract paragraph number if available
                                para_num_match = re.match(r'(\d+)\.', li.get_text(strip=True))
                                para_num = para_num_match.group(1) if para_num_match else ""
                                
                                # Extract paragraph content
                                paragraph_content = li.get_text(strip=True)
                                
                                # Check for nested bullets
                                bullets = []
                                bullet_list = li.find('ul')
                                
                                if bullet_list:
                                    for bullet in bullet_list.find_all('li'):
                                        bullet_text = bullet.get_text(strip=True)
                                        bullets.append(bullet_text)
                                    
                                    # Remove bullet list from content to avoid duplication
                                    bullet_list_text = bullet_list.get_text()
                                    paragraph_content = paragraph_content.replace(bullet_list_text, '').strip()
                                
                                # Process paragraph with legal NLP
                                processed_text = self.text_processor.preprocess_text(paragraph_content)
                                sentences = self.text_processor.segment_sentences(processed_text)
                                
                                # Create paragraph data
                                paragraph_data = {
                                    'type': 'numbered',
                                    'number': para_num,
                                    'id': para_id,
                                    'content': paragraph_content,
                                    'sentences': sentences
                                }
                                
                                if bullets:
                                    paragraph_data['bullets'] = bullets
                                
                                article['paragraphs'].append(paragraph_data)
                        
                        elif current_element.name == 'ul':
                            # Bullet list
                            bullets = []
                            for li in current_element.find_all('li'):
                                bullet_text = li.get_text(strip=True)
                                bullets.append(bullet_text)
                            
                            if bullets:
                                article['paragraphs'].append({
                                    'type': 'bullets',
                                    'content': bullets
                                })
                    
                    # Move to next element
                    current_element = current_element.next_sibling
                
                # Add article to list
                articles.append(article)
        
        instrument['articles'] = articles
        
        # Extract entities from the whole text
        full_text = ""
        if instrument['preamble']:
            full_text += instrument['preamble'] + " "
        
        for article in articles:
            for paragraph in article['paragraphs']:
                if isinstance(paragraph.get('content'), str):
                    full_text += paragraph['content'] + " "
                elif isinstance(paragraph.get('content'), list):
                    full_text += " ".join(paragraph['content']) + " "
        
        instrument['entities'] = self.text_processor.extract_entities(full_text)
        
        return instrument
    
    def scrape_all_protocols(self) -> List[Dict[str, Any]]:
        """
        Scrape all protocols from the protocols page.
        
        Returns:
            List of scraped protocols
        """
        # Extract protocol metadata
        protocols = self.extract_protocols()
        
        scraped_protocols = []
        
        # Scrape each protocol
        total = len(protocols)
        for i, protocol in enumerate(protocols):
            logger.info(f"Scraping protocol {i+1}/{total}: {protocol['code']} - {protocol['title']}")
            
            # Scrape protocol data
            protocol_data = self.scrape_instrument(protocol)
            
            if protocol_data:
                logger.info(f"Successfully scraped protocol: {protocol['code']}")
                scraped_protocols.append(protocol_data)
                
                # Add a delay to avoid overwhelming the server
                time.sleep(2)
            
        self.all_protocols = scraped_protocols
        return scraped_protocols
    
    def scrape_uptodate_conventions(self) -> List[Dict[str, Any]]:
        """
        Scrape up-to-date conventions organized by subject categories.
        
        Returns:
            List of scraped conventions
        """
        # Extract subjects and conventions
        subjects = self.extract_subjects_and_categories()
        
        scraped_conventions = []
        categories_data = []
        
        # Process each subject
        for subject, data in subjects.items():
            subject_key = re.sub(r'[^\w\s-]', '', subject).strip().lower()
            subject_key = re.sub(r'[-\s]+', '_', subject_key)
            
            subject_conventions = []
            
            # Create category data
            category_data = {
                'name': subject,
                'key': subject_key,
                'subcategories': [],
                'conventions': []
            }
            
            # Scrape conventions in the subject
            for convention in data['conventions']:
                logger.info(f"Scraping convention: {convention['code']} - {convention['title']}")
                
                # Add category info
                convention['category'] = subject
                
                # Scrape convention data
                convention_data = self.scrape_instrument(convention)
                
                if convention_data:
                    # Save metadata in categories
                    category_data['conventions'].append({
                        'code': convention_data['code'],
                        'title': convention_data['title'],
                        'url': convention_data['url']
                    })
                    
                    # Add to scraped conventions
                    scraped_conventions.append(convention_data)
                    subject_conventions.append(convention_data)
                    
                    # Add a delay to avoid overwhelming the server
                    time.sleep(2)
            
            # Process subcategories
            for subcategory, conventions in data['subcategories'].items():
                subcategory_key = re.sub(r'[^\w\s-]', '', subcategory).strip().lower()
                subcategory_key = re.sub(r'[-\s]+', '_', subcategory_key)
                
                subcategory_data = {
                    'name': subcategory,
                    'key': subcategory_key,
                    'conventions': []
                }
                
                # Scrape conventions in the subcategory
                for convention in conventions:
                    logger.info(f"Scraping convention: {convention['code']} - {convention['title']}")
                    
                    # Add category info
                    convention['category'] = subject
                    convention['subcategory'] = subcategory
                    
                    # Scrape convention data
                    convention_data = self.scrape_instrument(convention)
                    
                    if convention_data:
                        # Save metadata in subcategories
                        subcategory_data['conventions'].append({
                            'code': convention_data['code'],
                            'title': convention_data['title'],
                            'url': convention_data['url']
                        })
                        
                        # Add to scraped conventions
                        scraped_conventions.append(convention_data)
                        subject_conventions.append(convention_data)
                        
                        # Add a delay to avoid overwhelming the server
                        time.sleep(2)
                
                # Add subcategory to category
                category_data['subcategories'].append(subcategory_data)
            
            # Add category to categories data
            categories_data.append(category_data)
        
        # Store all conventions
        self.all_conventions = scraped_conventions
        
        return scraped_conventions
    
    # def save_data_files(self) -> None:
    #     """
    #     Save all collected data to a single file with type identification.
    #     """
    #     # Mark each instrument with its type
    #     for protocol in self.all_protocols:
    #         protocol['type'] = 'protocol'
        
    #     for convention in self.all_conventions:
    #         convention['type'] = 'convention'
        
    #     # Combine all instruments
    #     all_instruments = self.all_protocols + self.all_conventions
        
    #     # Save combined instruments
    #     instruments_path = os.path.join(self.output_dir, '../data/instruments.json')
    #     with open(instruments_path, 'w', encoding='utf-8') as f:
    #         json.dump(all_instruments, f, indent=2, ensure_ascii=False)
    #     logger.info(f"Saved {len(all_instruments)} instruments to {instruments_path}")
        
    #     # Save category structure
    #     categories_path = os.path.join(self.output_dir, '../data/categories.json')
        
    #     # Build category structure
    #     categories = {
    #         'subjects': []
    #     }
        
    #     # Add subject categories from mapping
    #     subject_categories = {}
    #     for code, category_info in self.category_mapping.items():
    #         subject = category_info.get('subject', '')
    #         if subject and subject not in subject_categories:
    #             subject_categories[subject] = {
    #                 'name': subject,
    #                 'subcategories': []
    #             }
            
    #         subcategory = category_info.get('subcategory', '')
    #         if subcategory and subcategory not in [sub.get('name') for sub in subject_categories.get(subject, {}).get('subcategories', [])]:
    #             subject_categories[subject]['subcategories'].append({
    #                 'name': subcategory
    #             })
        
    #     categories['subjects'] = list(subject_categories.values())
        
    #     with open(categories_path, 'w', encoding='utf-8') as f:
    #         json.dump(categories, f, indent=2, ensure_ascii=False)
    #     logger.info(f"Saved category structure to {categories_path}")

    def save_data_files(self) -> None:
        """
        Save all collected data to text files with paragraph formatting.
        """

        def format_instrument_text(instrument: Dict[str, Any]) -> str:
            """Helper function to format instrument data into paragraphs."""
            formatted_text = f"Title: {instrument['title']}\nURL: {instrument['url']}\n\n"

            if instrument.get('preamble'):
                formatted_text += "PREAMBLE:\n" + instrument['preamble'] + "\n\n"

            if instrument.get('articles'):
                formatted_text += "ARTICLES:\n"
                for article in instrument['articles']:
                    formatted_text += f"  Article {article['number']}: {article['title']}\n"
                    for paragraph in article['paragraphs']:
                        if paragraph['type'] == 'text' or paragraph['type'] == 'numbered':
                            formatted_text += "    " + paragraph['content'] + "\n"
                        elif paragraph['type'] == 'bullets':
                            for bullet in paragraph['content']:
                                formatted_text += "      - " + bullet + "\n"
                    formatted_text += "\n"  # Add space between articles
            
            formatted_text += "\n" # Add space between instruments
            return formatted_text

        # Combine all instruments and save to a single text file
        all_instruments = self.all_protocols + self.all_conventions
        
        instruments_path = os.path.join(self.output_dir, '../data/instruments.txt')
        with open(instruments_path, 'w', encoding='utf-8') as f:
            for instrument in all_instruments:
                f.write(format_instrument_text(instrument))
        logger.info(f"Saved {len(all_instruments)} instruments to {instruments_path}")
        
        # Save category structure (simplified for text output)
        categories_path = os.path.join(self.output_dir, 'categories.txt')
        with open(categories_path, 'w', encoding='utf-8') as f:
            for subject, data in self.extract_subjects_and_categories().items():
                f.write(f"Subject: {subject}\n")
                for subcategory, conventions in data['subcategories'].items():
                    f.write(f"  Subcategory: {subcategory}\n")
                    for convention in conventions:
                        f.write(f"    - {convention['code']} : {convention['title']}\n")
                f.write("\n")  # Add space between subjects
        logger.info(f"Saved category structure to {categories_path}")

    def run_test_scrape(self, limit: int = 2):
        """Run a limited test scrape for debugging"""
        logger.info(f"Running test scrape with limit of {limit} protocols")
        
        # Extract first few protocols
        protocols = self.extract_protocols()[:limit]
        
        for i, protocol in enumerate(protocols):
            logger.info(f"Test scraping protocol {i+1}/{len(protocols)}: {protocol['code']} - {protocol['title']}")
            protocol_data = self.scrape_instrument(protocol)
            
            if protocol_data:
                logger.info(f"Successfully scraped {protocol['code']}")
                # Print some stats about the extracted data
                logger.info(f"  - Articles: {len(protocol_data.get('articles', []))}")
                total_paragraphs = sum(len(article.get('paragraphs', [])) 
                                      for article in protocol_data.get('articles', []))
                logger.info(f"  - Total paragraphs: {total_paragraphs}")


# Testing code for notebook environment
if __name__ == "__main__":
    # Create an instance of the scraper
    output_dir = "ilo_data"
    scraper = ILOCategorizedScraper(output_dir=output_dir)
    
    # Print debug info
    print(f"Output directory: {os.path.abspath(scraper.output_dir)}")
    print(f"Directories exist: {os.path.exists(scraper.output_dir)}")
    
    # Test file creation to verify permissions
    test_file_path = os.path.join(scraper.output_dir, "test_file.txt")
    try:
        with open(test_file_path, "w") as f:
            f.write("This is a test file.")
        print(f"Test file created successfully at: {test_file_path}")
    except Exception as e:
        print(f"Error creating test file: {str(e)}")
    
    # Reset visited URLs
    scraper.reset_visited_urls()
    
    # Try a small test first
    # print("Running a small test scrape first...")
    # scraper.run_test_scrape(limit=1)
    
    # Reset visited URLs before full scrape
    scraper.reset_visited_urls()
    
    # Run the full scraping operations
    print("Starting protocol scraping...")
    protocols = scraper.scrape_all_protocols()
    print(f"Scraped {len(protocols)} protocols")
    
    # Reset visited URLs before convention scraping
    scraper.reset_visited_urls()
    
    print("Starting convention scraping...")
    conventions = scraper.scrape_uptodate_conventions()
    print(f"Scraped {len(conventions)} conventions")
    
    # Save all data to files
    print("Saving data to files...")
    scraper.save_data_files()
    print("Done!")