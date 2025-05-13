#!/usr/bin/env python3
"""
Legal NLP Analyzer for ILO Conventions

This script analyzes ILO conventions and protocols using specialized legal NLP libraries:
- LexNLP: For legal entity extraction and sentence segmentation
- spaCy: For general NER and syntax analysis
- NUPunkt/CharBoundary: For legal-specific sentence boundary detection

The analyzer extracts obligations, rights, definitions, and more from legal texts.
"""

import os
import sys
import re
import json
import logging
import argparse
import csv
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

# Import specialized legal NLP libraries
try:
    import lexnlp
    import lexnlp.extract
    import lexnlp.extract.en
    import lexnlp.nlp.en.segments.sentences
    import lexnlp.extract.en.entities
    import lexnlp.extract.en.definitions
    LEXNLP_AVAILABLE = True
except ImportError:
    LEXNLP_AVAILABLE = False
    print("LexNLP not available. Install using: pip install lexnlp")

try:
    # Try to import NUPunkt if available
    from nupunkt import NUPunkt
    NUPUNKT_AVAILABLE = True
except ImportError:
    NUPUNKT_AVAILABLE = False
    print("NUPunkt not available. Install using: pip install nupunkt")

try:
    # Try to import CharBoundary if available
    from charboundary import CharBoundaryModel
    CHARBOUNDARY_AVAILABLE = True
except ImportError:
    CHARBOUNDARY_AVAILABLE = False
    print("CharBoundary not available. Install using: pip install charboundary")

# Import spaCy if available
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        print("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not available. Install using: pip install spacy")
    nlp = None

# Import NLTK for fallback sentence tokenization
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt resource...")
    nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class LegalNLPAnalyzer:
    """
    Analyzer for legal documents using specialized legal NLP libraries.
    """
    
    def __init__(self):
        """Initialize the analyzer with available NLP components."""
        self.nupunkt = None
        self.charboundary = None
        
        # Initialize NUPunkt if available
        if NUPUNKT_AVAILABLE:
            try:
                self.nupunkt = NUPunkt()
                logger.info("Using NUPunkt for sentence segmentation")
            except Exception as e:
                logger.warning(f"Failed to initialize NUPunkt: {e}")
        
        # Initialize CharBoundary if available
        if CHARBOUNDARY_AVAILABLE:
            try:
                self.charboundary = CharBoundaryModel.load("medium")
                logger.info("Using CharBoundary for sentence segmentation")
            except Exception as e:
                logger.warning(f"Failed to initialize CharBoundary: {e}")
        
        # Initialize NLTK tokenizer as fallback
        self.nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        
        # Customize NLTK tokenizer for legal text
        self._customize_nltk_tokenizer()
        
        # Compile regex patterns for legal analysis
        self._compile_legal_patterns()
    
    def _customize_nltk_tokenizer(self):
        """Customize NLTK tokenizer for legal abbreviations."""
        legal_abbreviations = {
            'art', 'para', 'no', 'nos', 'ref', 'vol', 'vs', 'etc', 'i.e', 'e.g',
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            'co', 'inc', 'ltd', 'corp', 'llc', 'fig', 'p', 'pp', 'cf', 'op', 'cit',
            'v', 'al', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        }
        
        # Add legal abbreviations to the tokenizer
        self.nltk_tokenizer._params.abbrev_types.update(legal_abbreviations)
    
    def _compile_legal_patterns(self):
        """Compile regex patterns for legal analysis."""
        # Patterns for detecting different types of sentences
        self.patterns = {
            'obligation': re.compile(r'\b(?:shall|must|required|ensure|obligated|responsible|duty|duties|comply|undertake)\b', re.IGNORECASE),
            'right': re.compile(r'\b(?:right|rights|entitled|entitlement|freedom|liberty|protection|protected)\b', re.IGNORECASE),
            'prohibition': re.compile(r'\b(?:shall not|must not|prohibited|forbid|forbidden|ban|banned|illegal|unlawful|not permitted|not allowed)\b', re.IGNORECASE),
            'definition': re.compile(r'\b(?:means|signifies|refers to|defined as|definition|understood as|interpreted as|comprises|consisting of)\b', re.IGNORECASE),
            'cross_reference': re.compile(r'\b(?:Article|Section|paragraph|Convention|Protocol|Recommendation)\s+\d+[a-z]*\b', re.IGNORECASE),
            'temporal': re.compile(r'\b(?:within|after|before|during|following|prior to|subsequently|upon|when|while|until)\b \b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b \b(?:days?|weeks?|months?|years?|decades?)\b', re.IGNORECASE)
        }
    
    def segment_sentences(self, text: str, pre_segmented: Optional[List[str]] = None) -> List[str]:
        """
        Segment text into sentences using the best available method or use pre-segmented sentences if provided.
        
        Args:
            text: Text to segment
            pre_segmented: Pre-segmented sentences if available
            
        Returns:
            List of sentences
        """
        # Use pre-segmented sentences if provided
        if pre_segmented and isinstance(pre_segmented, list) and len(pre_segmented) > 0:
            return pre_segmented
        
        if not text:
            return []
        
        # Try LexNLP first (specialized for legal text)
        if LEXNLP_AVAILABLE:
            try:
                sentences = list(lexnlp.nlp.en.segments.sentences.get_sentences(text))
                if sentences:
                    return sentences
            except Exception as e:
                logger.warning(f"LexNLP segmentation failed: {e}")
        
        # Try NUPunkt if available
        if self.nupunkt:
            try:
                sentences = self.nupunkt.segment(text)
                if sentences:
                    return sentences
            except Exception as e:
                logger.warning(f"NUPunkt segmentation failed: {e}")
        
        # Try CharBoundary if available
        if self.charboundary:
            try:
                sentences = self.charboundary.segment(text)
                if sentences:
                    return sentences
            except Exception as e:
                logger.warning(f"CharBoundary segmentation failed: {e}")
        
        # Fallback to NLTK tokenizer
        return self.nltk_tokenizer.tokenize(text)
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal entities using LexNLP or fallback methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of entity types and instances
        """
        entities = {
            'organizations': [],
            'persons': [],
            'locations': [],
            'courts': [],
            'dates': [],
            'amounts': [],
            'citations': [],
            'regulations': [],
            'terms_of_art': []
        }
        
        if LEXNLP_AVAILABLE:
            try:
                # Extract organizations
                entities['organizations'] = list(lexnlp.extract.en.entities.get_organizations(text))
                
                # Extract geopolitical entities
                entities['locations'] = list(lexnlp.extract.en.entities.get_geopolitical(text))
                
                # Extract courts
                entities['courts'] = list(lexnlp.extract.en.entities.get_courts(text))
                
                # Extract dates
                entities['dates'] = [str(d) for d in lexnlp.extract.en.dates.get_dates(text)]
                
                # Extract amounts
                entities['amounts'] = [str(a) for a in lexnlp.extract.en.amounts.get_amounts(text)]
                
                # Extract citations
                entities['citations'] = list(lexnlp.extract.en.citations.get_citations(text))
                
                # Extract regulations
                entities['regulations'] = list(lexnlp.extract.en.regulations.get_regulations(text))
            except Exception as e:
                logger.warning(f"LexNLP entity extraction failed: {e}")
        
        # Use spaCy as fallback or supplement
        if SPACY_AVAILABLE and nlp:
            try:
                doc = nlp(text)
                
                for ent in doc.ents:
                    if ent.label_ == 'ORG' and ent.text not in entities['organizations']:
                        entities['organizations'].append(ent.text)
                    elif ent.label_ == 'PERSON' and ent.text not in entities['persons']:
                        entities['persons'].append(ent.text)
                    elif ent.label_ in ('GPE', 'LOC') and ent.text not in entities['locations']:
                        entities['locations'].append(ent.text)
                    elif ent.label_ == 'DATE' and ent.text not in entities['dates']:
                        entities['dates'].append(ent.text)
            except Exception as e:
                logger.warning(f"spaCy entity extraction failed: {e}")
        
        # Extract legal terms of art using custom patterns
        legal_terms = set()
        term_patterns = [
            r'\b(?:force majeure|mutatis mutandis|de jure|de facto|inter alia|bona fide)\b',
            r'\b(?:prima facie|ex parte|sine qua non|ipso facto|pro rata|status quo)\b',
            r'\b(?:due process|burden of proof|reasonable doubt|estoppel|mens rea|actus reus)\b'
        ]
        
        for pattern in term_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                legal_terms.add(match.group().lower())
        
        entities['terms_of_art'] = sorted(list(legal_terms))
        
        return entities
    
    def extract_definitions(self, text: str) -> List[Dict[str, str]]:
        """
        Extract definitions from legal text using LexNLP or fallback methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of definitions with term and definition
        """
        definitions = []
        
        if LEXNLP_AVAILABLE:
            try:
                # Use LexNLP for definition extraction
                lexnlp_definitions = lexnlp.extract.en.definitions.get_definitions(text)
                
                for definition in lexnlp_definitions:
                    definitions.append({
                        'term': definition.term,
                        'definition': definition.definition
                    })
            except Exception as e:
                logger.warning(f"LexNLP definition extraction failed: {e}")
        
        # Fallback or supplement with regex patterns
        definition_patterns = [
            r'"([^"]+)"\s+(?:means|signifies|refers to|is defined as)\s+([^\.]+)',
            r'the term\s+"([^"]+)"\s+(?:means|signifies|refers to|is defined as)\s+([^\.]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:means|signifies|refers to|is defined as)\s+([^\.]+)'
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                term = match.group(1).strip()
                definition_text = match.group(2).strip()
                
                # Check if this term is already in the definitions
                if not any(d['term'] == term for d in definitions):
                    definitions.append({
                        'term': term,
                        'definition': definition_text
                    })
        
        return definitions
    
    def analyze_sentence_types(self, sentences: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze sentences to categorize them by type (obligation, right, etc.).
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            Dictionary mapping sentence types to lists of sentences
        """
        sentence_types = {
            'obligation': [],
            'right': [],
            'prohibition': [],
            'definition': [],
            'cross_reference': [],
            'temporal': []
        }
        
        for i, sentence in enumerate(sentences):
            # Check each sentence against each pattern
            for sentence_type, pattern in self.patterns.items():
                if pattern.search(sentence):
                    sentence_types[sentence_type].append({
                        'text': sentence,
                        'index': i
                    })
        
        return sentence_types
    
    def analyze_obligation_strength(self, text: str) -> Dict[str, int]:
        """
        Analyze the strength of obligations in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping strength indicators to counts
        """
        strength_indicators = {
            'mandatory': 0,  # shall, must
            'recommended': 0,  # should
            'permissive': 0,  # may
            'conditional': 0  # if, when, provided that
        }
        
        # Count mandatory terms
        mandatory_matches = len(re.findall(r'\b(?:shall|must|required|obligated)\b', text, re.IGNORECASE))
        strength_indicators['mandatory'] = mandatory_matches
        
        # Count recommended terms
        recommended_matches = len(re.findall(r'\b(?:should|ought to|encouraged to|recommended)\b', text, re.IGNORECASE))
        strength_indicators['recommended'] = recommended_matches
        
        # Count permissive terms
        permissive_matches = len(re.findall(r'\b(?:may|can|could|permitted|allowed)\b', text, re.IGNORECASE))
        strength_indicators['permissive'] = permissive_matches
        
        # Count conditional terms
        conditional_matches = len(re.findall(r'\b(?:if|when|where|provided that|subject to|in case|on condition)\b', text, re.IGNORECASE))
        strength_indicators['conditional'] = conditional_matches
        
        return strength_indicators
    
    def _analyze_paragraph(self, paragraph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a paragraph of text.
        
        Args:
            paragraph: Paragraph data to analyze
            
        Returns:
            Analysis results for the paragraph
        """
        paragraph_analysis = {
            'type': paragraph.get('type', 'text'),
            'id': paragraph.get('id', ''),
            'number': paragraph.get('number', ''),
            'sentence_types': {},
            'entities': {},
            'structure': {}  # Add structure field to capture hierarchical content
        }
        
        # Get paragraph content
        content = paragraph.get('content', '')
        paragraph_text = ""
        
        # Handle different content formats
        if isinstance(content, str):
            paragraph_text = content
        elif isinstance(content, list):
            # For bullet points or nested content
            paragraph_text = " ".join(content)
            
            # Preserve hierarchical structure for nested bullets
            if paragraph.get('type') == 'bullets':
                paragraph_analysis['structure']['items'] = []
                for item in content:
                    # Check if item contains nested sub-points
                    if isinstance(item, str) and re.search(r'\([a-z]+\).*?\([i]+\)', item):
                        # Extract main point and sub-points
                        main_match = re.match(r'(\([a-z]+\))(.*?)(\([i]+\))', item)
                        if main_match:
                            main_point = main_match.group(1) + main_match.group(2)
                            sub_points = item[len(main_point):].split('and')
                            paragraph_analysis['structure']['items'].append({
                                'main': main_point.strip(),
                                'sub_items': [sp.strip() for sp in sub_points]
                            })
                        else:
                            paragraph_analysis['structure']['items'].append(item)
                    else:
                        paragraph_analysis['structure']['items'].append(item)
        
        # Check if we already have pre-segmented sentences
        if 'sentences' in paragraph and paragraph['sentences']:
            sentences = paragraph['sentences']
        else:
            # Segment sentences
            sentences = self.segment_sentences(paragraph_text)
        
        # Analyze sentence types
        paragraph_analysis['sentence_types'] = self.analyze_sentence_types(sentences)
        
        # Extract entities
        paragraph_analysis['entities'] = self.extract_legal_entities(paragraph_text)
        
        return paragraph_analysis
    
    def analyze_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a full legal document.
        
        Args:
            document: Document data to analyze
            
        Returns:
            Analysis results
        """
        # Check document completeness
        document_completeness = 'complete'
        
        if not document.get('id') or not document.get('code') or not document.get('title'):
            logger.warning(f"Document appears to be missing key metadata fields")
            document_completeness = 'incomplete'
        
        if not document.get('articles'):
            logger.warning(f"Document {document.get('id', 'unknown')} has no articles")
            document_completeness = 'incomplete'
        elif any(not article.get('paragraphs') for article in document.get('articles', [])):
            logger.warning(f"Document {document.get('id', 'unknown')} has articles with missing paragraphs")
            document_completeness = 'incomplete'
        
        # Check for truncated content
        last_article = document.get('articles', [])[-1] if document.get('articles') else None
        if last_article and last_article.get('paragraphs'):
            last_paragraph = last_article['paragraphs'][-1]
            if isinstance(last_paragraph.get('content'), str) and last_paragraph['content'].endswith(','):
                logger.warning(f"Document {document.get('id', 'unknown')} appears to be truncated")
                document_completeness = 'truncated'
        
        analysis = {
            'id': document.get('id', ''),
            'code': document.get('code', ''),
            'title': document.get('title', ''),
            'metadata': document.get('metadata', {}),
            'sentence_types': {
                'obligation': [],
                'right': [],
                'prohibition': [],
                'definition': [],
                'cross_reference': [],
                'temporal': []
            },
            'entities': {},
            'definitions': [],
            'obligation_strength': {},
            'articles_analysis': [],
            'summary': {},
            'document_completeness': document_completeness
        }
        
        # Extract full text from the document
        full_text = ""
        
        # Add preamble if available
        if 'preamble' in document:
            full_text += document['preamble'] + " "
        
        # Process articles
        for article in document.get('articles', []):
            article_text = ""
            article_analysis = {
                'number': article.get('number', ''),
                'title': article.get('title', ''),
                'sentence_types': {},
                'entities': {},
                'obligation_strength': {},
                'paragraphs_analysis': []
            }
            
            # Process paragraphs
            for paragraph in article.get('paragraphs', []):
                paragraph_text = ""
                
                # Get paragraph content
                content = paragraph.get('content', '')
                if isinstance(content, str):
                    paragraph_text = content
                elif isinstance(content, list):
                    paragraph_text = " ".join(content)
                
                # Add to article text
                article_text += paragraph_text + " "
                
                # Analyze paragraph using pre-segmented sentences if available
                paragraph_analysis = self._analyze_paragraph(paragraph)
                article_analysis['paragraphs_analysis'].append(paragraph_analysis)
            
            # Add to full text
            full_text += article_text + " "
            
            # Use pre-segmented sentences if available or segment manually
            article_sentences = []
            for paragraph in article.get('paragraphs', []):
                if 'sentences' in paragraph and paragraph['sentences']:
                    article_sentences.extend(paragraph['sentences'])
                else:
                    # Get paragraph content
                    content = paragraph.get('content', '')
                    if isinstance(content, str):
                        paragraph_text = content
                    elif isinstance(content, list):
                        paragraph_text = " ".join(content)
                    # Segment the paragraph text
                    article_sentences.extend(self.segment_sentences(paragraph_text))
            
            # Analyze article sentence types
            article_analysis['sentence_types'] = self.analyze_sentence_types(article_sentences)
            
            # Extract entities
            article_analysis['entities'] = self.extract_legal_entities(article_text)
            
            # Analyze obligation strength
            article_analysis['obligation_strength'] = self.analyze_obligation_strength(article_text)
            
            # Add article analysis to document analysis
            analysis['articles_analysis'].append(article_analysis)
            
            # Aggregate sentence types
            for sentence_type, sentences in article_analysis['sentence_types'].items():
                for sentence in sentences:
                    # Add article reference
                    sentence_with_ref = sentence.copy()
                    sentence_with_ref['article'] = article.get('number', '')
                    analysis['sentence_types'][sentence_type].append(sentence_with_ref)
        
        # Analyze full document
        full_text = full_text.strip()
        
        # Extract entities from full text
        analysis['entities'] = self.extract_legal_entities(full_text)
        
        # Extract definitions
        analysis['definitions'] = self.extract_definitions(full_text)
        
        # Analyze obligation strength for full document
        analysis['obligation_strength'] = self.analyze_obligation_strength(full_text)
        
        # Generate summary statistics
        analysis['summary'] = {
            'total_articles': len(document.get('articles', [])),
            'total_obligations': len(analysis['sentence_types']['obligation']),
            'total_rights': len(analysis['sentence_types']['right']),
            'total_prohibitions': len(analysis['sentence_types']['prohibition']),
            'total_definitions': len(analysis['definitions']),
            'obligation_strength': analysis['obligation_strength'],
            'document_completeness': document_completeness
        }
        
        return analysis
    
    def _analyze_file(self, file_path: str, output_dir: str) -> Optional[str]:
        """
        Analyze a single legal document file.
        
        Args:
            file_path: Path to the document file
            output_dir: Directory to save analysis results
            
        Returns:
            Path to the analysis file or None if analysis fails
        """
        try:
            # Load document
            with open(file_path, 'r', encoding='utf-8') as f:
                document_data = json.load(f)
            
            # Handle case where the file contains an array of documents
            if isinstance(document_data, list):
                logger.info(f"File {file_path} contains {len(document_data)} documents")
                
                result_paths = []
                for i, document in enumerate(document_data):
                    try:
                        # Analyze document
                        analysis = self.analyze_document(document)
                        
                        # Generate output filename
                        doc_id = document.get('id', f'doc_{i}')
                        doc_code = document.get('code', '')
                        output_filename = f"{doc_code}_{doc_id}_analysis.json"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Save analysis
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(analysis, f, indent=2, ensure_ascii=False)
                        
                        logger.info(f"Saved analysis to {output_path}")
                        result_paths.append(output_path)
                    except Exception as e:
                        logger.error(f"Error analyzing document {doc_id}: {str(e)}")
                
                return result_paths[0] if result_paths else None
                
            else:
                # Single document
                document = document_data
                # Analyze document
                analysis = self.analyze_document(document)
                
                # Generate output filename
                basename = os.path.basename(file_path)
                doc_id = document.get('id', '')
                doc_code = document.get('code', '')
                if doc_id and doc_code:
                    output_filename = f"{doc_code}_{doc_id}_analysis.json"
                else:
                    output_filename = basename.replace('.json', '_analysis.json')
                output_path = os.path.join(output_dir, output_filename)
                
                # Save analysis
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved analysis to {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error analyzing document {file_path}: {str(e)}")
            return None
    
    def analyze_directory(self, input_dir: str, output_dir: str, parallel: bool = False) -> List[str]:
        """
        Analyze all legal documents in a directory.
        
        Args:
            input_dir: Directory containing legal documents
            output_dir: Directory to save analysis results
            parallel: Whether to process files in parallel
            
        Returns:
            List of paths to analysis files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all JSON files
        json_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(json_files)} JSON files to analyze")
        
        output_files = []
        
        if parallel and len(json_files) > 1:
            # Process in parallel
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_file = {
                    executor.submit(self._analyze_file, file_path, output_dir): file_path
                    for file_path in json_files
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(json_files), desc="Analyzing documents"):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            if isinstance(result, list):
                                output_files.extend(result)
                            else:
                                output_files.append(result)
                    except Exception as e:
                        logger.error(f"Error analyzing {file_path}: {str(e)}")
        else:
            # Process sequentially
            for file_path in tqdm(json_files, desc="Analyzing documents"):
                result = self._analyze_file(file_path, output_dir)
                if result:
                    if isinstance(result, list):
                        output_files.extend(result)
                    else:
                        output_files.append(result)
        
        return output_files
    
    def generate_report(self, analysis_dir: str, output_file: str, format: str = 'json') -> str:
        """
        Generate a consolidated report from multiple analyses.
        
        Args:
            analysis_dir: Directory containing analysis files
            output_file: Path to save the report
            format: Report format ('json' or 'csv')
            
        Returns:
            Path to the report file
        """
        # Find all analysis files
        analysis_files = []
        for root, _, files in os.walk(analysis_dir):
            for file in files:
                if file.endswith('_analysis.json'):
                    analysis_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(analysis_files)} analysis files")
        
        # Load all analyses
        analyses = []
        for file_path in analysis_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error loading analysis {file_path}: {str(e)}")
        
        # Generate report
        report = {
            'generated_date': datetime.now().isoformat(),
            'total_documents': len(analyses),
            'documents': [],
            'aggregate_statistics': {
                'total_articles': 0,
                'total_obligations': 0,
                'total_rights': 0,
                'total_prohibitions': 0,
                'total_definitions': 0,
                'obligation_strength': {
                    'mandatory': 0,
                    'recommended': 0,
                    'permissive': 0,
                    'conditional': 0
                },
                'document_completeness': {
                    'complete': 0,
                    'incomplete': 0,
                    'truncated': 0
                }
            }
        }
        
        # Process each analysis
        for analysis in analyses:
            # Add document summary
            document_summary = {
                'code': analysis.get('code', ''),
                'title': analysis.get('title', ''),
                'summary': analysis.get('summary', {})
            }
            
            report['documents'].append(document_summary)
            
            # Update aggregate statistics
            summary = analysis.get('summary', {})
            report['aggregate_statistics']['total_articles'] += summary.get('total_articles', 0)
            report['aggregate_statistics']['total_obligations'] += summary.get('total_obligations', 0)
            report['aggregate_statistics']['total_rights'] += summary.get('total_rights', 0)
            report['aggregate_statistics']['total_prohibitions'] += summary.get('total_prohibitions', 0)
            report['aggregate_statistics']['total_definitions'] += summary.get('total_definitions', 0)
            
            # Update obligation strength
            strength = summary.get('obligation_strength', {})
            report['aggregate_statistics']['obligation_strength']['mandatory'] += strength.get('mandatory', 0)
            report['aggregate_statistics']['obligation_strength']['recommended'] += strength.get('recommended', 0)
            report['aggregate_statistics']['obligation_strength']['permissive'] += strength.get('permissive', 0)
            report['aggregate_statistics']['obligation_strength']['conditional'] += strength.get('conditional', 0)
            
            # Update document completeness stats
            completeness = summary.get('document_completeness', 'complete')
            report['aggregate_statistics']['document_completeness'][completeness] += 1
        
        # Save report
        if format == 'csv':
            # Save as CSV
            self._save_report_csv(report, output_file)
        else:
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated report and saved to {output_file}")
        return output_file
    
    def _save_report_csv(self, report: Dict[str, Any], output_file: str):
        """
        Save report as CSV.
        
        Args:
            report: Report data
            output_file: Path to save the report
        """
        # Save document summaries
        documents_file = output_file.replace('.csv', '_documents.csv')
        with open(documents_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Code', 'Title', 'Articles', 'Obligations', 'Rights', 'Prohibitions', 'Definitions', 'Completeness'])
            
            for document in report['documents']:
                summary = document.get('summary', {})
                writer.writerow([
                    document.get('code', ''),
                    document.get('title', ''),
                    summary.get('total_articles', 0),
                    summary.get('total_obligations', 0),
                    summary.get('total_rights', 0),
                    summary.get('total_prohibitions', 0),
                    summary.get('total_definitions', 0),
                    summary.get('document_completeness', 'complete')
                ])
        
        # Save aggregate statistics
        stats_file = output_file.replace('.csv', '_stats.csv')
        with open(stats_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            
            stats = report['aggregate_statistics']
            writer.writerow(['Total Documents', report['total_documents']])
            writer.writerow(['Total Articles', stats['total_articles']])
            writer.writerow(['Total Obligations', stats['total_obligations']])
            writer.writerow(['Total Rights', stats['total_rights']])
            writer.writerow(['Total Prohibitions', stats['total_prohibitions']])
            writer.writerow(['Total Definitions', stats['total_definitions']])
            writer.writerow(['Mandatory Obligations', stats['obligation_strength']['mandatory']])
            writer.writerow(['Recommended Actions', stats['obligation_strength']['recommended']])
            writer.writerow(['Permissive Provisions', stats['obligation_strength']['permissive']])
            writer.writerow(['Conditional Provisions', stats['obligation_strength']['conditional']])
            writer.writerow(['Complete Documents', stats['document_completeness']['complete']])
            writer.writerow(['Incomplete Documents', stats['document_completeness']['incomplete']])
            writer.writerow(['Truncated Documents', stats['document_completeness']['truncated']])


def main():
    """Main function to run the analyzer."""
    parser = argparse.ArgumentParser(description="Analyze legal documents using specialized NLP")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing legal documents")
    parser.add_argument("--output", type=str, default="analysis", help="Output directory for analysis results")
    parser.add_argument("--report", type=str, help="Generate consolidated report")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Report format")
    parser.add_argument("--parallel", action="store_true", help="Process files in parallel")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = LegalNLPAnalyzer()
    
    # Analyze documents
    output_files = analyzer.analyze_directory(args.input, args.output, args.parallel)
    
    # Generate report if requested
    if args.report:
        analyzer.generate_report(args.output, args.report, args.format)


if __name__ == "__main__":
    main()