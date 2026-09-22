"""Enhanced education extraction with improved level and major detection."""
import re
import json
from typing import Dict, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from .config import LLM_MODEL, EnhancedExtractionConfig, MAJOR_TAXONOMY_FILE
import torch

class EnhancedEducationExtractor:
    """Extract education level and major with enhanced accuracy."""

    def __init__(self, config: EnhancedExtractionConfig = None):
        """Initialize enhanced education extractor."""
        self.config = config or EnhancedExtractionConfig()

        # Load major taxonomy
        self.major_taxonomy = self._load_major_taxonomy()
        self.major_set = set(self.major_taxonomy)

        # Lazy load LLM
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_pipe = None
        self.llm_load_failed = False

    def _load_major_taxonomy(self) -> List[str]:
        """Load list of valid majors."""
        try:
            with open(MAJOR_TAXONOMY_FILE, 'r', encoding='utf-8') as f:
                taxonomy_data = json.load(f)

            majors = []
            for category, major_list in taxonomy_data.items():
                majors.extend([m.lower() for m in major_list])

            return majors
        except FileNotFoundError:
            print(f"Warning: Major taxonomy file not found")
            return []

    def _load_llm(self):
        """Lazy load LLM model. Tries once per process: on failure this is
        never retried (it would otherwise reload and fail again for every
        job whose regex extraction comes up empty)."""
        if self.llm_pipe is None and not self.llm_load_failed:
            device = 0 if torch.cuda.is_available() else -1
            print(f"Loading LLM model: {LLM_MODEL} on device: {'GPU' if device == 0 else 'CPU'}...")
            try:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
                self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
                self.llm_pipe = pipeline(
                    "text2text-generation",
                    model=self.llm_model,
                    tokenizer=self.llm_tokenizer,
                    device=device
                )
                print("LLM model loaded successfully!")
            except Exception as e:
                print(f"Failed to load LLM model, disabling LLM fallback for this run: {e}")
                self.llm_load_failed = True

    def extract(self, text: str) -> Dict[str, Optional[str]]:
        """Extract education level and major with enhanced accuracy."""
        if not text:
            return {"level": None, "major": None}

        text_lower = text.lower()

        # Step 1: Extract education level (primary: regex, fallback: LLM)
        level = self._extract_level_enhanced(text_lower)

        # Step 2: Extract major (multi-strategy)
        major = self._extract_major_enhanced(text_lower, text)

        # Step 3: If either failed, try LLM fallback
        if not level or not major:
            level_llm, major_llm = self._extract_with_llm(text)
            level = level or level_llm
            major = major or major_llm

        # Step 4: Clean and validate major
        if major:
            major = self._clean_major(major)
            major = self._validate_major(major)

        return {"level": level, "major": major}

    def _extract_level_enhanced(self, text: str) -> Optional[str]:
        """Enhanced education level extraction with context awareness."""
        # Direct pattern matching (highest confidence)
        for level_name, pattern in self.config.EDUCATION_LEVEL_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return level_name

        # Context-based inference
        # If mentions "degree" but no level, check for clues
        if "degree" in text:
            # Check for year indicators
            if any(year in text for year in ["3 year", "4 year", "4-year", "three year", "four year"]):
                return "bachelor's degree"
            if any(year in text for year in ["2 year", "2-year", "two year"]):
                return "associate"
            if any(term in text for term in ["graduate degree", "postgraduate", "post-graduate"]):
                if "phd" in text or "doctoral" in text:
                    return "phd"
                return "master's degree"

        # Check for implicit bachelor's (common in requirements)
        if any(phrase in text for phrase in ["bachelor", "undergraduate", "4 years"]):
            return "bachelor's degree"

        return None

    def _extract_major_enhanced(self, text_lower: str, text_original: str) -> Optional[str]:
        """Enhanced major extraction with multiple strategies."""
        candidates = []

        # Strategy 1: Regex pattern matching (multiple patterns)
        for pattern in self.config.MAJOR_EXTRACTION_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else (match[1] if len(match) > 1 else "")

                cleaned = self._clean_major(match)
                if cleaned and len(cleaned) > 2:
                    candidates.append((cleaned, 'regex'))

        # Strategy 2: Look for major taxonomy matches in text
        for major in self.major_taxonomy:
            if len(major) > 3:  # Avoid very short majors
                # Check if major appears as whole word
                if re.search(rf'\b{re.escape(major)}\b', text_lower):
                    candidates.append((major, 'taxonomy'))

        # Strategy 3: Common field indicators
        field_indicators = [
            (r'(?:knowledge|experience|background)\s+in\s+([a-z][a-z\s&/,-]{3,40})', 'experience'),
            (r'(?:specialized|specialization)\s+in\s+([a-z][a-z\s&/,-]{3,40})', 'specialization'),
        ]

        for pattern, source in field_indicators:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                cleaned = self._clean_major(match)
                if cleaned and len(cleaned) > 2:
                    candidates.append((cleaned, source))

        # Rank candidates
        if candidates:
            # Remove duplicates, prefer taxonomy matches
            seen = set()
            ranked = []

            for candidate, source in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    # Score by source priority
                    score = 0
                    if source == 'taxonomy':
                        score = 10
                    elif source == 'regex':
                        score = 8
                    elif source == 'specialization':
                        score = 6
                    else:
                        score = 5

                    # Bonus if in taxonomy
                    if candidate in self.major_set:
                        score += 5

                    ranked.append((candidate, score))

            # Sort by score and return best
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked[0][0] if ranked else None

        return None

    def _extract_with_llm(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract using LLM fallback."""
        if self.llm_pipe is None:
            self._load_llm()

        if self.llm_pipe is None:
            return (None, None)

        level = None
        major = None

        try:
            # Extract level
            level_prompt = (
                "What is the required education level in this job requirement? "
                "Answer with ONLY ONE of these: high school, associate, bachelor's degree, "
                "master's degree, PhD, or none.\\n\\n"
                f"Text: {text[:500]}"
            )

            level_result = self.llm_pipe(level_prompt, max_new_tokens=20, num_return_sequences=1)
            level_text = level_result[0]['generated_text'].strip().lower()

            # Parse level
            if 'bachelor' in level_text or 'ba' in level_text or 'bs' in level_text or 'undergraduate' in level_text:
                level = "bachelor's degree"
            elif 'master' in level_text or 'ms' in level_text or 'ma' in level_text or 'postgraduate' in level_text:
                level = "master's degree"
            elif 'phd' in level_text or 'doctorate' in level_text or 'doctoral' in level_text:
                level = "phd"
            elif 'high school' in level_text or 'secondary' in level_text:
                level = "high school"
            elif 'associate' in level_text:
                level = "associate"

            # Extract major
            major_prompt = (
                "What is the required field of study or major in this job requirement? "
                "Answer with ONLY the field name (e.g., 'Computer Science', 'Engineering'), "
                "or 'none' if not specified.\\n\\n"
                f"Text: {text[:500]}"
            )

            major_result = self.llm_pipe(major_prompt, max_new_tokens=30, num_return_sequences=1)
            major_text = major_result[0]['generated_text'].strip()

            if major_text and major_text.lower() not in ['none', 'not specified', 'any', 'n/a', 'not mentioned']:
                major = self._clean_major(major_text)

        except Exception as e:
            print(f"LLM extraction error: {e}")

        return (level, major)

    def _clean_major(self, major: str) -> Optional[str]:
        """Clean and normalize major field."""
        if not major:
            return None

        major = major.lower().strip()

        # Remove noise phrases
        noise_phrases = [
            'or related field', 'or related', 'or equivalent', 'or similar',
            'or higher', 'or above', 'preferred', 'required', 'degree in',
            'bachelor in', 'bachelor of', 'master in', 'master of',
            'major in', 'field of', 'study in', 'diploma in',
            'or any related', 'and related', 'related field', 'related discipline'
        ]

        for noise in noise_phrases:
            major = re.sub(rf'\b{re.escape(noise)}\b', '', major, flags=re.IGNORECASE)

        major = major.strip()

        # Remove trailing/leading punctuation
        major = re.sub(r'^[,;.\s/&-]+|[,;.\s/&-]+$', '', major)

        # Remove redundant spaces
        major = re.sub(r'\s+', ' ', major)

        # Skip if too short or too long
        if len(major) < 3 or len(major) > 60:
            return None

        # Reject if it's a technical tool/software name (not a major)
        tech_tools_keywords = [
            'office', 'outlook', 'excel', 'word', 'powerpoint', 'autocad', 'photoshop',
            'illustrator', 'python', 'java', 'javascript', 'sql', 'html', 'css',
            'react', 'vue', 'angular', 'node', 'django', 'flask', 'aws', 'azure',
            'docker', 'kubernetes', 'git', 'linux', 'windows', 'mac', 'ios', 'android'
        ]

        # Check if major is just a tech tool name or list of tools
        major_words = set(major.split())
        if any(tool in major_words for tool in tech_tools_keywords):
            return None

        # Reject if contains multiple commas (likely a list of tools)
        if major.count(',') > 1:
            return None

        # Skip if it's generic
        generic_terms = ['any', 'related', 'all', 'various', 'other', 'relevant', 'appropriate', 'similar']
        if major in generic_terms:
            return None

        # Handle compound majors (e.g., "design/architect/information technology")
        if '/' in major or '&' in major:
            # Keep compound but validate parts
            parts = re.split(r'[/&]', major)
            valid_parts = []
            for part in parts:
                part = part.strip()
                if len(part) > 2 and part not in generic_terms:
                    valid_parts.append(part)

            if valid_parts:
                if len(valid_parts) == 1:
                    return valid_parts[0]
                else:
                    # Keep compound form but clean
                    return '/'.join(valid_parts[:3])  # Max 3 parts

        return major if len(major) > 2 else None

    def _validate_major(self, major: str) -> Optional[str]:
        """Validate and potentially improve major using taxonomy."""
        if not major:
            return None

        # Exact match in taxonomy
        if major in self.major_set:
            return major

        # Fuzzy match - check if any taxonomy item is in the major
        for taxonomy_major in self.major_taxonomy:
            if taxonomy_major in major or major in taxonomy_major:
                if len(taxonomy_major) > 3:
                    return taxonomy_major  # Prefer taxonomy version

        # Partial match - check if major contains common field names
        common_fields = {
            'engineering': 'engineering',
            'computer': 'computer science',
            'business': 'business administration',
            'design': 'design',
            'architecture': 'architecture',
            'marketing': 'marketing',
            'finance': 'finance',
            'accounting': 'accounting',
            'management': 'management',
            'education': 'education',
            'science': 'science',
        }

        for key, value in common_fields.items():
            if key in major.lower():
                return value

        # Return as-is if reasonable
        return major

    def extract_batch(self, texts: List[str]) -> List[Dict[str, Optional[str]]]:
        """Extract education from multiple texts."""
        return [self.extract(text) for text in texts]
