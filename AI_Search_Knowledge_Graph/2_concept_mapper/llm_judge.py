"""LLM-as-Judge for concept evaluation with retry logic."""
import json
import logging
import re
from typing import List, Dict, Tuple, Set

import requests

from config import config

logger = logging.getLogger(__name__)


class LLMJudge:
    """LLM-based judge for evaluating concept relevance."""

    def __init__(self):
        self.endpoint = config.foundry_endpoint
        self.api_key = config.foundry_api_key
        self.deployment = config.foundry_deployment
        self.api_version = config.foundry_api_version
        self.timeout = config.llm_timeout

        # HTTP session
        self._http = requests.Session()
        self._base_url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}"
        self._headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

    def judge_concepts(
        self,
        doc_title: str,
        doc_summary: str,
        candidate_concepts: List[Dict],
        top_k: int = 20
    ) -> Tuple[List[Dict], Dict]:
        """
        Evaluate all candidate concepts and return top_k with scores.

        Implements retry logic:
        - Attempts 1-2: Original prompt
        - Attempts 3-5: Modified prompt emphasizing distinct concepts
        - Consolidation if all attempts fail

        Args:
            doc_title: Document title
            doc_summary: Document summary text
            candidate_concepts: List of {concept_id, label, description, score}
            top_k: Number of top concepts to return

        Returns:
            Tuple of (evaluated_concepts, metadata)
            evaluated_concepts: List of {concept_id, label, score, rationale}
            metadata: Dict with attempts_made, consolidation_used, final_count
        """
        all_attempts = []
        valid_concept_ids = {c['concept_id'] for c in candidate_concepts}

        for attempt in range(1, config.max_retries + 1):
            try:
                prompt = self._create_prompt(
                    doc_title, doc_summary, candidate_concepts, top_k, attempt
                )
                response = self._call_llm(prompt)

                if not response:
                    logger.warning(f"Attempt {attempt}: Empty response from LLM")
                    all_attempts.append({
                        'attempt': attempt,
                        'success': False,
                        'error': 'Empty response',
                        'concepts': []
                    })
                    continue

                # Parse and validate response
                parsed = self._parse_response(response)
                validated = self._validate_concepts(parsed, valid_concept_ids)

                all_attempts.append({
                    'attempt': attempt,
                    'success': True,
                    'count': len(validated),
                    'concepts': validated
                })

                if len(validated) >= config.min_valid_concepts:
                    logger.info(f"Attempt {attempt} succeeded with {len(validated)} valid concepts")
                    return validated[:top_k], {
                        'attempts_made': attempt,
                        'consolidation_used': False,
                        'final_count': len(validated[:top_k])
                    }
                else:
                    logger.warning(
                        f"Attempt {attempt}: Only {len(validated)}/{config.min_valid_concepts} valid concepts"
                    )

            except Exception as e:
                logger.error(f"Attempt {attempt} failed with exception: {e}")
                all_attempts.append({
                    'attempt': attempt,
                    'success': False,
                    'error': str(e),
                    'concepts': []
                })

        # All attempts exhausted - consolidation strategy
        consolidated = self._consolidate_attempts(all_attempts)
        logger.warning(f"Consolidation produced {len(consolidated)} concepts from {len(all_attempts)} attempts")

        return consolidated[:top_k], {
            'attempts_made': config.max_retries,
            'consolidation_used': True,
            'final_count': len(consolidated[:top_k]),
            'insufficient': len(consolidated) < config.min_valid_concepts
        }

    def _create_prompt(
        self,
        title: str,
        summary: str,
        concepts: List[Dict],
        top_k: int,
        attempt: int
    ) -> str:
        """Create evaluation prompt, modified for retries."""
        # Build concepts list
        concepts_text = ""
        for i, c in enumerate(concepts, 1):
            desc = c.get('description', '')[:200]
            concepts_text += f"{i}. ID: {c['concept_id']}\n"
            concepts_text += f"   Label: {c['label']}\n"
            if desc:
                concepts_text += f"   Description: {desc}...\n"
            concepts_text += "\n"

        base_prompt = f"""TASK: Evaluate the relevance of {len(concepts)} candidate concepts to this document.

DOCUMENT:
Title: {title}
Summary: {summary[:1500]}

CANDIDATE CONCEPTS:
{concepts_text}

REQUIREMENTS:
1. Evaluate ALL {len(concepts)} concepts for relevance to the document
2. Score each concept 0-10 based on how relevant it is to the document content
3. Provide a brief rationale (1-2 sentences) for each score
4. Return the top {top_k} concepts ranked by score (highest first)

SCORING GUIDE:
- 9-10: Highly relevant - concept is a core topic of the document
- 7-8: Relevant - concept is clearly discussed in the document
- 5-6: Somewhat relevant - concept has tangential connection
- 3-4: Marginally relevant - weak connection
- 0-2: Not relevant - no meaningful connection

OUTPUT FORMAT (JSON only):
{{
  "evaluated_concepts": [
    {{
      "concept_id": "exact_id_from_list",
      "label": "concept label",
      "score": 8.5,
      "rationale": "Brief explanation of relevance"
    }},
    ... (top {top_k} concepts by score)
  ]
}}

IMPORTANT: Return ONLY valid JSON. Use exact concept_ids from the candidate list above."""

        # Modify prompt for attempts >= threshold
        if attempt >= config.retry_prompt_change_threshold:
            emphasis = f"""
CRITICAL REQUIREMENTS FOR THIS ATTEMPT:
- You MUST return exactly {top_k} DISTINCT concepts (no duplicate concept_ids)
- Each concept_id MUST exist in the candidate list above - do not invent new IDs
- Provide clear scores in 0-10 range for each concept
- Double-check your response contains {top_k} unique, valid concepts

"""
            return emphasis + base_prompt

        return base_prompt

    def _call_llm(self, prompt: str) -> str:
        """Call Azure Foundry LLM API."""
        url = f"{self._base_url}/chat/completions?api-version={self.api_version}"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }

        logger.debug(f"Calling LLM API: {url}")
        resp = self._http.post(
            url,
            headers=self._headers,
            json=payload,
            timeout=(10, self.timeout)
        )

        if resp.status_code >= 300:
            raise RuntimeError(f"LLM API error: {resp.status_code} - {resp.text[:500]}")

        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            logger.debug(f"LLM response length: {len(content)} chars")
            return content
        return ""

    def _parse_response(self, response: str) -> List[Dict]:
        """Parse JSON response from LLM."""
        try:
            # Try to find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed.get("evaluated_concepts", [])
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
        return []

    def _validate_concepts(
        self,
        concepts: List[Dict],
        valid_ids: Set[str]
    ) -> List[Dict]:
        """Validate and deduplicate concepts."""
        seen = set()
        validated = []

        for c in concepts:
            cid = str(c.get('concept_id', '')).strip()
            score = c.get('score', 0)

            # Skip invalid concept IDs
            if not cid or cid not in valid_ids:
                logger.debug(f"Skipping invalid concept_id: {cid}")
                continue

            # Skip duplicates
            if cid in seen:
                logger.debug(f"Skipping duplicate concept_id: {cid}")
                continue

            # Validate score
            try:
                score = float(score)
                if not (0 <= score <= 10):
                    logger.debug(f"Score out of range for {cid}: {score}")
                    score = max(0, min(10, score))  # Clamp to valid range
            except (TypeError, ValueError):
                logger.debug(f"Invalid score for {cid}: {score}")
                continue

            validated.append({
                'concept_id': cid,
                'label': c.get('label', ''),
                'score': score,
                'rationale': c.get('rationale', '')
            })
            seen.add(cid)

        # Sort by score descending
        validated.sort(key=lambda x: x['score'], reverse=True)
        return validated

    def _consolidate_attempts(self, attempts: List[Dict]) -> List[Dict]:
        """Merge concepts from all attempts, keeping highest scores."""
        best = {}
        for attempt in attempts:
            for c in attempt.get('concepts', []):
                cid = c['concept_id']
                if cid not in best or c['score'] > best[cid]['score']:
                    best[cid] = c

        consolidated = list(best.values())
        consolidated.sort(key=lambda x: x['score'], reverse=True)
        logger.debug(f"Consolidated {len(consolidated)} unique concepts from all attempts")
        return consolidated

    def cleanup(self):
        """Clean up HTTP session."""
        self._http.close()
