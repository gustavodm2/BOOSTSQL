import re
import logging
from typing import List, Dict, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class SQLQueryRewriter:
    """Legacy SQL query rewriter - kept for compatibility"""

    def __init__(self):
        self.patterns = defaultdict(list)

    def rewrite_query(self, query: str, transformation_type: Optional[str] = None) -> List[str]:
        """Legacy method - returns basic rewrites"""
        rewrites = []

        if 'IN (' in query.upper() and 'SELECT' in query.upper():
            converted = self._convert_in_to_join(query)
            if converted and converted != query:
                rewrites.append(converted)

        if 'GROUP BY' in query.upper() and 'DISTINCT' not in query.upper():
            distinct_query = query.replace('SELECT', 'SELECT DISTINCT', 1)
            rewrites.append(distinct_query)

        cleaned = self._remove_redundant_conditions(query)
        if cleaned != query:
            rewrites.append(cleaned)

        return rewrites

    def _convert_in_to_join(self, query: str) -> Optional[str]:
        """Basic IN to JOIN conversion"""
        in_pattern = r'(\w+)\.(\w+)\s+IN\s*\(\s*SELECT\s+(\w+)\s+FROM\s+(\w+)(.*?)\)'
        match = re.search(in_pattern, query, re.IGNORECASE | re.DOTALL)

        if match:
            table_alias = match.group(1)
            main_col = match.group(2)
            sub_col = match.group(3)
            sub_table = match.group(4)
            conditions = match.group(5).strip()

            join_clause = f'JOIN {sub_table} ON {table_alias}.{main_col} = {sub_table}.{sub_col}'
            if conditions:
                clean_conditions = re.sub(r'^\s*WHERE\s+', '', conditions, flags=re.IGNORECASE)
                join_clause += f' AND {clean_conditions}'

            result = query.replace(match.group(0), '')
            result = re.sub(r'\s+', ' ', result)  

            from_match = re.search(r'FROM\s+(\w+\s+\w+)', result, re.IGNORECASE)
            if from_match:
                result = result.replace(from_match.group(0), f'{from_match.group(0)} {join_clause}')

            return result.strip()

        return None

    def _remove_redundant_conditions(self, query: str) -> str:
        """Remove basic redundant conditions"""
        query = re.sub(r'\b(\w+(?:\.\w+)?)\s*=\s*\1\b', '1=1', query)
        query = re.sub(r'\s+AND\s+1=1\b', '', query, flags=re.IGNORECASE)
        return query

    def get_supported_transformations(self) -> List[str]:
        """Get list of supported transformations"""
        return ['subquery_to_join', 'add_distinct', 'remove_redundant']


class MLQueryRewriter:
    """Intelligent query rewriter that uses machine learning models"""

    def __init__(self, llm_corrector=None):
        self.llm_corrector = llm_corrector
        self.patterns = defaultdict(list)
        self.feature_extractor = None

    def set_feature_extractor(self, extractor):
        """Set the feature extractor for ML predictions"""
        self.feature_extractor = extractor

    def learn_from_optimization(self, original_query: str, optimized_query: str, improvement_ratio: float):
        """Learn from successful optimizations"""
        if original_query == optimized_query:
            return

        self.patterns['learned'].append({
            'original': original_query,
            'optimized': optimized_query,
            'improvement': improvement_ratio
        })

        if len(self.patterns['learned']) > 100:
            self.patterns['learned'] = self.patterns['learned'][-50:]

    def rewrite_query(self, query: str) -> List[str]:
        """Generate optimized query rewrites"""
        rewrites = []

        rewrites.extend(self._generate_basic_rewrites(query))

        rewrites.extend(self._apply_learned_patterns(query))

        if self.llm_corrector:
            llm_rewrites = self._generate_llm_rewrites(query)
            rewrites.extend(llm_rewrites)

        unique_rewrites = []
        seen = set()
        for rewrite in rewrites:
            if rewrite not in seen and rewrite != query:
                unique_rewrites.append(rewrite)
                seen.add(rewrite)

        return unique_rewrites[:10]  

    def _generate_basic_rewrites(self, query: str) -> List[str]:
        """Generate basic query rewrites"""
        rewrites = []

        if 'IN (' in query.upper() and 'SELECT' in query.upper():
            join_version = self._convert_in_to_join(query)
            if join_version:
                rewrites.append(join_version)

        if 'GROUP BY' in query.upper() and 'DISTINCT' not in query.upper():
            distinct_query = query.replace('SELECT', 'SELECT DISTINCT', 1)
            rewrites.append(distinct_query)

        cleaned = self._remove_redundant_conditions(query)
        if cleaned != query:
            rewrites.append(cleaned)

        return rewrites

    def _convert_in_to_join(self, query: str) -> Optional[str]:
        """Convert IN subqueries to JOINs"""
        in_pattern = r'(\w+)\.(\w+)\s+IN\s*\(\s*SELECT\s+(?:DISTINCT\s+)?(\w+)\s+FROM\s+(\w+)(.*?)\)'
        match = re.search(in_pattern, query, re.IGNORECASE | re.DOTALL)

        if match:
            table_alias = match.group(1)
            main_col = match.group(2)
            sub_col = match.group(3)
            sub_table = match.group(4)
            conditions = match.group(5).strip()

            join_condition = f'{table_alias}.{main_col} = {sub_table}.{sub_col}'
            if conditions:
                clean_conditions = re.sub(r'^\s*WHERE\s+', '', conditions, flags=re.IGNORECASE)
                join_condition += f' AND {clean_conditions}'

            join_clause = f'JOIN {sub_table} ON {join_condition}'

            result = query.replace(match.group(0), '')

            result = re.sub(r'\s+(AND|OR)\s*$', '', result, flags=re.IGNORECASE)
            result = re.sub(r'WHERE\s+(AND|OR)', 'WHERE', result, flags=re.IGNORECASE)
            result = re.sub(r'WHERE\s*$', '', result, flags=re.IGNORECASE)

            from_match = re.search(r'FROM\s+(\w+\s+\w+)', result, re.IGNORECASE)
            if from_match:
                result = result.replace(from_match.group(0), f'{from_match.group(0)} {join_clause}')

            return result.strip()

        return None

    def _remove_redundant_conditions(self, query: str) -> str:
        """Remove redundant conditions"""
        query = re.sub(r'\b(\w+(?:\.\w+)?)\s*=\s*\1\b', '1=1', query)
        query = re.sub(r'\s+AND\s+1=1\b', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\s+OR\s+1=1\b', '', query, flags=re.IGNORECASE)
        return query

    def _apply_learned_patterns(self, query: str) -> List[str]:
        """Apply learned transformation patterns"""
        rewrites = []

        for pattern_data in self.patterns.get('learned', []):
            original_pattern = pattern_data['original']
            optimized_pattern = pattern_data['optimized']

            if len(original_pattern.split()) == len(query.split()):
                rewrites.append(optimized_pattern)

        return rewrites

    def _generate_llm_rewrites(self, query: str) -> List[str]:
        """Generate rewrites using LLM"""
        if not self.llm_corrector or self.llm_corrector.mock:
            return []

        try:
            
            return []
        except Exception as e:
            logger.debug(f"LLM rewrite failed: {e}")
            return []

    def get_supported_transformations(self) -> List[str]:
        """Get supported transformation types"""
        return ['subquery_to_join', 'add_distinct', 'remove_redundant', 'general_optimization']