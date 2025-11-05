from typing import Dict, List, Any, Set
import re
import logging

logger = logging.getLogger(__name__)

try:
    import sqlparse
    from sqlparse import sql as sqlparse_sql
    HAS_SQLPARSE = True
except ImportError:
    HAS_SQLPARSE = False
    logger.warning("sqlparse not available, using fallback feature extraction")

class SQLFeatureExtractor:
    

    def __init__(self):
        self.feature_names = [
            'estimated_complexity_score',
            'num_tables',
            'num_joins',
            'has_aggregation',
            'num_subqueries',
            'has_window_functions',
            'num_conditions',
            'nested_level',
            'query_length'
        ]

        self.aggregation_functions = {
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'VARIANCE'
        }

        self.window_functions = {
            'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE', 'LAG', 'LEAD',
            'FIRST_VALUE', 'LAST_VALUE', 'CUME_DIST', 'PERCENT_RANK'
        }

    def extract_features(self, query: str) -> Dict[str, float]:
        
        try:
            query = self._normalize_query(query)

            features = {}

            features['query_length'] = float(len(query))

            tree_features = self._analyze_parse_tree(query)
            features.update(tree_features)

            features['estimated_complexity_score'] = self._calculate_complexity_score(features)

            return features

        except Exception as e:
            logger.warning(f"Feature extraction failed for query: {e}")
            return self._get_default_features()

    def _normalize_query(self, query: str) -> str:
        
        query = re.sub(r'\s+', ' ', query.strip())
        return query

    def _get_default_features(self) -> Dict[str, float]:
        
        return {name: 0.0 for name in self.feature_names}

    def _analyze_parse_tree(self, query: str) -> Dict[str, float]:
        
        features = {
            'num_tables': 0.0,
            'num_joins': 0.0,
            'has_aggregation': 0.0,
            'num_subqueries': 0.0,
            'has_window_functions': 0.0,
            'num_conditions': 0.0,
            'nested_level': 0.0
        }

        query_str = query.upper()

        features['num_tables'] = self._count_tables(query)

        features['num_joins'] = self._count_joins(query_str)

        features['has_aggregation'] = 1.0 if self._has_aggregations(query_str) else 0.0

        features['num_subqueries'] = self._count_subqueries(query)

        features['has_window_functions'] = 1.0 if self._has_window_functions(query_str) else 0.0

        features['num_conditions'] = self._count_conditions(query_str)

        features['nested_level'] = self._calculate_nesting_level(query)

        return features

    def _count_tables(self, query: str) -> float:
        
        try:
            query_str = query.upper()

            from_matches = re.findall(r'\bFROM\s+(\w+)', query_str, re.IGNORECASE)
            join_matches = re.findall(r'\bJOIN\s+(\w+)', query_str, re.IGNORECASE)

            all_tables = set(from_matches + join_matches)

            return float(len(all_tables))
        except:
            return 0.0

    def _count_joins(self, query_str: str) -> float:
        
        join_keywords = ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN']
        count = 0

        for keyword in join_keywords:
            count += query_str.count(keyword)

        return float(count)

    def _has_aggregations(self, query_str: str) -> bool:
        
        for func in self.aggregation_functions:
            if func in query_str:
                return True
        return False

    def _count_subqueries(self, parsed_query) -> float:
        
        try:
            query_str = str(parsed_query)
            select_count = query_str.upper().count('SELECT')
            subquery_count = max(0, select_count - 1)
            return float(subquery_count)
        except:
            return 0.0

    def _has_window_functions(self, query_str: str) -> bool:
        
        if 'OVER' in query_str:
            return True

        for func in self.window_functions:
            if func in query_str:
                return True

        return False

    def _count_conditions(self, query_str: str) -> float:
        
        condition_keywords = ['WHERE', 'HAVING', 'ON']
        count = 0

        for keyword in condition_keywords:
            count += query_str.count(keyword)

        where_section = self._extract_where_section(query_str)
        if where_section:
            and_count = where_section.count(' AND ')
            or_count = where_section.count(' OR ')
            count += min(and_count + or_count, 10)

        return float(count)

    def _extract_where_section(self, query_str: str) -> str:
        
        where_match = re.search(r'\bWHERE\b(.*?)(?:\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|\bHAVING\b|$)',
                               query_str, re.IGNORECASE | re.DOTALL)
        return where_match.group(1).strip() if where_match else ""

    def _calculate_nesting_level(self, parsed_query) -> float:
        
        try:
            query_str = str(parsed_query)
            nesting_score = 0
            max_depth = 0
            current_depth = 0

            for char in query_str:
                if char == '(':
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == ')':
                    current_depth = max(0, current_depth - 1)

            subquery_count = self._count_subqueries(parsed_query)

            nesting_score = max_depth + subquery_count

            return float(min(nesting_score, 10))
        except:
            return 0.0

    def _calculate_complexity_score(self, features: Dict[str, float]) -> float:
        
        score = (
            features['num_tables'] * 2.0 +
            features['num_joins'] * 3.0 +
            features['num_subqueries'] * 5.0 +
            features['has_aggregation'] * 4.0 +
            features['has_window_functions'] * 6.0 +
            features['num_conditions'] * 1.0 +
            features['nested_level'] * 8.0 +
            features['query_length'] / 100.0
        )

        return round(score, 2)