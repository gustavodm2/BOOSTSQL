import sqlparse
import re
import numpy as np
from typing import Dict

class SQLFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'num_joins', 'num_subqueries', 'has_group_by', 'has_order_by',
            'has_where', 'num_conditions', 'num_tables', 'query_length',
            'has_aggregation', 'num_select_columns', 'has_having',
            'has_limit', 'has_distinct', 'join_complexity', 'nested_level',
            'has_union', 'has_cte', 'has_window_functions', 'string_operations'
        ]
    
    def extract_features(self, query: str) -> Dict[str, float]:
        """Extrai 19 características das queries SQL"""
        try:
            parsed = sqlparse.parse(query)[0]
        except:
            return self._default_features()
        
        query_upper = query.upper()
        
        features = {
            'num_joins': self._count_joins(parsed),
            'num_subqueries': self._count_subqueries(query),
            'has_group_by': 1.0 if 'GROUP BY' in query_upper else 0.0,
            'has_order_by': 1.0 if 'ORDER BY' in query_upper else 0.0,
            'has_where': 1.0 if 'WHERE' in query_upper else 0.0,
            'num_conditions': self._count_conditions(query),
            'num_tables': self._count_tables(query),
            'query_length': len(query),
            'has_aggregation': 1.0 if self._has_aggregation(query) else 0.0,
            'num_select_columns': self._count_select_columns(parsed),
            'has_having': 1.0 if 'HAVING' in query_upper else 0.0,
            'has_limit': 1.0 if 'LIMIT' in query_upper else 0.0,
            'has_distinct': 1.0 if 'DISTINCT' in query_upper else 0.0,
            'join_complexity': self._calculate_join_complexity(parsed),
            'nested_level': query.count('('),
            'has_union': 1.0 if 'UNION' in query_upper else 0.0,
            'has_cte': 1.0 if 'WITH' in query_upper else 0.0,
            'has_window_functions': 1.0 if any(f in query_upper for f in ['OVER(', 'RANK()', 'ROW_NUMBER()']) else 0.0,
            'string_operations': self._count_string_operations(query)
        }
        
        return features
    
    def _count_joins(self, parsed):
        joins = ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN']
        count = 0
        for token in parsed.tokens:
            if hasattr(token, 'ttype') and token.ttype is sqlparse.tokens.Keyword:
                if token.value.upper() in joins:
                    count += 1
        return count
    
    def _count_subqueries(self, query):
        return len(re.findall(r'\(\s*SELECT', query.upper()))
    
    def _count_conditions(self, query):
        where_pos = query.upper().find('WHERE')
        if where_pos == -1:
            return 0
        where_clause = query[where_pos:]
        return where_clause.count('AND') + where_clause.count('OR') + 1
    
    def _count_tables(self, query):
        tables = set()
        from_match = re.search(r'FROM\s+([^(]+?)(?:\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|$)', query.upper())
        if from_match:
            tables.update(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', from_match.group(1)))
        return len(tables)
    
    def _has_aggregation(self, query):
        aggs = ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']
        return any(agg in query.upper() for agg in aggs)
    
    def _count_select_columns(self, parsed):
        select_seen = False
        for token in parsed.tokens:
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'SELECT':
                select_seen = True
            elif select_seen and token.ttype is None:
                return str(token).count(',') + 1
        return 1
    
    def _calculate_join_complexity(self, parsed):
        joins = self._count_joins(parsed)
        conditions = self._count_conditions(str(parsed))
        return joins * conditions * 0.1
    
    def _count_string_operations(self, query):
        operations = ['LIKE', 'CONCAT', 'SUBSTRING', 'UPPER', 'LOWER']
        return sum(query.upper().count(op) for op in operations)
    
    def _default_features(self):
        return {feature: 0.0 for feature in self.feature_names}