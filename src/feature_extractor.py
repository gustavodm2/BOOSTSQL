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
            'has_limit', 'has_distinct', 'join_complexity', 'nested_level'
        ]
    
    def extract_features(self, query: str) -> Dict[str, float]:
        """Extrai características de uma query SQL"""
        try:
            parsed = sqlparse.parse(query)[0]
        except:
            return self._default_features()
        
        features = {
            'num_joins': self._count_joins(parsed),
            'num_subqueries': self._count_subqueries(parsed),
            'has_group_by': 1.0 if self._has_clause(parsed, 'GROUP BY') else 0.0,
            'has_order_by': 1.0 if self._has_clause(parsed, 'ORDER BY') else 0.0,
            'has_where': 1.0 if self._has_clause(parsed, 'WHERE') else 0.0,
            'num_conditions': self._count_conditions(parsed),
            'num_tables': self._count_tables(parsed),
            'query_length': len(query),
            'has_aggregation': 1.0 if self._has_aggregation(parsed) else 0.0,
            'num_select_columns': self._count_select_columns(parsed),
            'has_having': 1.0 if self._has_clause(parsed, 'HAVING') else 0.0,
            'has_limit': 1.0 if self._has_clause(parsed, 'LIMIT') else 0.0,
            'has_distinct': 1.0 if 'DISTINCT' in query.upper() else 0.0,
            'join_complexity': self._calculate_join_complexity(parsed),
            'nested_level': self._calculate_nested_level(parsed)
        }
        
        return features
    
    def _count_joins(self, parsed):
        return len([token for token in parsed.tokens if token.ttype is sqlparse.tokens.Keyword and 'JOIN' in token.value.upper()])
    
    def _count_subqueries(self, parsed):
        return len(re.findall(r'\(SELECT', str(parsed).upper()))
    
    def _has_clause(self, parsed, clause):
        return clause.upper() in str(parsed).upper()
    
    def _count_conditions(self, parsed):
        where_clause = self._get_clause_tokens(parsed, 'WHERE')
        return len([token for token in where_clause if token.ttype is sqlparse.tokens.Keyword])
    
    def _count_tables(self, parsed):
        # Conta tabelas na cláusula FROM
        tables = set()
        from_seen = False
        
        for token in parsed.tokens:
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                from_seen = True
            elif from_seen and token.ttype is None:
                # Encontrou a lista de tabelas
                table_text = str(token)
                # Extrai nomes de tabelas (simplificado)
                tables.update(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', table_text))
                break
        
        return len(tables)
    
    def _has_aggregation(self, parsed):
        aggs = ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']
        return any(agg in str(parsed).upper() for agg in aggs)
    
    def _count_select_columns(self, parsed):
        select_seen = False
        columns = 0
        
        for token in parsed.tokens:
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'SELECT':
                select_seen = True
            elif select_seen and token.ttype is None:
                # Conta vírgulas para estimar número de colunas
                columns = str(token).count(',') + 1
                break
        
        return columns
    
    def _calculate_join_complexity(self, parsed):
        joins = self._count_joins(parsed)
        conditions = self._count_conditions(parsed)
        return joins * conditions * 0.1
    
    def _calculate_nested_level(self, parsed):
        return str(parsed).count('(')
    
    def _get_clause_tokens(self, parsed, clause):
        clause_tokens = []
        in_clause = False
        
        for token in parsed.tokens:
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == clause:
                in_clause = True
            elif in_clause and token.ttype is sqlparse.tokens.Keyword:
                break
            elif in_clause:
                clause_tokens.append(token)
        
        return clause_tokens
    
    def _default_features(self):
        return {feature: 0.0 for feature in self.feature_names}