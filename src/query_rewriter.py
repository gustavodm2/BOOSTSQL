import sqlparse
from sqlparse import sql, tokens
from typing import List, Dict, Optional, Tuple, Any
import re
import logging

logger = logging.getLogger(__name__)

class SQLQueryRewriter:
    

    def __init__(self):
        self.supported_transformations = {
            'subquery_to_join': self._subquery_to_join,
            'join_reordering': self._reorder_joins,
            'cte_materialization': self._materialize_cte,
            'where_pushdown': self._pushdown_where,
            'eliminate_redundant_joins': self._eliminate_redundant_joins,
            'simplify_expressions': self._simplify_expressions
        }

    def rewrite_query(self, query: str, transformation: str, **kwargs) -> List[str]:
        
        if transformation not in self.supported_transformations:
            logger.warning(f"Unknown transformation: {transformation}")
            return [query]

        try:
            parsed = sqlparse.parse(query)[0]
            rewriter_func = self.supported_transformations[transformation]
            rewritten_queries = rewriter_func(parsed, **kwargs)

            valid_queries = []
            for rewritten in rewritten_queries:
                if self._validate_query(rewritten):
                    valid_queries.append(rewritten)
                else:
                    logger.warning(f"Invalid rewritten query discarded: {rewritten[:100]}...")

            return valid_queries if valid_queries else [query]

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return [query]

    def _validate_query(self, query: str) -> bool:
        
        try:
            parsed = sqlparse.parse(query)
            return len(parsed) > 0 and parsed[0] is not None
        except:
            return False

    def _subquery_to_join(self, parsed_query: sql.Statement, **kwargs) -> List[str]:

        rewritten_queries = []

        query_str = str(parsed_query)

        in_subquery_pattern = r'(\w+(?:\.\w+)?)\s+IN\s+\(\s*SELECT\s+(\w+(?:\.\w+)?)\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+?))?\s*\)'
        matches = re.findall(in_subquery_pattern, query_str, re.IGNORECASE | re.DOTALL)

        for main_col, sub_col, sub_table, where_clause in matches:
            try:
                    # Remove the entire WHERE condition containing the IN subquery
                    # Pattern: WHERE column IN (subquery) [AND/OR ...]
                    where_pattern = r'WHERE\s+(\w+(?:\.\w+)?)\s+IN\s+\(\s*SELECT\s+(\w+(?:\.\w+)?)\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+?))?\s*\)\s*(AND|OR)?'
                    join_query = re.sub(where_pattern, '', query_str, count=1, flags=re.IGNORECASE | re.DOTALL)

                    # If WHERE was completely removed and there's a trailing AND/OR, clean it up
                    join_query = re.sub(r'\s+(AND|OR)\s+(GROUP BY|ORDER BY|LIMIT|$)', r' \2', join_query, flags=re.IGNORECASE)

                    # Add JOIN clause to FROM
                    join_clause = f'JOIN {sub_table} ON {sub_table}.{sub_col} = {main_col}'
                    join_query = re.sub(
                        r'(FROM\s+\w+(?:\s+\w+)?)',
                        f'\\1 {join_clause}',
                        join_query,
                        count=1,
                        flags=re.IGNORECASE
                    )

                    # Add subquery WHERE conditions to main WHERE
                    if where_clause and where_clause.strip():
                        where_condition = f'{sub_table}.{where_clause.strip()}'
                        if 'WHERE' in join_query.upper():
                            join_query = re.sub(
                                r'(WHERE\s+.+?)(GROUP BY|ORDER BY|LIMIT|$)',
                                f'\\1 AND {where_condition} \\2',
                                join_query,
                                flags=re.IGNORECASE | re.DOTALL
                            )
                        else:
                            join_query = re.sub(
                                r'(FROM\s+.+?)(GROUP BY|ORDER BY|LIMIT|$)',
                                f'\\1 WHERE {where_condition} \\2',
                                join_query,
                                flags=re.IGNORECASE | re.DOTALL
                            )

                    rewritten_queries.append(join_query.strip())

            except Exception as e:
                logger.debug(f"Failed to convert subquery: {e}")
                continue

        return rewritten_queries

    def _reorder_joins(self, parsed_query: sql.Statement, **kwargs) -> List[str]:
        
        rewritten_queries = []

        joins_info = self._extract_join_info(parsed_query)
        if len(joins_info) <= 1:
            return []


        if len(joins_info) == 2:
            reversed_query = self._swap_joins(parsed_query, joins_info)
            if reversed_query:
                rewritten_queries.append(reversed_query)

        return rewritten_queries

    def _extract_join_info(self, parsed_query: sql.Statement) -> List[Dict]:
        
        joins = []

        tokens = list(parsed_query.flatten())
        i = 0
        while i < len(tokens):
            if (tokens[i].ttype is tokens.Keyword and
                'JOIN' in str(tokens[i]).upper()):

                join_info = {'type': str(tokens[i]).upper(), 'table': None, 'condition': None}

                j = i + 1
                while j < len(tokens) and not (tokens[j].ttype is tokens.Keyword and 'ON' in str(tokens[j]).upper()):
                    if tokens[j].ttype is None and str(tokens[j]).strip():
                        join_info['table'] = str(tokens[j]).strip()
                        break
                    j += 1

                if j < len(tokens) and str(tokens[j]).upper() == 'ON':
                    k = j + 1
                    condition_parts = []
                    while k < len(tokens) and not (tokens[k].ttype is tokens.Keyword and str(tokens[k]).upper() in ['JOIN', 'WHERE', 'GROUP', 'ORDER', 'LIMIT']):
                        if str(tokens[k]).strip():
                            condition_parts.append(str(tokens[k]))
                        k += 1
                    join_info['condition'] = ' '.join(condition_parts)

                joins.append(join_info)
            i += 1

        return joins

    def _swap_joins(self, parsed_query: sql.Statement, joins_info: List[Dict]) -> Optional[str]:
        
        if len(joins_info) != 2:
            return None

        query_str = str(parsed_query)

        try:
            join1_pattern = r'(JOIN\s+\w+\s+ON\s+[^J]+?)(?=JOIN|WHERE|GROUP|ORDER|LIMIT|$)'
            matches = re.findall(join1_pattern, query_str, re.IGNORECASE | re.DOTALL)

            if len(matches) >= 2:
                swapped = query_str.replace(matches[0], 'TEMP_JOIN_1', 1)
                swapped = swapped.replace(matches[1], matches[0], 1)
                swapped = swapped.replace('TEMP_JOIN_1', matches[1], 1)
                return swapped

        except Exception as e:
            logger.debug(f"Failed to swap joins: {e}")

        return None

    def _materialize_cte(self, parsed_query: sql.Statement, **kwargs) -> List[str]:
        
        rewritten_queries = []

        query_str = str(parsed_query)

        if 'WITH' in query_str.upper():
            materialized_query = re.sub(
                r'\bWITH\b',
                'WITH MATERIALIZED',
                query_str,
                count=1,
                flags=re.IGNORECASE
            )
            rewritten_queries.append(materialized_query)

        return rewritten_queries

    def _pushdown_where(self, parsed_query: sql.Statement, **kwargs) -> List[str]:
        
        rewritten_queries = []

        return rewritten_queries

    def _eliminate_redundant_joins(self, parsed_query: sql.Statement, **kwargs) -> List[str]:
        
        rewritten_queries = []


        return rewritten_queries

    def _simplify_expressions(self, parsed_query: sql.Statement, **kwargs) -> List[str]:
        
        rewritten_queries = []

        query_str = str(parsed_query)

        simplifications = [
            (r'\b(\w+)\s*=\s*\1\b', 'TRUE'),
            (r'\b(\w+)\s*!=\s*\1\b', 'FALSE'),
            (r'NOT\s*\(\s*TRUE\s*\)', 'FALSE'),
            (r'NOT\s*\(\s*FALSE\s*\)', 'TRUE'),
        ]

        simplified = query_str
        for pattern, replacement in simplifications:
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)

        if simplified != query_str:
            rewritten_queries.append(simplified)

        return rewritten_queries

    def get_supported_transformations(self) -> List[str]:
        
        return list(self.supported_transformations.keys())