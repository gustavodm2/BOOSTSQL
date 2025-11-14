import sqlparse
from sqlparse import sql, tokens
from typing import List, Dict, Optional, Tuple, Any
import re
import logging
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

logger = logging.getLogger(__name__)

class QueryRepresentation:
    """
    Represents a parsed SQL query with semantic components for intelligent analysis
    """

    def __init__(self, query: str):
        self.original_query = query
        self.parsed = None
        self.components = {
            'select': [],
            'from': [],
            'where': [],
            'joins': [],
            'group_by': [],
            'order_by': [],
            'having': [],
            'limit': None,
            'subqueries': [],
            'ctes': []
        }
        self._parse_query()

    def _parse_query(self):
        """Parse the query and extract semantic components"""
        try:
            self.parsed = sqlparse.parse(self.original_query)[0]
            self._extract_components()
        except Exception as e:
            logger.warning(f"Failed to parse query: {e}")

    def _extract_components(self):
        """Extract semantic components from the parsed query"""
        if not self.parsed:
            return

        current_section = None

        for token in self.parsed.flatten():
            token_str = str(token).strip().upper()

            # Identify section keywords
            if token.ttype is tokens.Keyword:
                if token_str in ['SELECT']:
                    current_section = 'select'
                elif token_str in ['FROM']:
                    current_section = 'from'
                elif token_str in ['WHERE']:
                    current_section = 'where'
                elif token_str in ['GROUP', 'GROUP BY']:
                    current_section = 'group_by'
                elif token_str in ['ORDER', 'ORDER BY']:
                    current_section = 'order_by'
                elif token_str in ['HAVING']:
                    current_section = 'having'
                elif token_str in ['LIMIT']:
                    current_section = 'limit'
                elif 'JOIN' in token_str:
                    current_section = 'joins'
                    self.components['joins'].append({'type': token_str, 'table': None, 'condition': None})
                elif token_str == 'WITH':
                    current_section = 'ctes'

            # Extract content based on current section
            elif current_section and token.ttype is None and str(token).strip():
                content = str(token).strip()
                if current_section == 'select':
                    self.components['select'].append(content)
                elif current_section == 'from':
                    self.components['from'].append(content)
                elif current_section == 'where':
                    self.components['where'].append(content)
                elif current_section == 'group_by':
                    self.components['group_by'].append(content)
                elif current_section == 'order_by':
                    self.components['order_by'].append(content)
                elif current_section == 'having':
                    self.components['having'].append(content)
                elif current_section == 'limit':
                    self.components['limit'] = content
                elif current_section == 'joins':
                    if self.components['joins'] and self.components['joins'][-1]['table'] is None:
                        self.components['joins'][-1]['table'] = content
                elif current_section == 'ctes':
                    self.components['ctes'].append(content)

            # Handle ON conditions for joins
            elif current_section == 'joins' and token.ttype is tokens.Keyword and token_str == 'ON':
                # Next tokens until next keyword are the condition
                pass  # Would need more complex parsing here

        # Extract subqueries
        self._extract_subqueries()

    def _extract_subqueries(self):
        """Extract subqueries from the query"""
        query_str = self.original_query.upper()

        # Find IN subqueries
        in_pattern = r'IN\s*\(\s*SELECT.*?\)'
        in_matches = re.findall(in_pattern, query_str, re.IGNORECASE | re.DOTALL)
        for match in in_matches:
            self.components['subqueries'].append({
                'type': 'IN',
                'content': match
            })

        # Find EXISTS subqueries
        exists_pattern = r'EXISTS\s*\(\s*SELECT.*?\)'
        exists_matches = re.findall(exists_pattern, query_str, re.IGNORECASE | re.DOTALL)
        for match in exists_matches:
            self.components['subqueries'].append({
                'type': 'EXISTS',
                'content': match
            })

    def get_similarity_score(self, other: 'QueryRepresentation') -> float:
        """Calculate similarity score between two query representations"""
        if not isinstance(other, QueryRepresentation):
            return 0.0

        score = 0.0
        total_weight = 0.0

        # Compare subqueries (high weight)
        subquery_similarity = len(set([s['type'] for s in self.components['subqueries']]) &
                                 set([s['type'] for s in other.components['subqueries']])) / \
                            max(len(self.components['subqueries']), len(other.components['subqueries']), 1)
        score += subquery_similarity * 0.4
        total_weight += 0.4

        # Compare joins (medium weight)
        join_similarity = len(self.components['joins']) == len(other.components['joins'])
        score += join_similarity * 0.3
        total_weight += 0.3

        # Compare CTEs (medium weight)
        cte_similarity = len(self.components['ctes']) == len(other.components['ctes'])
        score += cte_similarity * 0.2
        total_weight += 0.2

        # Compare select complexity (low weight)
        select_similarity = min(len(self.components['select']), len(other.components['select'])) / \
                           max(len(self.components['select']), len(other.components['select']), 1)
        score += select_similarity * 0.1
        total_weight += 0.1

        return score / total_weight if total_weight > 0 else 0.0


class QueryPattern:
    """
    Represents a learned transformation pattern with similarity matching
    """

    def __init__(self, transformation_type: str, before_representation: QueryRepresentation,
                 after_representation: QueryRepresentation, improvement_ratio: float):
        self.transformation_type = transformation_type
        self.before = before_representation
        self.after = after_representation
        self.improvement_ratio = improvement_ratio
        self.usage_count = 1
        self.success_rate = 1.0 if improvement_ratio > 1.0 else 0.0

    def matches_query(self, query_rep: QueryRepresentation, threshold: float = 0.7) -> bool:
        """Check if this pattern matches a given query representation"""
        similarity = self.before.get_similarity_score(query_rep)
        return similarity >= threshold

    def apply_to_query(self, query: str) -> Optional[str]:
        """Apply this transformation pattern to a query"""
        # For now, delegate to the fallback rewriter for the specific transformation type
        # In a full implementation, this would use learned transformation rules

        rewriter = SQLQueryRewriter()

        try:
            results = rewriter.rewrite_query(query, self.transformation_type)
            return results[0] if results else None
        except Exception:
            return None

    def update_success_rate(self, successful: bool):
        """Update the success rate of this pattern"""
        self.usage_count += 1
        if successful:
            self.success_rate = (self.success_rate * (self.usage_count - 1) + 1) / self.usage_count
        else:
            self.success_rate = (self.success_rate * (self.usage_count - 1)) / self.usage_count


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

        in_subquery_pattern = r'(\w+(?:\.\w+)?)\s+IN\s+\(\s*SELECT\s+(\w+(?:\.\w+)?)\s+FROM\s+(\w+)(?:\s+(\w+))?(?:\s+WHERE\s+(.+?))?\s*\)'
        matches = re.findall(in_subquery_pattern, query_str, re.IGNORECASE | re.DOTALL)

        for match in matches:
            main_col, sub_col, sub_table, sub_alias, where_clause = match
            try:
                    # Remove the IN subquery from the query
                    join_query = re.sub(in_subquery_pattern, '', query_str, count=1, flags=re.IGNORECASE | re.DOTALL)

                    # Clean up WHERE clause: remove double AND/OR, trailing/leading AND/OR
                    join_query = re.sub(r'\b(AND|OR)\s+\1\b', r'\1', join_query, flags=re.IGNORECASE)  # Remove double AND/OR
                    join_query = re.sub(r'\s+(AND|OR)\s+(GROUP BY|ORDER BY|LIMIT|$)', r' \2', join_query, flags=re.IGNORECASE)  # Remove trailing AND/OR
                    join_query = re.sub(r'(WHERE\s+(AND|OR)\s+)', r'WHERE ', join_query, flags=re.IGNORECASE)  # Remove leading AND/OR after WHERE
                    join_query = re.sub(r'WHERE\s*$', '', join_query, flags=re.IGNORECASE | re.MULTILINE)  # Remove empty WHERE

                    # Add JOIN clause to FROM
                    if sub_alias:
                        sub_column = sub_col.split('.')[1] if '.' in sub_col else sub_col
                        join_clause = f'JOIN {sub_table} {sub_alias} ON {sub_alias}.{sub_column} = {main_col}'
                    else:
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
                        if sub_alias:
                            where_condition = f'{sub_alias}.{where_clause.strip()}'
                        else:
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

    def _extract_join_info(self, parsed_query: sql.Statement) -> List[Dict]:

        joins = []

        token_list = list(parsed_query.flatten())
        i = 0
        while i < len(token_list):
            if (token_list[i].ttype is tokens.Keyword and
                'JOIN' in str(token_list[i]).upper()):

                join_info = {'type': str(tokens[i]).upper(), 'table': None, 'condition': None}

                j = i + 1
                while j < len(token_list) and not (token_list[j].ttype is tokens.Keyword and 'ON' in str(token_list[j]).upper()):
                    if token_list[j].ttype is None and str(token_list[j]).strip():
                        join_info['table'] = str(token_list[j]).strip()
                        break
                    j += 1

                if j < len(token_list) and str(token_list[j]).upper() == 'ON':
                    k = j + 1
                    condition_parts = []
                    while k < len(token_list) and not (token_list[k].ttype is tokens.Keyword and str(token_list[k]).upper() in ['JOIN', 'WHERE', 'GROUP', 'ORDER', 'LIMIT']):
                        if str(token_list[k]).strip():
                            condition_parts.append(str(token_list[k]))
                        k += 1
                    join_info['condition'] = ' '.join(condition_parts)

                joins.append(join_info)
            i += 1

        return joins

    def _validate_query(self, query: str) -> bool:
        """
        Validate that a rewritten query is syntactically correct
        """
        try:
            parsed = sqlparse.parse(query)
            return len(parsed) > 0 and parsed[0] is not None
        except:
            return False

    def train_model(self, training_data: List[Dict]):
        """
        Train the ML model on query optimization data
        """
        if not training_data:
            return

        # Prepare training data
        X = []
        y = []

        for item in training_data:
            original = item['original_query']
            optimized = item['optimized_query']
            improvement = item.get('improvement_ratio', 1.0)

            if self.feature_extractor:
                features = self.feature_extractor.extract_features(original)
                X.append(list(features.values()))

                # Label: 1 if optimization was successful (improvement > 1.1)
                y.append(1 if improvement > 1.1 else 0)

        if len(X) < 10:
            logger.warning("Not enough training data for ML model")
            return

        # Train a simple classifier
        self.transformation_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.transformation_model.fit(X, y)

        self._save_model()
        logger.info(f"Trained ML rewriter model on {len(X)} samples")


class MLQueryRewriter:
    """
    Intelligent query rewriter that uses machine learning models to generate
    optimized query rewrites based on learned patterns from execution data.
    """

    def __init__(self, model_path: str = 'models/query_rewriter_model.pkl'):
        self.model_path = model_path
        self.transformation_model = None
        self.patterns = defaultdict(list)
        self.feature_extractor = None  # Will be set by ML agent

        self._load_model()

    def _load_model(self):
        """Load the trained ML model for query rewriting"""
        if os.path.exists(self.model_path):
            try:
                loaded = joblib.load(self.model_path)
                self.transformation_model = loaded.get('model')
                self.patterns = loaded.get('patterns', defaultdict(list))
                logger.info(f"Loaded ML rewriter model with {len(self.patterns)} learned patterns")
            except Exception as e:
                logger.warning(f"Could not load ML rewriter model: {e}")
        else:
            logger.info("No ML rewriter model found, using fallback patterns")

    def _save_model(self):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        data = {
            'model': self.transformation_model,
            'patterns': self.patterns
        }
        joblib.dump(data, self.model_path)
        logger.info(f"Saved ML rewriter model to {self.model_path}")

    def set_feature_extractor(self, extractor):
        """Set the feature extractor for generating features"""
        self.feature_extractor = extractor

    def learn_from_optimization(self, original_query: str, optimized_query: str, improvement_ratio: float):
        """
        Learn from a successful optimization to improve future rewrites
        """
        if original_query == optimized_query:
            return

        # Extract pattern from the transformation
        pattern = self._extract_transformation_pattern(original_query, optimized_query)

        if pattern:
            # Create QueryPattern object for similarity matching
            before_rep = QueryRepresentation(original_query)
            after_rep = QueryRepresentation(optimized_query)
            pattern_obj = QueryPattern(pattern['type'], before_rep, after_rep, improvement_ratio)

            self.patterns[pattern['type']].append({
                'original': original_query,
                'optimized': optimized_query,
                'improvement': improvement_ratio,
                'pattern': pattern,
                'pattern_obj': pattern_obj
            })

            # Keep only top patterns to avoid memory issues
            if len(self.patterns[pattern['type']]) > 100:
                self.patterns[pattern['type']] = sorted(
                    self.patterns[pattern['type']],
                    key=lambda x: x['improvement'],
                    reverse=True
                )[:50]

    def rewrite_query(self, query: str) -> List[str]:
        """
        Generate intelligent rewrites using ML and learned patterns
        """
        rewrites = []

        # Always try to generate intelligent rewrites, even without ML model
        rewrites.extend(self._generate_intelligent_rewrites(query))

        # Use ML model if available for additional predictions
        if self.transformation_model and self.feature_extractor:
            try:
                features = self.feature_extractor.extract_features(query)
                # Predict which transformations to apply
                predictions = self._predict_transformations(features)
                for trans_type, confidence in predictions.items():
                    if confidence > 0.5:  # Lower threshold for more rewrites
                        candidates = self._apply_learned_transformation(query, trans_type)
                        rewrites.extend(candidates)
            except Exception as e:
                logger.debug(f"ML prediction failed: {e}")

        # Apply learned patterns
        rewrites.extend(self._apply_learned_patterns(query))

        # Generate additional intelligent variations
        rewrites.extend(self._generate_query_variations(query))

        # Remove duplicates and invalid queries
        valid_rewrites = []
        seen = set()
        for rewrite in rewrites:
            if rewrite not in seen and self._validate_query(rewrite) and rewrite != query:
                seen.add(rewrite)
                valid_rewrites.append(rewrite)

        return valid_rewrites[:5]  # Limit to top 5 candidates

    def _predict_transformations(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Use ML model to predict which transformations are likely to be beneficial
        """
        if not self.transformation_model:
            return {}

        # This is a simplified prediction - in practice, you'd have a trained classifier
        # For now, return some basic predictions based on features
        predictions = {}

        if features.get('num_subqueries', 0) > 0:
            predictions['subquery_to_join'] = 0.8

        if features.get('num_joins', 0) > 1:
            predictions['join_reordering'] = 0.7

        if 'WITH' in str(features.get('query_text', '')).upper():
            predictions['cte_materialization'] = 0.6

        return predictions

    def _apply_learned_transformation(self, query: str, trans_type: str) -> List[str]:
        """
        Apply transformations based on learned patterns
        """
        candidates = []

        # Find similar patterns in learned data
        for pattern_data in self.patterns.get(trans_type, []):
            pattern = pattern_data['pattern']
            if self._query_matches_pattern(query, pattern):
                # Apply the learned transformation
                rewritten = self._apply_pattern_transformation(query, pattern)
                if rewritten and rewritten != query:
                    candidates.append(rewritten)

        return candidates

    def _generate_intelligent_rewrites(self, query: str) -> List[str]:
        """
        Comprehensive intelligent query optimization pipeline
        """
        rewrites = []

        # Parse and analyze the query structure
        try:
            parsed = sqlparse.parse(query)[0]
            query_info = self._analyze_query_structure(query)
        except Exception as e:
            logger.debug(f"Failed to parse query: {e}")
            return rewrites

        # Create optimization pipeline - apply transformations in optimal order
        optimized_query = query

        # Phase 1: Structural transformations (subqueries, CTEs)
        structural_optimizations = self._apply_structural_optimizations(optimized_query, query_info)
        if structural_optimizations:
            optimized_query = structural_optimizations[0]  # Take the best structural optimization
            rewrites.extend(structural_optimizations)

        # Phase 2: Join optimizations (reordering, elimination)
        join_optimizations = self._apply_join_optimizations(optimized_query, query_info)
        if join_optimizations:
            # Apply join optimizations to the already optimized query
            for join_opt in join_optimizations:
                if join_opt != optimized_query:  # Only add if different
                    rewrites.append(join_opt)

        # Phase 3: Expression-level optimizations (simplifications, pushdown)
        expression_optimizations = self._apply_expression_optimizations(optimized_query, query_info)
        if expression_optimizations:
            for expr_opt in expression_optimizations:
                if expr_opt != optimized_query:  # Only add if different
                    rewrites.append(expr_opt)

        # Phase 4: Combined optimizations (apply multiple transformations together)
        combined_optimizations = self._apply_combined_optimizations(query, query_info)
        rewrites.extend(combined_optimizations)

        return rewrites

    def _apply_structural_optimizations(self, query: str, query_info: Dict) -> List[str]:
        """Apply structural transformations like subquery->join and CTE materialization"""
        optimizations = []

        # Subquery to JOIN conversions
        if query_info['has_subquery']:
            subquery_opts = self._intelligent_subquery_rewrites(query, query_info)
            optimizations.extend(subquery_opts)

        # CTE materialization
        if query_info['has_cte']:
            cte_opts = self._intelligent_cte_rewrites(query, query_info)
            optimizations.extend(cte_opts)

        return optimizations

    def _apply_join_optimizations(self, query: str, query_info: Dict) -> List[str]:
        """Apply JOIN-related optimizations"""
        optimizations = []

        # JOIN reordering
        if query_info['join_count'] > 1:
            join_opts = self._intelligent_join_rewrites(query, query_info)
            optimizations.extend(join_opts)

        # Redundant JOIN elimination
        if query_info['join_count'] > 0:
            redundant_opts = self._intelligent_redundant_join_elimination(query, query_info)
            optimizations.extend(redundant_opts)

        return optimizations

    def _apply_expression_optimizations(self, query: str, query_info: Dict) -> List[str]:
        """Apply expression-level optimizations"""
        optimizations = []

        # Expression simplifications
        expr_opts = self._intelligent_expression_simplifications(query)
        optimizations.extend(expr_opts)

        # WHERE pushdown
        if query_info['has_complex_where']:
            pushdown_opts = self._intelligent_where_pushdown(query, query_info)
            optimizations.extend(pushdown_opts)

        return optimizations

    def _apply_combined_optimizations(self, original_query: str, query_info: Dict) -> List[str]:
        """Apply multiple optimizations together for complex queries"""
        optimizations = []

        # For complex queries with multiple issues, try comprehensive optimization
        if (query_info['complexity_score'] > 3 and
            (query_info['has_subquery'] or query_info['has_cte'] or query_info['join_count'] > 1)):

            # Try to apply subquery->join AND expression simplification together
            combined_query = original_query

            # First apply subquery conversion
            if query_info['has_subquery']:
                subquery_opts = self._intelligent_subquery_rewrites(combined_query, query_info)
                if subquery_opts:
                    combined_query = subquery_opts[0]

            # Then apply expression simplifications
            expr_opts = self._intelligent_expression_simplifications(combined_query)
            if expr_opts:
                combined_query = expr_opts[0]

            # Then apply CTE materialization
            if query_info['has_cte']:
                cte_opts = self._intelligent_cte_rewrites(combined_query, query_info)
                if cte_opts:
                    combined_query = cte_opts[0]

            if combined_query != original_query:
                optimizations.append(combined_query)

        return optimizations

    def _analyze_query_structure(self, query: str) -> Dict:
        """
        Analyze the structure of the query to understand what rewrites are possible
        """
        info = {
            'has_subquery': False,
            'subquery_types': [],
            'join_count': 0,
            'has_cte': False,
            'has_aggregation': False,
            'has_window_functions': False,
            'complexity_score': 0
        }

        upper_query = query.upper()

        # Check for subqueries
        if 'SELECT' in upper_query and ('IN (' in upper_query or 'EXISTS (' in upper_query):
            info['has_subquery'] = True
            if 'IN (' in upper_query:
                info['subquery_types'].append('in_subquery')
            if 'EXISTS (' in upper_query:
                info['subquery_types'].append('exists_subquery')

        # Check for CTEs
        if 'WITH ' in upper_query:
            info['has_cte'] = True

        # Check for joins
        parsed = sqlparse.parse(query)[0]
        joins = self._extract_join_info(parsed)
        info['join_count'] = len(joins)
        info['joins'] = joins

        # Check for aggregation
        if any(word in upper_query for word in ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(']):
            info['has_aggregation'] = True

        # Check for window functions
        if any(word in upper_query for word in ['OVER (', 'PARTITION BY', 'ORDER BY']):
            info['has_window_functions'] = True

        # Check for redundant expressions
        info['has_redundant_expressions'] = self._detect_redundant_expressions(query)

        # Check for complex WHERE conditions that could be pushed down
        info['has_complex_where'] = self._detect_complex_where(query)

        # Check for join ordering opportunities
        info['join_order_issues'] = self._detect_join_order_issues(joins)

        return info

    def _detect_redundant_expressions(self, query: str) -> bool:
        """Detect expressions that can be simplified"""
        upper_query = query.upper()
        redundant_patterns = [
            r'\b(\w+(?:\.\w+)?)\s*=\s*\1\b',  # column = column (handles table.column)
            r'\b(\w+(?:\.\w+)?)\s*!=\s*\1\b',  # column != column
            r'NOT\s*\(\s*TRUE\s*\)',  # NOT TRUE
            r'NOT\s*\(\s*FALSE\s*\)',  # NOT FALSE
            r'\b1\s*=\s*1\b',  # 1=1
            r'\b0\s*=\s*1\b',  # 0=1
        ]
        return any(re.search(pattern, upper_query) for pattern in redundant_patterns)

    def _detect_complex_where(self, query: str) -> bool:
        """Detect WHERE clauses that might benefit from pushdown"""
        upper_query = query.upper()
        # Look for WHERE clauses with multiple conditions and JOINs
        has_where = 'WHERE' in upper_query
        has_joins = 'JOIN' in upper_query
        has_multiple_conditions = (' AND ' in upper_query or ' OR ' in upper_query)
        return has_where and (has_joins or has_multiple_conditions)

    def _detect_join_order_issues(self, joins: List[Dict]) -> bool:
        """Detect if JOIN order might need reordering"""
        if len(joins) < 2:
            return False
        # Simple heuristic: if we have more than 2 joins, reordering might help
        return len(joins) > 2

    def _intelligent_subquery_rewrites(self, query: str, query_info: Dict) -> List[str]:
        """
        Intelligently rewrite subqueries to joins using semantic analysis
        """
        rewrites = []

        if not query_info['has_subquery']:
            return rewrites

        try:
            # Parse the query to understand its structure
            parsed = sqlparse.parse(query)[0]

            # Find all IN subqueries
            in_subqueries = self._find_in_subqueries(parsed)

            for subquery_info in in_subqueries:
                # Analyze if this subquery can be converted to JOIN
                join_candidate = self._analyze_subquery_for_join(query, subquery_info, parsed)
                if join_candidate:
                    rewrites.append(join_candidate)

        except Exception as e:
            logger.debug(f"Intelligent subquery analysis failed: {e}")

        return rewrites

    def _find_in_subqueries(self, parsed_query) -> List[Dict]:
        """
        Find all IN subqueries in the parsed query
        """
        subqueries = []

        def traverse_tokens(token):
            if hasattr(token, 'tokens'):
                for sub_token in token.tokens:
                    traverse_tokens(sub_token)
            elif hasattr(token, 'ttype') and str(token).upper() == 'IN':
                # Found an IN keyword, look for the following subquery
                subquery_info = self._extract_in_subquery_info(token, parsed_query)
                if subquery_info:
                    subqueries.append(subquery_info)

        traverse_tokens(parsed_query)
        return subqueries

    def _extract_in_subquery_info(self, in_token, parsed_query) -> Optional[Dict]:
        """
        Extract information about an IN subquery
        """
        try:
            # Use flattened tokens from the parsed query
            all_tokens = list(parsed_query.flatten())
            in_index = all_tokens.index(in_token) if in_token in all_tokens else -1

            if in_index == -1:
                return None

            # Look for the column before IN (may include table alias)
            column_parts = []
            for i in range(in_index - 1, max(-1, in_index - 5), -1):  # Look back up to 5 tokens
                token = all_tokens[i]
                content = str(token).strip()
                if not content or content.isspace():
                    continue
                if content in ['WHERE', 'AND', 'OR', 'FROM', 'JOIN', 'ON']:  # Stop at keywords
                    break
                column_parts.insert(0, content)
                if len(column_parts) >= 3:  # table.alias.column max
                    break

            column = ''.join(column_parts).strip() if column_parts else None

            # Look for the subquery after IN
            subquery = None
            i = in_index + 1
            paren_count = 0
            subquery_tokens = []

            while i < len(all_tokens):
                token = all_tokens[i]
                content = str(token).strip()

                if content == '(':
                    paren_count += 1
                    subquery_tokens.append(str(token))  # Keep original token
                elif content == ')':
                    paren_count -= 1
                    subquery_tokens.append(str(token))
                    if paren_count == 0:
                        # Found the complete subquery
                        subquery = ''.join(subquery_tokens).strip()
                        if 'SELECT' in subquery.upper():
                            break
                        else:
                            subquery_tokens = []  # Reset if not a SELECT subquery
                elif paren_count > 0:
                    subquery_tokens.append(str(token))  # Keep original formatting
                elif content and not content.isspace() and paren_count == 0:
                    break  # Found something else before parentheses

                i += 1
                if i > in_index + 50:  # Prevent infinite loop
                    break

            if column and subquery:
                return {
                    'column': column,
                    'subquery': subquery,
                    'type': 'IN'
                }

        except Exception as e:
            logger.debug(f"Failed to extract IN subquery info: {e}")

        return None

    def _analyze_subquery_for_join(self, query: str, subquery_info: Dict, parsed_query) -> Optional[str]:
        """
        Analyze if a subquery can be converted to a JOIN and generate the rewrite
        """
        try:
            column = subquery_info.get('column')
            subquery = subquery_info.get('subquery')

            if not column or not subquery:
                return None

            # Parse the subquery to extract table and column
            # Strip outer parentheses if present
            subquery_content = subquery.strip()
            if subquery_content.startswith('(') and subquery_content.endswith(')'):
                subquery_content = subquery_content[1:-1].strip()
            subquery_upper = subquery_content.upper()
            # Handle SELECT [DISTINCT] column FROM table [alias]
            select_match = re.search(r'SELECT\s+(?:DISTINCT\s+)?(\w+(?:\.\w+)?)', subquery_upper)
            from_match = re.search(r'FROM\s+(\w+)\s*(\w+)?', subquery_upper)
            where_match = re.search(r'WHERE\s+(.+)', subquery_upper)

            if not select_match or not from_match:
                return None

            sub_column = select_match.group(1).strip()
            # If we captured DISTINCT, it means there's no column name - try again
            if sub_column.upper() == 'DISTINCT':
                # Look for the actual column after DISTINCT
                distinct_match = re.search(r'SELECT\s+DISTINCT\s+(\w+(?:\.\w+)?)', subquery_upper)
                if distinct_match:
                    sub_column = distinct_match.group(1).strip()
                else:
                    return None

            sub_table = from_match.group(1).strip()
            sub_alias = from_match.group(2).strip() if from_match.group(2) else None
            sub_where = where_match.group(1).strip() if where_match else None

            # Determine the main table column (handle table.column format)
            if '.' in column:
                main_table, main_column = column.split('.', 1)
            else:
                main_column = column
                # Try to infer main table from FROM clause - handle CTEs
                query_upper = query.upper()
                if 'WITH ' in query_upper:
                    # For CTE queries, find the main FROM after the CTE definition
                    # Pattern: WITH ... ) SELECT ... FROM cte_name
                    main_from_match = re.search(r'\)\s*SELECT.*?FROM\s+(\w+)', query_upper, re.DOTALL)
                    if main_from_match:
                        main_table = main_from_match.group(1).strip()
                    else:
                        main_table = 'cte_table'
                else:
                    from_clause = re.search(r'FROM\s+(\w+)', query_upper)
                    main_table = from_clause.group(1) if from_clause else 'main_table'

            # Remove the IN subquery from the query
            in_condition = f"{column} IN {subquery}"
            query_without_in = query.replace(in_condition, '')

            # Clean up the query
            query_without_in = re.sub(r'WHERE\s*$', '', query_without_in, flags=re.IGNORECASE | re.MULTILINE)
            query_without_in = re.sub(r'\s+(AND|OR)\s+(GROUP BY|ORDER BY|LIMIT|$)', r' \2', query_without_in, flags=re.IGNORECASE)
            query_without_in = re.sub(r'WHERE\s+(AND|OR)\s+', 'WHERE ', query_without_in, flags=re.IGNORECASE)

            # Add JOIN to FROM clause
            if 'WITH ' in query_without_in.upper():
                # Handle CTE
                cte_end_match = re.search(r'\)\s*SELECT', query_without_in.upper(), re.DOTALL)
                if not cte_end_match:
                    return None
                cte_end = cte_end_match.end() - 6
                main_select_from = query_without_in[cte_end:]

                main_from_match = re.search(r'FROM\s+(\w+)(?:\s+(\w+))?', main_select_from, re.IGNORECASE)
                if not main_from_match:
                    return None

                table_name = main_from_match.group(1).strip()
                alias = main_from_match.group(2).strip() if main_from_match.group(2) else table_name

                if sub_alias:
                    sub_col_clean = sub_column.split('.')[1] if '.' in sub_column else sub_column
                    join_clause = f'FROM {table_name} {alias} JOIN {sub_table} {sub_alias} ON {alias}.{main_column} = {sub_alias}.{sub_col_clean}'
                else:
                    join_clause = f'FROM {table_name} {alias} JOIN {sub_table} ON {alias}.{main_column} = {sub_table}.{sub_column}'
                query_with_join = query_without_in[:cte_end] + main_select_from.replace(main_from_match.group(0), join_clause)
            else:
                from_match = re.search(r'FROM\s+(\w+)\s+(\w+)', query_without_in, re.IGNORECASE)
                if not from_match:
                    return None

                table_name, alias = from_match.groups()
                if sub_alias:
                    sub_col_clean = sub_column.split('.')[1] if '.' in sub_column else sub_column
                    join_clause = f'FROM {table_name} {alias} JOIN {sub_table} {sub_alias} ON {alias}.{main_column} = {sub_alias}.{sub_col_clean}'
                else:
                    join_clause = f'FROM {table_name} {alias} JOIN {sub_table} ON {alias}.{main_column} = {sub_table}.{sub_column}'
                query_with_join = query_without_in.replace(f'FROM {table_name} {alias}', join_clause)

            # Add subquery WHERE conditions
            final_query = query_with_join
            if sub_where:
                if 'WHERE' in final_query.upper():
                    final_query += f' AND {sub_where}'
                else:
                    final_query += f' WHERE {sub_where}'

            return final_query

        except Exception as e:
            logger.debug(f"Failed to analyze subquery for join: {e}")
            return None

    def _intelligent_join_rewrites(self, query: str, query_info: Dict) -> List[str]:
        """
        Intelligently reorder joins for better performance
        """
        rewrites = []

        if query_info['join_count'] == 2:
            # Try swapping join order
            parsed = sqlparse.parse(query)[0]
            joins = self._extract_join_info(parsed)

            if len(joins) == 2:
                query_str = str(parsed)
                # Extract JOIN clauses
                join_clauses = []
                for match in re.finditer(r'\bJOIN\b', query_str, re.IGNORECASE):
                    start = match.start()
                    # Find the end of this JOIN clause
                    next_keywords = []
                    for m in re.finditer(r'\b(JOIN|WHERE|GROUP|ORDER|LIMIT)\b', query_str[start+1:], re.IGNORECASE):
                        next_keywords.append((m.start() + start + 1, m.group(1)))

                    if next_keywords:
                        end = next_keywords[0][0]
                        join_clause = query_str[start:end].strip()
                        join_clauses.append(join_clause)
                    else:
                        join_clause = query_str[start:].strip()
                        join_clauses.append(join_clause)

                if len(join_clauses) == 2:
                    # Swap the JOINs
                    swapped = query_str.replace(join_clauses[0], 'TEMP_JOIN_1', 1)
                    swapped = swapped.replace(join_clauses[1], join_clauses[0], 1)
                    swapped = swapped.replace('TEMP_JOIN_1', join_clauses[1], 1)
                    rewrites.append(swapped)

        return rewrites

    def _intelligent_cte_rewrites(self, query: str, query_info: Dict) -> List[str]:
        """
        Intelligently optimize CTEs
        """
        rewrites = []

        if 'WITH' in query.upper():
            # Add MATERIALIZED hint
            materialized_query = re.sub(
                r'\bWITH\b',
                'WITH MATERIALIZED',
                query,
                count=1,
                flags=re.IGNORECASE
            )
            rewrites.append(materialized_query)

        return rewrites

    def _intelligent_expression_simplifications(self, query: str) -> List[str]:
        """
        Apply intelligent expression simplifications
        """
        rewrites = []

        # Boolean simplifications
        simplifications = [
            (r'\b(\w+(?:\.\w+)?)\s*=\s*\1\b', 'TRUE'),  # column = column (handles table.column)
            (r'\b(\w+(?:\.\w+)?)\s*!=\s*\1\b', 'FALSE'),  # column != column
            (r'NOT\s*\(\s*TRUE\s*\)', 'FALSE'),
            (r'NOT\s*\(\s*FALSE\s*\)', 'TRUE'),
            (r'\b1\s*=\s*1\b', 'TRUE'),
            (r'\b0\s*=\s*1\b', 'FALSE'),
        ]

        simplified = query
        changed = False
        for pattern, replacement in simplifications:
            new_simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
            if new_simplified != simplified:
                simplified = new_simplified
                changed = True

        if changed:
            rewrites.append(simplified)

        return rewrites

    def _intelligent_where_pushdown(self, query: str, query_info: Dict) -> List[str]:
        """
        Intelligently push WHERE conditions down to reduce data processed
        """
        rewrites = []

        # Simple WHERE pushdown for basic cases
        # This is a simplified implementation - real pushdown would be more complex
        upper_query = query.upper()

        # Look for conditions that can be pushed to JOINs
        if 'JOIN' in upper_query and 'WHERE' in upper_query:
            # Extract WHERE conditions that reference joined tables
            where_match = re.search(r'WHERE\s+(.+?)(GROUP BY|ORDER BY|LIMIT|$)', upper_query, re.DOTALL)
            if where_match:
                where_clause = where_match.group(1).strip()

                # Simple heuristic: if WHERE contains table.column references, try to push to ON clause
                # This is a very basic implementation
                if '.' in where_clause and (' AND ' in where_clause or ' OR ' in where_clause):
                    # For now, just return the original query as pushdown might not always help
                    pass

        return rewrites

    def _intelligent_redundant_join_elimination(self, query: str, query_info: Dict) -> List[str]:
        """
        Intelligently eliminate redundant JOINs
        """
        rewrites = []

        joins = query_info.get('joins', [])
        if len(joins) < 2:
            return rewrites

        # Very basic redundant join detection
        # In practice, this would require schema analysis to determine if joins are truly redundant
        upper_query = query.upper()

        # Look for obvious redundant patterns (very simplified)
        # This is just a placeholder - real redundant join elimination needs schema knowledge
        redundant_patterns = [
            # Self-joins that might be redundant
            r'JOIN\s+\w+\s+ON\s+\w+\.\w+\s*=\s*\w+\.\w+',
        ]

        for pattern in redundant_patterns:
            if re.search(pattern, upper_query):
                # For now, don't actually remove joins without schema knowledge
                # Just mark that we detected a potential issue
                pass

        return rewrites

    def _apply_learned_patterns(self, query: str) -> List[str]:
        """
        Apply rewrites based on learned patterns from successful optimizations
        """
        rewrites = []
        query_rep = QueryRepresentation(query)

        # Find matching patterns using similarity
        for pattern_type, patterns in self.patterns.items():
            for pattern_data in patterns:
                pattern_obj = pattern_data.get('pattern_obj')
                if pattern_obj and pattern_obj.matches_query(query_rep, threshold=0.7):
                    # Apply the transformation using the learned pattern
                    applied_rewrite = pattern_obj.apply_to_query(query)
                    if applied_rewrite and applied_rewrite != query:
                        rewrites.append(applied_rewrite)

        return rewrites

    def get_transformation_candidates(self, query: str) -> List[Dict]:
        """
        Analyze query structure to identify potential optimizations
        """
        candidates = []
        query_rep = QueryRepresentation(query)
        query_info = self._analyze_query_structure(query)

        # Subquery to JOIN candidates
        if query_info['has_subquery'] and query_info['subquery_types']:
            for subquery_type in query_info['subquery_types']:
                if subquery_type == 'in_subquery':
                    candidates.append({
                        'type': 'subquery_to_join',
                        'confidence': 0.8,
                        'reason': 'IN subquery detected, can potentially be converted to JOIN'
                    })

        # JOIN reordering candidates
        if query_info['join_count'] > 1:
            candidates.append({
                'type': 'join_reordering',
                'confidence': 0.6,
                'reason': f'Multiple joins ({query_info["join_count"]}) detected, reordering may improve performance'
            })

        # CTE materialization candidates
        if query_info['has_cte']:
            candidates.append({
                'type': 'cte_materialization',
                'confidence': 0.7,
                'reason': 'CTE detected, materialization may improve performance'
            })

        # Expression simplification candidates
        if query_info['complexity_score'] > 2:
            candidates.append({
                'type': 'simplify_expressions',
                'confidence': 0.5,
                'reason': f'Complex query (score: {query_info["complexity_score"]}), expression simplification may help'
            })

        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)

        return candidates

    def _generate_query_variations(self, query: str) -> List[str]:
        """
        Generate additional query variations for testing
        """
        rewrites = []

        # Try different JOIN types
        if 'JOIN' in query.upper():
            inner_join = re.sub(r'\bLEFT JOIN\b', 'INNER JOIN', query, flags=re.IGNORECASE)
            if inner_join != query:
                rewrites.append(inner_join)

            left_join = re.sub(r'\bINNER JOIN\b', 'LEFT JOIN', query, flags=re.IGNORECASE)
            if left_join != query:
                rewrites.append(left_join)

        # Try different WHERE clause placements
        # This is a simplified example - in practice would be more sophisticated

        return rewrites

    def _apply_pattern_based_rewrites(self, query: str) -> List[str]:
        """
        Legacy method - now delegates to intelligent rewrites
        """
        return self._generate_intelligent_rewrites(query)

    def _extract_transformation_pattern(self, original: str, optimized: str) -> Optional[Dict]:
        """
        Extract the pattern of transformation between original and optimized queries
        """
        # This is a simplified pattern extraction
        # In practice, you'd use more sophisticated diffing

        orig_parsed = sqlparse.parse(original)[0]
        opt_parsed = sqlparse.parse(optimized)[0]

        orig_joins = self._extract_join_info(orig_parsed)
        opt_joins = self._extract_join_info(opt_parsed)

        if len(orig_joins) != len(opt_joins):
            if len(opt_joins) > len(orig_joins):
                return {'type': 'subquery_to_join', 'details': 'added_join'}
            elif len(orig_joins) > len(opt_joins):
                return {'type': 'join_elimination', 'details': 'removed_join'}

        # Check for CTE materialization
        if 'WITH' in original.upper() and 'WITH MATERIALIZED' in optimized.upper():
            return {'type': 'cte_materialization', 'details': 'added_materialized'}

        return None

    def _query_matches_pattern(self, query: str, pattern: Dict) -> bool:
        """
        Check if a query matches a learned pattern
        """
        # Simplified matching - in practice, use feature similarity
        query_features = self.feature_extractor.extract_features(query) if self.feature_extractor else {}

        if pattern['type'] == 'subquery_to_join':
            return query_features.get('num_subqueries', 0) > 0

        return False

    def _apply_pattern_transformation(self, query: str, pattern: Dict) -> Optional[str]:
        """
        Apply a learned transformation pattern to a query
        """
        # For now, delegate to fallback methods
        # In practice, you'd have learned transformation rules
        fallback_rewriter = SQLQueryRewriter()

        if pattern['type'] == 'subquery_to_join':
            results = fallback_rewriter.rewrite_query(query, 'subquery_to_join')
            return results[0] if results else None

        return None

    def _extract_join_info(self, parsed_query: sql.Statement) -> List[Dict]:
        """
        Extract join information from parsed query (copied from SQLQueryRewriter)
        """
        joins = []
        token_list = list(parsed_query.flatten())
        i = 0
        while i < len(token_list):
            if (token_list[i].ttype is tokens.Keyword and
                'JOIN' in str(token_list[i]).upper()):

                join_info = {'type': str(token_list[i]).upper(), 'table': None, 'condition': None}

                j = i + 1
                while j < len(token_list) and not (token_list[j].ttype is tokens.Keyword and 'ON' in str(token_list[j]).upper()):
                    if token_list[j].ttype is None and str(token_list[j]).strip():
                        join_info['table'] = str(token_list[j]).strip()
                        break
                    j += 1

                if j < len(token_list) and str(token_list[j]).upper() == 'ON':
                    k = j + 1
                    condition_parts = []
                    while k < len(token_list) and not (token_list[k].ttype is tokens.Keyword and str(token_list[k]).upper() in ['JOIN', 'WHERE', 'GROUP', 'ORDER', 'LIMIT']):
                        if str(token_list[k]).strip():
                            condition_parts.append(str(token_list[k]))
                        k += 1
                    join_info['condition'] = ' '.join(condition_parts)

                joins.append(join_info)
            i += 1

        return joins

    def _validate_query(self, query: str) -> bool:
        """
        Validate that a rewritten query is syntactically correct
        """
        try:
            parsed = sqlparse.parse(query)
            return len(parsed) > 0 and parsed[0] is not None
        except:
            return False

    def train_model(self, training_data: List[Dict]):
        """
        Train the ML model on query optimization data
        """
        if not training_data:
            return

        # Prepare training data
        X = []
        y = []

        for item in training_data:
            original = item['original_query']
            optimized = item['optimized_query']
            improvement = item.get('improvement_ratio', 1.0)

            if self.feature_extractor:
                features = self.feature_extractor.extract_features(original)
                X.append(list(features.values()))

                # Label: 1 if optimization was successful (improvement > 1.1)
                y.append(1 if improvement > 1.1 else 0)

        if len(X) < 10:
            logger.warning("Not enough training data for ML model")
            return

        # Train a simple classifier
        self.transformation_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.transformation_model.fit(X, y)

        self._save_model()
        logger.info(f"Trained ML rewriter model on {len(X)} samples")