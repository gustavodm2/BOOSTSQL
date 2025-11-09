# SQLBoost API

A FastAPI-based REST API for SQL query optimization using machine learning.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your database configuration in `config.py`

3. Train the ML agent (optional, but recommended):
```bash
python scripts/train_advanced_ml_agent.py
```

## Running the API

Start the API server:
```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET /
Basic health check endpoint.

**Response:**
```json
{
  "message": "SQLBoost API",
  "status": "running"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### POST /optimize
Optimize a SQL query using the ML agent.

**Request:**
```json
{
  "query": "SELECT u.name FROM users u WHERE u.id IN (SELECT user_id FROM orders WHERE total > 100)"
}
```

**Response:**
```json
{
  "original_query": "SELECT u.name FROM users u WHERE u.id IN (SELECT user_id FROM orders WHERE total > 100)",
  "original_execution_time": 1000.0,
  "best_optimization": {
    "optimization_type": "subquery_to_join",
    "improvement_ratio": 1.0,
    "optimized_query": "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id WHERE o.total > 100",
    "original_time": 1000.0,
    "optimized_time": 1000.0
  },
  "all_candidates_evaluated": 4,
  "candidates": [
    {
      "query": "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id WHERE o.total > 100",
      "type": "subquery_to_join",
      "confidence": 0.0
    }
  ],
  "successful_optimizations": 0,
  "recommendations": ["Consider reducing the number of JOINs or using intermediate tables"],
  "agent_insights": {
    "complexity_assessment": "low",
    "performance_prediction": "unknown",
    "similar_queries_analyzed": 0,
    "potential_bottlenecks": []
  }
}
```

### POST /rewrite
Directly rewrite a query using specific transformations.

**Request:**
```json
{
  "query": "SELECT u.name FROM users u WHERE u.id IN (SELECT user_id FROM orders WHERE total > 100)",
  "transformation": "subquery_to_join"
}
```

**Response:**
```json
{
  "original_query": "SELECT u.name FROM users u WHERE u.id IN (SELECT user_id FROM orders WHERE total > 100)",
  "transformations": {
    "subquery_to_join": [
      "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id WHERE o.total > 100"
    ]
  }
}
```

### GET /status
Get the current status of the ML agent.

**Response:**
```json
{
  "knowledge_base_patterns": 8,
  "optimizations_learned": 60,
  "strategies_available": 6,
  "exploration_rate": 0.3,
  "learning_rate": 0.1,
  "strategy_effectiveness": {
    "index_suggestion": {
      "success_rate": 0.0,
      "avg_improvement": 0.0,
      "attempts": 0
    }
  }
}
```

### GET /transforms
Get list of available transformation types.

**Response:**
```json
{
  "transformations": [
    "subquery_to_join",
    "join_reordering",
    "cte_materialization",
    "where_pushdown",
    "eliminate_redundant_joins",
    "simplify_expressions"
  ]
}
```

## Frontend Integration

The API includes CORS middleware, so it can be directly called from web browsers.

### JavaScript Example:

```javascript
// Optimize a query
const response = await fetch('http://localhost:8000/optimize', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)'
  })
});

const result = await response.json();
console.log('Optimized query:', result.best_optimization?.optimized_query);
```

### React Example:

```jsx
import { useState } from 'react';

function QueryOptimizer() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);

  const optimizeQuery = async () => {
    const response = await fetch('http://localhost:8000/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const data = await response.json();
    setResult(data);
  };

  return (
    <div>
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter SQL query..."
      />
      <button onClick={optimizeQuery}>Optimize</button>

      {result && (
        <div>
          <h3>Original Query:</h3>
          <pre>{result.original_query}</pre>

          {result.best_optimization && (
            <div>
              <h3>Optimized Query:</h3>
              <pre>{result.best_optimization.optimized_query}</pre>
              <p>Improvement: {result.best_optimization.improvement_ratio}x</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

## API Documentation

When the API is running, visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid input)
- `500`: Internal server error
- `503`: Service unavailable (agent not initialized)

Error responses include a `detail` field with error information.