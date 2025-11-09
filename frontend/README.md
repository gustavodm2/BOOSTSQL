# SQLBoost Frontend

A comprehensive, modern web interface for the SQLBoost AI-powered SQL query optimization platform.

## ✨ Features

### 🎨 Modern UI/UX
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Dark/Light Themes**: Automatic system theme detection with manual override
- **Smooth Animations**: Polished transitions and micro-interactions
- **Professional Typography**: Inter and JetBrains Mono fonts for optimal readability

### 🚀 Core Functionality
- **Real-time Optimization**: Instant query optimization with live feedback
- **Multi-section Interface**: Navigate between Optimize, History, Analytics, and Settings
- **Smart Query Analysis**: AI-powered complexity assessment and pattern recognition
- **Performance Dashboard**: Detailed metrics and improvement tracking

### 📊 Advanced Features
- **Query History**: Persistent storage with search and filtering capabilities
- **Analytics Dashboard**: Performance trends and strategy effectiveness insights
- **Settings Management**: Customizable themes, timeouts, and data retention
- **Local Storage**: All data persists locally in your browser

### 🔧 Developer Experience
- **Keyboard Shortcuts**: Ctrl+Enter for quick optimization
- **API Status Monitoring**: Real-time connection and agent status
- **Error Handling**: Comprehensive error messages and recovery
- **Export Options**: Copy, download, and save optimization results

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with required dependencies
- Modern web browser (Chrome 80+, Firefox 75+, Safari 13+, Edge 80+)

### Installation & Setup

1. **Install Backend Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API Server:**
   ```bash
   python run_api.py
   ```
   The API will be available at `http://localhost:8000`

3. **Open the Frontend:**
   - **Option A: Direct File Access**
     ```bash
     # Open index.html directly in your browser
     # Note: Some features may be limited due to CORS
     ```

   - **Option B: Local Web Server (Recommended)**
     ```bash
     # Using Python's built-in server
     cd frontend
     python -m http.server 3000
     ```
     Then visit `http://localhost:3000`

   - **Option C: Using Node.js (Alternative)**
     ```bash
     npx serve frontend -p 3000
     ```

## 📖 Usage Guide

### Getting Started
1. **Welcome Screen**: The app opens with an attractive hero section showcasing SQLBoost's capabilities
2. **Click "Get Started"** or navigate to the **Optimize** tab to begin

### Optimizing Queries
1. **Enter Query**: Paste or type your SQL query in the input area
2. **Load Example**: Click "Load Example" to see a sample complex query
3. **Optimize**: Click "Optimize Query" or press `Ctrl+Enter`
4. **View Results**: Examine the optimized query and performance metrics

### Understanding Results
- **Optimized Query**: The improved SQL with better performance
- **Performance Metrics**: Execution time, improvement ratio, candidates tested
- **AI Insights**: Query complexity, pattern recognition, potential issues
- **Recommendations**: Specific suggestions for further optimization

### Managing History
- **View Past Optimizations**: Switch to the History tab
- **Search & Filter**: Find specific queries by content or filter by type
- **Load Previous**: Click any history item to reload it for further optimization
- **Clear History**: Remove all stored history (irreversible)

### Analytics & Insights
- **Performance Trends**: Track your average improvement over time
- **Strategy Effectiveness**: See which optimization strategies work best
- **Query Patterns**: Understand your most common query types

### Customization
- **Theme Settings**: Choose between light, dark, or auto themes
- **Font Size**: Adjust text size for better readability
- **API Configuration**: Modify connection settings and timeouts
- **Data Management**: Control history retention and clear all data

## 🎯 Example Workflow

```sql
-- Original Query (paste this or load example)
SELECT u.name, COUNT(o.id) as orders
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_date >= '2023-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5
ORDER BY orders DESC

-- SQLBoost Optimization Result:
-- - Converts to more efficient JOIN patterns
-- - Simplifies expressions
-- - Shows 2.3x performance improvement
-- - Provides specific recommendations
```

## 🔌 API Integration

The frontend communicates with the SQLBoost API using REST endpoints:

### Core Endpoints
- `POST /optimize` - Query optimization with ML agent
- `GET /status` - Agent and system status
- `GET /transforms` - Available transformation strategies
- `GET /docs` - Interactive API documentation

### Connection Details
- **Default URL**: `http://localhost:8000`
- **Timeout**: Configurable (default: 30 seconds)
- **CORS**: Enabled for cross-origin requests
- **Status Updates**: Automatic polling every 30 seconds

## 🎨 Customization

### Changing API Endpoint
```javascript
// In script.js, modify:
const API_BASE_URL = 'https://your-custom-api.com';
```

### Custom Styling
Edit `styles.css` to modify:
- Color schemes and gradients
- Font families and sizes
- Layout spacing and breakpoints
- Animation durations and effects

### Adding New Features
Extend the `SQLBoostApp` class in `script.js`:
```javascript
// Example: Add export functionality
exportToFile() {
    // Implementation here
}
```

## 🛠️ Development

### File Structure
```
frontend/
├── index.html          # Main HTML structure with all sections
├── styles.css          # Comprehensive CSS with themes and animations
├── script.js           # Full application logic and API integration
└── README.md           # This documentation
```

### Key Components
- **Navigation System**: Tab-based navigation with URL hash support
- **State Management**: Local storage for settings and history
- **API Client**: Centralized HTTP request handling
- **UI Components**: Modular sections with consistent styling
- **Event System**: Comprehensive event binding and handling

### Browser Compatibility
- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## 🔧 Troubleshooting

### Common Issues

**API Connection Failed**
```
Error: Failed to fetch
```
- Verify API server is running: `curl http://localhost:8000/status`
- Check CORS settings in API
- Ensure firewall allows port 8000

**Optimization Not Working**
```
Error: Agent not ready
```
- Train the ML agent: `python scripts/train_advanced_ml_agent.py`
- Check agent status in Settings tab
- Review API logs for training issues

**UI Not Loading Properly**
- Use a local web server instead of file:// protocol
- Clear browser cache and local storage
- Check browser console for JavaScript errors

**Dark Theme Not Working**
- Check system theme settings
- Manually select theme in Settings
- Clear local storage if corrupted

### Debug Mode
Enable verbose logging in browser console:
```javascript
localStorage.setItem('debug', 'true');
```

### Reset Application
To completely reset the frontend:
1. Clear browser local storage
2. Hard refresh (Ctrl+F5)
3. Reopen the application

## 📊 Performance Tips

- **Large Queries**: Break complex queries into smaller parts
- **Batch Processing**: Use the API directly for bulk optimization
- **Browser Resources**: Close other tabs for better performance
- **Cache Management**: Regularly clear history if storage is limited

## 🤝 Contributing

### Development Setup
1. Fork and clone the repository
2. Install backend dependencies
3. Start API server for testing
4. Make frontend changes
5. Test across different browsers
6. Submit pull request with description

### Code Style
- Use modern ES6+ JavaScript
- Follow consistent naming conventions
- Add JSDoc comments for functions
- Test all new features thoroughly

### Feature Requests
- Check existing issues before creating new ones
- Provide detailed use cases and mockups
- Consider backward compatibility

## 📄 License

Licensed under the same terms as the main SQLBoost project. See main repository for details.

## 🆘 Support

- **Documentation**: Check this README and API docs
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **API Docs**: Visit `http://localhost:8000/docs` when running

---

**SQLBoost**: Transforming SQL queries with the power of AI 🤖✨