const API_BASE_URL = 'http://localhost:8002';

class SQLBoostApp {
    constructor() {
        this.currentSection = 'hero';
        this.theme = localStorage.getItem('theme') || 'light';
        this.history = JSON.parse(localStorage.getItem('queryHistory')) || [];
        this.settings = JSON.parse(localStorage.getItem('settings')) || {
            autoSave: true,
            historyRetention: 30,
            apiTimeout: 30,
            fontSize: 'medium'
        };

        this.init();
        this.bindEvents();
        this.loadSettings();
        this.updateStatus();
        this.renderHistory();
        this.renderAnalytics();
    }

    
    safeFormatNumber(value, decimals = 1) {
        if (value === null || value === undefined || typeof value !== 'number' || isNaN(value) || !isFinite(value)) {
            return '--';
        }
        
        const factor = Math.pow(10, decimals);
        const rounded = Math.round(value * factor) / factor;
        const parts = rounded.toString().split('.');
        if (decimals === 0) {
            return parts[0];
        }
        const decimalPart = (parts[1] || '').padEnd(decimals, '0');
        return `${parts[0]}.${decimalPart}`;
    }

    init() {
        
        document.documentElement.setAttribute('data-theme', this.theme);
        this.updateThemeToggle();

        
        this.showSection('hero');
    }

    bindEvents() {
        
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.currentTarget.dataset.section;
                this.showSection(section);
            });
        });

        
        document.getElementById('theme-toggle').addEventListener('click', () => {
            this.toggleTheme();
        });

        
        document.getElementById('get-started-btn').addEventListener('click', () => {
            this.showSection('optimize');
        });

        
        document.getElementById('query-input').addEventListener('input', (e) => {
            this.updateCharCount();
        });

        document.getElementById('query-input').addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this.optimizeQuery();
            }
        });

        document.getElementById('optimize-btn').addEventListener('click', () => {
            this.optimizeQuery();
        });

        document.getElementById('load-example-btn').addEventListener('click', () => {
            this.loadExampleQuery();
        });

        document.getElementById('clear-input-btn').addEventListener('click', () => {
            this.clearInput();
        });

        document.getElementById('copy-result-btn').addEventListener('click', () => {
            this.copyResult();
        });

        document.getElementById('download-result-btn').addEventListener('click', () => {
            this.downloadResult();
        });

        document.getElementById('save-results-btn').addEventListener('click', () => {
            this.saveResults();
        });

        
        document.getElementById('history-search').addEventListener('input', (e) => {
            this.filterHistory(e.target.value);
        });

        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.filterHistoryByType(e.currentTarget.dataset.filter);
            });
        });

        document.getElementById('clear-history-btn').addEventListener('click', () => {
            this.clearHistory();
        });

        
        document.getElementById('theme-select').addEventListener('change', (e) => {
            this.setTheme(e.target.value);
        });

        document.getElementById('font-size-select').addEventListener('change', (e) => {
            this.setFontSize(e.target.value);
        });

        document.getElementById('auto-save-toggle').addEventListener('change', (e) => {
            this.settings.autoSave = e.target.checked;
            this.saveSettings();
        });

        document.getElementById('history-retention-select').addEventListener('change', (e) => {
            this.settings.historyRetention = parseInt(e.target.value);
            this.saveSettings();
        });

        document.getElementById('timeout-input').addEventListener('change', (e) => {
            this.settings.apiTimeout = parseInt(e.target.value);
            this.saveSettings();
        });

        document.getElementById('clear-all-data-btn').addEventListener('click', () => {
            this.clearAllData();
        });

        
        setInterval(() => this.updateStatus(), 30000); 
    }

    showSection(sectionName) {
        
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.toggle('active', link.dataset.section === sectionName);
        });

        
        document.querySelectorAll('.section').forEach(section => {
            section.classList.toggle('active', section.id === `${sectionName}-section`);
        });

        this.currentSection = sectionName;

        
        if (sectionName !== 'hero') {
            window.history.replaceState(null, null, `#${sectionName}`);
        } else {
            window.history.replaceState(null, null, ' ');
        }
    }

    toggleTheme() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        this.setTheme(this.theme);
    }

    setTheme(theme) {
        this.theme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        this.updateThemeToggle();
    }

    updateThemeToggle() {
        const toggle = document.getElementById('theme-toggle');
        const icon = toggle.querySelector('i');
        icon.className = this.theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    setFontSize(size) {
        this.settings.fontSize = size;
        document.documentElement.setAttribute('data-font-size', size);
        localStorage.setItem('fontSize', size);
        this.saveSettings();
    }

    updateCharCount() {
        const input = document.getElementById('query-input');
        const count = input.value.length;
        document.getElementById('char-count').textContent = `${count} characters`;
    }

    async optimizeQuery() {
        const query = document.getElementById('query-input').value.trim();

        if (!query) {
            this.showNotification('Please enter a SQL query to optimize', 'error');
            return;
        }

        this.setLoading(true);
        this.showLoadingOverlay(true);

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.settings.apiTimeout * 1000);

            const response = await fetch(`${API_BASE_URL}/optimize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Optimization failed');
            }

            this.displayResults(data);
            this.addToHistory(query, data);
            this.showNotification('Query optimized successfully!', 'success');

        } catch (error) {
            if (error.name === 'AbortError') {
                this.showNotification('Request timed out. Please try again.', 'error');
            } else {
                console.error('Optimization error:', error);
                this.showNotification(`Error: ${error.message}`, 'error');
            }
            this.hideResults();
        } finally {
            this.setLoading(false);
            this.showLoadingOverlay(false);
        }
    }

    displayResults(data) {
        try {
            if (!data || typeof data !== 'object') {
                throw new Error('Invalid data received from API');
            }
            
            const safeGetElement = (id) => {
                try {
                    return document.getElementById(id) || { textContent: '', value: '' };
                } catch (e) {
                    return { textContent: '', value: '' };
                }
            };

            const output = safeGetElement('query-output');
            const optimizedQuery = (data && data.best_optimization && data.best_optimization.optimized_query) ||
                                    (data && data.optimized_query) ||
                                    (data && data.original_query) || '';
            output.value = optimizedQuery;



            const copyBtn = document.getElementById('copy-result-btn');
            const downloadBtn = document.getElementById('download-result-btn');
            if (copyBtn) copyBtn.disabled = false;
            if (downloadBtn) downloadBtn.disabled = false;

            
            this.updateLastUpdate();

        } catch (error) {
            console.error('Error in displayResults:', error);
            
            try {
                this.showNotification('Error displaying results. Please check the console for details.', 'error');
            } catch (e) {
                
            }
        }
    }

    hideResults() {
        document.getElementById('copy-result-btn').disabled = true;
        document.getElementById('download-result-btn').disabled = true;
    }

    displayRecommendations(recommendations) {
        const container = document.getElementById('recommendations-list');

        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-info-circle"></i><p>Query appears well-optimized</p></div>';
            return;
        }

        container.innerHTML = recommendations.map(rec => `
            <div class="recommendation-item">
                <div class="recommendation-icon">
                    <i class="fas fa-lightbulb"></i>
                </div>
                <div class="recommendation-content">
                    <p>${rec}</p>
                </div>
            </div>
        `).join('');
    }

    analyzeComplexity(query) {
        const lines = query.split('\n').length;
        const joins = (query.match(/\bJOIN\b/gi) || []).length;
        const subqueries = (query.match(/\(\s*SELECT/gi) || []).length;

        if (subqueries > 0) return 'High - Contains subqueries';
        if (joins > 2) return 'Medium - Multiple joins';
        if (lines > 5) return 'Medium - Multi-line query';
        return 'Low - Simple query';
    }

    identifyPattern(query) {
        if (query.includes('GROUP BY') && query.includes('HAVING')) {
            return 'Aggregation with filtering';
        }
        if (query.includes('JOIN') && query.includes('WHERE')) {
            return 'Join with conditions';
        }
        if (query.includes('ORDER BY') && query.includes('LIMIT')) {
            return 'Ordered result set';
        }
        if (query.includes('IN (SELECT')) {
            return 'Subquery in WHERE clause';
        }
        return 'Standard SELECT';
    }

    identifyIssues(query) {
        const issues = [];

        if (query.includes('SELECT *')) {
            issues.push('Uses SELECT *');
        }
        if (!query.includes('WHERE') && !query.includes('LIMIT')) {
            issues.push('No filtering or limiting');
        }
        if (query.includes('ORDER BY') && !query.includes('LIMIT')) {
            issues.push('ORDER BY without LIMIT');
        }

        return issues.length > 0 ? issues.join(', ') : 'No major issues detected';
    }

    getQueryType(query) {
        const upperQuery = query.toUpperCase().trim();
        if (upperQuery.startsWith('SELECT')) return 'SELECT';
        if (upperQuery.startsWith('INSERT')) return 'INSERT';
        if (upperQuery.startsWith('UPDATE')) return 'UPDATE';
        if (upperQuery.startsWith('DELETE')) return 'DELETE';
        return 'Unknown';
    }

    getTablesInvolved(query) {
        const fromMatch = query.match(/FROM\s+([`\w]+(?:\s*,\s*[`\w]+)*)/i);
        if (!fromMatch) return '--';

        const tables = fromMatch[1].split(',').map(t => t.trim().replace(/`/g, ''));
        return tables.length > 3 ? `${tables.slice(0, 3).join(', ')}...` : tables.join(', ');
    }



    loadExampleQuery() {
        const exampleQuery = `SELECT
    u.name,
    u.email,
    COUNT(o.id) as order_count,
    SUM(o.total_amount) as total_spent
FROM users u
WHERE u.id IN (
    SELECT DISTINCT user_id
    FROM orders
    WHERE order_date >= '2023-01-01'
    AND status = 'completed'
)
AND u.age > 18
AND u.age = u.age
GROUP BY u.id, u.name, u.email
HAVING COUNT(o.id) > 0
ORDER BY total_spent DESC
LIMIT 10`;

        document.getElementById('query-input').value = exampleQuery;
        this.updateCharCount();
        this.showNotification('Example query loaded!', 'info');
    }

    clearInput() {
        document.getElementById('query-input').value = '';
        this.updateCharCount();
        this.hideResults();
    }

    copyResult() {
        const output = document.getElementById('query-output');
        output.select();
        document.execCommand('copy');
        this.showNotification('Result copied to clipboard!', 'success');
    }

    downloadResult() {
        const output = document.getElementById('query-output').value;
        const blob = new Blob([output], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'optimized_query.sql';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        this.showNotification('Result downloaded!', 'success');
    }

    saveResults() {
        const data = {
            query: document.getElementById('query-input').value,
            optimized: document.getElementById('query-output').value,
            improvement: document.getElementById('improvement').textContent,
            timestamp: new Date().toISOString()
        };

        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'optimization_results.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        this.showNotification('Results saved!', 'success');
    }

    addToHistory(originalQuery, data) {
        const historyItem = {
            id: Date.now(),
            originalQuery,
            optimizedQuery: data.optimized_query || data.best_optimization?.optimized_query || originalQuery,
            improvement: data.best_optimization?.improvement_ratio || 1,
            strategy: data.best_optimization?.optimization_type || 'None',
            timestamp: new Date().toISOString(),
            performanceComparison: data.performance_comparison
        };

        this.history.unshift(historyItem);

        
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - this.settings.historyRetention);
        this.history = this.history.filter(item =>
            new Date(item.timestamp) > cutoffDate
        );

        this.saveHistory();
        this.renderHistory();
        this.renderAnalytics();
    }

    saveHistory() {
        localStorage.setItem('queryHistory', JSON.stringify(this.history));
    }

    renderHistory() {
        try {
            const container = document.getElementById('history-list');
            if (!container) return;

            if (!this.history || this.history.length === 0) {
                container.innerHTML = '<div class="empty-state"><i class="fas fa-history"></i><p>No optimization history yet</p><p>Start optimizing queries to build your history</p></div>';
                return;
            }

            container.innerHTML = this.history.map(item => {
                try {
                    const improvement = (item && item.improvement && typeof item.improvement === 'number') ?
                        this.safeFormatNumber(item.improvement) : '--';
                    const performance = (item && item.performanceComparison) || '--';
                    const strategy = (item && item.strategy) || '--';
                    const date = (item && item.timestamp) ? this.formatDate(item.timestamp) : '--';
                    const query = (item && item.originalQuery) ? this.escapeHtml(item.originalQuery.substring(0, 100)) : '';

                    return `
                        <div class="history-item" data-id="${item && item.id || 0}">
                            <div class="history-item-header">
                                <div class="history-query">${query}...</div>
                                <div class="history-meta">
                                    <span class="history-stat">
                                        <i class="fas fa-rocket"></i> ${improvement}x
                                    </span>
                                    <span class="history-stat">
                                        <i class="fas fa-tachometer-alt"></i> ${performance}
                                    </span>
                                    <span class="history-stat">
                                        <i class="fas fa-cogs"></i> ${strategy}
                                    </span>
                                    <span class="history-stat">
                                        <i class="fas fa-calendar"></i> ${date}
                                    </span>
                                </div>
                            </div>
                        </div>
                    `;
                } catch (e) {
                    return '<div class="history-item">Error loading history item</div>';
                }
            }).join('');

            
            try {
                container.querySelectorAll('.history-item').forEach(itemEl => {
                    itemEl.addEventListener('click', () => {
                        try {
                            const id = parseInt(itemEl.dataset.id);
                            const historyItem = this.history.find(h => h && h.id === id);
                            if (historyItem) {
                                this.loadFromHistory(historyItem);
                            }
                        } catch (e) {
                            console.error('Error loading history item:', e);
                        }
                    });
                });
            } catch (e) {
                console.error('Error adding history click handlers:', e);
            }
        } catch (error) {
            console.error('Error in renderHistory:', error);
        }
    }

    loadFromHistory(item) {
        document.getElementById('query-input').value = item.originalQuery;
        document.getElementById('query-output').value = item.optimizedQuery;
        this.updateCharCount();
        this.displayResults({
            original_query: item.originalQuery,
            best_optimization: {
                optimized_query: item.optimizedQuery,
                improvement_ratio: item.improvement,
                optimization_type: item.strategy
            },
            original_execution_time: item.executionTime,
            all_candidates_evaluated: 1
        });
        this.showSection('optimize');
        this.showNotification('Query loaded from history!', 'info');
    }

    filterHistory(searchTerm) {
        const items = document.querySelectorAll('.history-item');
        items.forEach(item => {
            const query = item.querySelector('.history-query').textContent.toLowerCase();
            const matches = query.includes(searchTerm.toLowerCase());
            item.style.display = matches ? 'block' : 'none';
        });
    }

    filterHistoryByType(type) {
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === type);
        });

        const items = document.querySelectorAll('.history-item');
        const now = new Date();

        items.forEach(item => {
            let show = true;

            switch (type) {
                case 'improved':
                    const improvement = parseFloat(item.querySelector('.history-stat:first-child').textContent.split('x')[0]);
                    show = improvement > 1;
                    break;
                case 'recent':
                    const timestamp = item.querySelector('.history-stat:last-child').textContent;
                    const itemDate = new Date(timestamp);
                    const hoursDiff = (now - itemDate) / (1000 * 60 * 60);
                    show = hoursDiff < 24;
                    break;
                default:
                    show = true;
            }

            item.style.display = show ? 'block' : 'none';
        });
    }

    clearHistory() {
        if (confirm('Are you sure you want to clear all history? This action cannot be undone.')) {
            this.history = [];
            this.saveHistory();
            this.renderHistory();
            this.renderAnalytics();
            this.showNotification('History cleared!', 'info');
        }
    }

    renderAnalytics() {
        try {
            if (!this.history || this.history.length === 0) {
                const avgGainEl = document.getElementById('avg-gain');
                if (avgGainEl) avgGainEl.textContent = '--x';
                return;
            }

            
            const improvements = this.history
                .map(h => h && h.improvement)
                .filter(i => i !== null && i !== undefined && typeof i === 'number' && !isNaN(i) && isFinite(i));
            const avgImprovement = improvements.length > 0 ? improvements.reduce((a, b) => a + b, 0) / improvements.length : 0;
            const avgGainEl = document.getElementById('avg-gain');
            if (avgGainEl) avgGainEl.textContent = `${this.safeFormatNumber(avgImprovement)}x`;

            
            const strategies = {};
            this.history.forEach(item => {
                try {
                    const strategy = (item && item.strategy) || 'Unknown';
                    strategies[strategy] = (strategies[strategy] || 0) + 1;
                } catch (e) {
                    
                }
            });

            const total = this.history.length;
            const breakdownContainer = document.getElementById('strategy-breakdown');
            if (breakdownContainer) {
                breakdownContainer.innerHTML = Object.entries(strategies)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 3)
                    .map(([strategy, count]) => {
                        try {
                            const percentage = total > 0 ? (count / total) * 100 : 0;
                            return `
                                <div class="strategy-item">
                                    <span class="strategy-name">${strategy || 'Unknown'}</span>
                                    <div class="strategy-bar">
                                        <div class="strategy-fill" style="width: ${this.safeFormatNumber(percentage)}%"></div>
                                    </div>
                                    <span class="strategy-percent">${this.safeFormatNumber(percentage)}%</span>
                                </div>
                            `;
                        } catch (e) {
                            return '<div class="strategy-item">Error</div>';
                        }
                    }).join('');
            }

            
            try {
                const selectCount = this.history.filter(h => h && h.originalQuery && h.originalQuery.toUpperCase().includes('SELECT')).length;
                const typeCountEl = document.querySelector('.type-count');
                if (typeCountEl) typeCountEl.textContent = selectCount;
            } catch (e) {
                console.error('Error calculating query types:', e);
            }
        } catch (error) {
            console.error('Error in renderAnalytics:', error);
        }
    }

    loadSettings() {
        
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            this.setTheme(savedTheme);
        }

        
        const savedFontSize = localStorage.getItem('fontSize');
        if (savedFontSize) {
            this.setFontSize(savedFontSize);
        }

        
        document.getElementById('theme-select').value = this.theme;
        document.getElementById('font-size-select').value = this.settings.fontSize;
        document.getElementById('auto-save-toggle').checked = this.settings.autoSave;
        document.getElementById('history-retention-select').value = this.settings.historyRetention;
        document.getElementById('timeout-input').value = this.settings.apiTimeout;
    }

    saveSettings() {
        localStorage.setItem('settings', JSON.stringify(this.settings));
    }

    clearAllData() {
        if (confirm('Are you sure you want to clear all data? This includes history, settings, and cached data. This action cannot be undone.')) {
            localStorage.clear();
            this.history = [];
            this.settings = {
                autoSave: true,
                historyRetention: 30,
                apiTimeout: 30,
                fontSize: 'medium'
            };
            this.theme = 'light';
            this.setTheme('light');
            this.loadSettings();
            this.renderHistory();
            this.renderAnalytics();
            this.showNotification('All data cleared!', 'warning');
        }
    }

    async updateStatus() {
        try {
            const response = await fetch(`${API_BASE_URL}/status`);
            const data = await response.json();

            document.getElementById('api-status').innerHTML = `API: <span class="status-text">Connected</span>`;
            document.getElementById('agent-status').innerHTML = `Agent: <span class="status-text">${data.agent_status || 'Unknown'}</span>`;
        } catch (error) {
            document.getElementById('api-status').innerHTML = `API: <span class="status-text">Disconnected</span>`;
            document.getElementById('agent-status').innerHTML = `Agent: <span class="status-text">Unknown</span>`;
        }

        this.updateLastUpdate();
    }

    updateLastUpdate() {
        const now = new Date();
        document.getElementById('last-update').textContent = `Last update: ${now.toLocaleTimeString()}`;
    }

    setLoading(loading) {
        const btn = document.getElementById('optimize-btn');
        const spinner = document.getElementById('optimize-spinner');
        const text = btn.querySelector('.btn-text');

        btn.classList.toggle('loading', loading);
        btn.disabled = loading;

        document.getElementById('load-example-btn').disabled = loading;
        document.getElementById('clear-input-btn').disabled = loading;
        document.getElementById('query-input').disabled = loading;
    }

    showLoadingOverlay(show) {
        document.getElementById('loading-overlay').classList.toggle('show', show);
    }

    showNotification(message, type = 'info') {
        const notifications = document.getElementById('notifications');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;

        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };

        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">
                    <i class="${icons[type]}"></i>
                </div>
                <div class="notification-message">${message}</div>
                <button class="notification-close">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        notifications.appendChild(notification);

        
        setTimeout(() => notification.classList.add('show'), 10);

        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 5000);

        
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        });
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        const diffHours = diffMs / (1000 * 60 * 60);

        if (diffHours < 1) {
            return 'Just now';
        } else if (diffHours < 24) {
            return `${Math.floor(diffHours)}h ago`;
        } else {
            return date.toLocaleDateString();
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}


document.addEventListener('DOMContentLoaded', () => {
    window.sqlBoostApp = new SQLBoostApp();

    
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.substring(1);
        if (hash && ['optimize', 'history', 'analytics', 'settings'].includes(hash)) {
            window.sqlBoostApp.showSection(hash);
        }
    });

    
    const initialHash = window.location.hash.substring(1);
    if (initialHash && ['optimize', 'history', 'analytics', 'settings'].includes(initialHash)) {
        window.sqlBoostApp.showSection(initialHash);
    }
});