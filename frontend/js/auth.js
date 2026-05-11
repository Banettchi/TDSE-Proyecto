/**
 * DermAI — Auth module
 * JWT token management and API communication
 */
const API_BASE = '/api';

const Auth = {
    getToken() { return localStorage.getItem('dermai_token'); },
    setToken(token) { localStorage.setItem('dermai_token', token); },
    getUser() { try { return JSON.parse(localStorage.getItem('dermai_user')); } catch { return null; } },
    setUser(user) { localStorage.setItem('dermai_user', JSON.stringify(user)); },
    isLoggedIn() { return !!this.getToken(); },
    logout() { localStorage.clear(); window.location.href = '/'; },

    headers() {
        const h = { 'Accept': 'application/json' };
        const t = this.getToken();
        if (t) h['Authorization'] = `Bearer ${t}`;
        return h;
    },

    async api(endpoint, options = {}) {
        const url = `${API_BASE}${endpoint}`;
        const config = { headers: this.headers(), ...options };
        if (config.body && !(config.body instanceof FormData)) {
            config.headers['Content-Type'] = 'application/json';
            config.body = JSON.stringify(config.body);
        }
        const res = await fetch(url, config);
        if (res.status === 401) { this.logout(); return; }
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Error del servidor' }));
            throw new Error(err.detail || `Error ${res.status}`);
        }
        return res.json();
    },

    async login(email, password) {
        const data = await this.api('/auth/login', {
            method: 'POST',
            body: { email, password }
        });
        this.setToken(data.access_token);
        this.setUser({ name: data.user_name, role: data.role, tenant: data.tenant_name, tenant_id: data.tenant_id });
        return data;
    },

    checkAuth() {
        if (!this.isLoggedIn()) { window.location.href = '/'; return false; }
        return true;
    },

    initSidebar() {
        const user = this.getUser();
        if (!user) return;
        const el = document.getElementById('userInfo');
        if (el) el.innerHTML = `<div class="user-card"><div class="user-avatar">${user.name[0]}</div><div><div class="user-name">${user.name}</div><div class="user-role">${user.role} — ${user.tenant}</div></div></div>`;
    }
};

// Toast notifications
function showToast(message, type = 'info') {
    let container = document.querySelector('.toast-container');
    if (!container) { container = document.createElement('div'); container.className = 'toast-container'; document.body.appendChild(container); }
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<span>${type === 'success' ? '✓' : type === 'error' ? '✗' : 'ℹ'}</span> ${message}`;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 4000);
}

function showLoading(text = 'Procesando...') {
    let overlay = document.querySelector('.loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `<div class="spinner"></div><div class="loading-text">${text}</div><div class="loading-progress"><div class="loading-progress-bar"></div></div>`;
        document.body.appendChild(overlay);
    }
    overlay.querySelector('.loading-text').textContent = text;
    overlay.classList.add('active');
}

function hideLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) overlay.classList.remove('active');
}

function formatDate(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('es-CO', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function riskBadge(level) {
    const map = { bajo: 'low', medio: 'medium', alto: 'high', muy_alto: 'very-high' };
    const labels = { bajo: 'Bajo', medio: 'Medio', alto: 'Alto', muy_alto: 'Muy Alto' };
    return `<span class="badge badge-${map[level] || 'medium'}">${labels[level] || level}</span>`;
}
