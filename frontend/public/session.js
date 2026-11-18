(() => {
  const TOKEN_KEY = 'access_token';
  const TOKEN_EXPIRY_KEY = 'access_token_expires_at';
  let expiryTimer;

  function clearAuthentication() {
    if (expiryTimer) window.clearTimeout(expiryTimer);
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(TOKEN_EXPIRY_KEY);
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(TOKEN_EXPIRY_KEY);
  }

  function redirectToLogin() {
    if (window.location.pathname !== '/' && window.location.pathname !== '/index.html') {
      window.location.replace('/');
    }
  }

  function expireAuthenticationAt(expiresAt) {
    if (expiryTimer) window.clearTimeout(expiryTimer);
    const remaining = expiresAt - Date.now();
    if (remaining <= 0) {
      clearAuthentication();
      redirectToLogin();
      return;
    }

    expiryTimer = window.setTimeout(() => {
      clearAuthentication();
      redirectToLogin();
    }, remaining);
  }

  function getStoredAuthentication() {
    const storage = sessionStorage.getItem(TOKEN_KEY) ? sessionStorage : localStorage;
    const token = storage.getItem(TOKEN_KEY);
    if (!token) return null;

    const expiryValue = storage.getItem(TOKEN_EXPIRY_KEY);
    if (expiryValue) {
      const expiresAt = Number(expiryValue);
      if (!Number.isFinite(expiresAt) || expiresAt <= Date.now()) {
        clearAuthentication();
        return null;
      }
      expireAuthenticationAt(expiresAt);
    }

    return { token, storage };
  }

  function saveAccessToken(token, rememberMe, expiresIn) {
    if (
      typeof token !== 'string'
      || !token
      || !Number.isFinite(expiresIn)
      || expiresIn <= 0
    ) {
      throw new Error('The authentication response contains an invalid token lifetime.');
    }

    clearAuthentication();
    const storage = rememberMe ? localStorage : sessionStorage;
    const expiresAt = Date.now() + expiresIn * 1000;
    storage.setItem(TOKEN_KEY, token);
    storage.setItem(TOKEN_EXPIRY_KEY, String(expiresAt));
    expireAuthenticationAt(expiresAt);
  }

  async function getCurrentUser({ redirectOnUnauthorized = true } = {}) {
    const authentication = getStoredAuthentication();
    if (!authentication) {
      if (redirectOnUnauthorized) redirectToLogin();
      return null;
    }

    const response = await fetch('/api/v1/me', {
      headers: { Authorization: `Bearer ${authentication.token}` },
    });

    if (response.status === 401) {
      clearAuthentication();
      if (redirectOnUnauthorized) redirectToLogin();
      return null;
    }
    if (!response.ok) {
      throw new Error(`Could not load the current user (HTTP ${response.status}).`);
    }

    const user = await response.json();
    if (
      !user
      || typeof user.full_name !== 'string'
      || typeof user.username !== 'string'
      || typeof user.role !== 'string'
    ) {
      throw new Error('The current-user response is incomplete.');
    }
    return user;
  }

  async function authenticatedFetch(url, options = {}) {
    const authentication = getStoredAuthentication();
    if (!authentication) {
      redirectToLogin();
      throw new Error('Your sign-in has expired. Please sign in again.');
    }

    const response = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        Authorization: `Bearer ${authentication.token}`,
      },
    });

    if (response.status === 401) {
      clearAuthentication();
      redirectToLogin();
    }
    return response;
  }

  function getInitials(user) {
    const nameParts = user.full_name.trim().split(/\s+/).filter(Boolean);
    if (nameParts.length === 0) {
      return user.username.slice(0, 2).toUpperCase();
    }
    return nameParts.slice(0, 2).map((part) => part[0]).join('').toUpperCase();
  }

  function formatRole(role) {
    if (role === 'ADMINISTRATOR') return 'Administrator';
    if (role === 'ENFORCER') return 'Traffic enforcer';
    return role;
  }

  function logout() {
    clearAuthentication();
    window.location.replace('/');
  }

  window.smartLaneSession = {
    authenticatedFetch,
    getCurrentUser,
    getInitials,
    formatRole,
    logout,
    saveAccessToken,
  };
})();
