import axios from 'axios';
import { isDev, isServedStatic } from '@/lib/utils';
import { VITE_DEFAULT_PORT } from '@/constants/dev';

export const getApiUrl = () => {
    // Adjust API URL based on the environment
    let apiBaseURL: string = window.location.origin;
    if (window.ENV?.APP_BASE_URL) {
        apiBaseURL = `${apiBaseURL}/${window.ENV.APP_BASE_URL}`;
    }
    if (window.ENV?.APP_BASE_URL?.includes('notebook-sessions') && isDev()) {
        apiBaseURL += `/ports/` + VITE_DEFAULT_PORT;
    }
    if (
        window.ENV?.APP_BASE_URL?.includes('notebook-sessions') &&
        window.ENV?.API_PORT &&
        isServedStatic()
    ) {
        apiBaseURL += `/ports/` + window.ENV.API_PORT;
    }
    apiBaseURL += '/api';
    return apiBaseURL;
};

const apiClient = axios.create({
    baseURL: `${getApiUrl()}`,
    headers: {
        Accept: 'application/json',
        'Content-type': 'application/json',
    },
    withCredentials: true,
});

export default apiClient;

const drClient = axios.create({
    baseURL: window.ENV?.DATAROBOT_ENDPOINT || `${window.location.origin}/api/v2`,
    headers: {
        Accept: 'application/json',
        'Content-type': 'application/json',
    },
    withCredentials: true,
});

export { drClient, apiClient };
