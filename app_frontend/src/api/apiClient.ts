import axios from 'axios';
import axiosRetry from 'axios-retry';
import { getApiUrl } from '@/lib/utils';

const baseApiUrl = getApiUrl();

const apiClient = axios.create({
  baseURL: baseApiUrl,
  headers: {
    Accept: 'application/json',
    'Content-type': 'application/json',
  },
  withCredentials: true,
});

axiosRetry(apiClient, {
  retries: 5,
  retryDelay: axiosRetry.exponentialDelay,
  // Only retry idempotent methods — POSTs (create chat, send message, upload
  // dataset) must not retry on network errors, since a lost response after a
  // successful write would create duplicates.
  retryCondition: axiosRetry.isIdempotentRequestError,
});

export default apiClient;

export { apiClient };
