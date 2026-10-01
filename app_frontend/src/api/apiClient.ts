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
  retryCondition: error =>
    axiosRetry.isIdempotentRequestError(error) ||
    // The platform allows one AI Catalog search per user at a time and answers
    // 409 to a concurrent one. It isn't in the default idempotent set, but on a
    // read it just means "try again shortly" (AECO-44).
    (error.config?.method?.toLowerCase() === 'get' && error.response?.status === 409),
});

export default apiClient;

export { apiClient };
