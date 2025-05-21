import { getBaseUrl } from "@/utils";
import axios from "axios";

const apiClient = axios.create({
  baseURL: `${getBaseUrl()}/api`,
  headers: {
    Accept: "application/json",
    "Content-type": "application/json",
    "X-user-email": "test@test.com", // TODO: remove this
  },
  withCredentials: true,
});

export default apiClient;

const drClient = axios.create({
  baseURL: `${window.location.origin}/api/v2`,
  headers: {
    Accept: "application/json",
    "Content-type": "application/json",
  },
  withCredentials: true,
});

export { drClient, apiClient };
