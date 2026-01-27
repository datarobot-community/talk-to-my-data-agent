import { apiClient } from '../apiClient';

interface DataRobotInfoResponse {
  datarobot_account_info: {
    uid: string;
    username: string;
    email: string;
    language: string;
    [key: string]: string | number | boolean | null | undefined;
  } | null;
  datarobot_api_token: string | null;
  datarobot_api_scoped_token: string | null;
}

export const getDataRobotInfo = async (): Promise<DataRobotInfoResponse> => {
  const response = await apiClient.get<DataRobotInfoResponse>('/v1/user/datarobot-account');

  if (!response.data.datarobot_api_scoped_token && !response.data.datarobot_api_token) {
    try {
      const updatedResponse = await apiClient.get<DataRobotInfoResponse>(
        '/v1/user/datarobot-account'
      );
      return updatedResponse.data;
    } catch (error) {
      console.error('Error fetching DataRobot info:', error);
    }
  }

  return response.data;
};

export const updateApiToken = async (apiToken: string): Promise<void> => {
  try {
    await apiClient.post('/v1/user/datarobot-account', {
      api_token: apiToken,
    });
  } catch (error) {
    console.error('Error updating API token:', error);
    throw error;
  }
};
