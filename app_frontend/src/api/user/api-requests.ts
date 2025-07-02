import { apiClient, drClient } from '../apiClient';

export interface DataRobotInfoResponse {
    datarobot_account_info: {
        uid: string;
        username: string;
        email: string;
        [key: string]: string | number | boolean | null | undefined;
    } | null;
    datarobot_api_token: string | null;
    datarobot_api_skoped_token: string | null;
}

export interface DataRobotAccount {
    uid: string;
    username: string;
    email: string;
    [key: string]: string | number | boolean | null | undefined;
}

export interface DataRobotStoreInfoRequest {
    account_info?: {
        uid: string;
        username: string;
        email: string;
        [key: string]: string | number | boolean | null | undefined;
    } | null;
    api_token?: string | null;
}

export const getDataRobotInfo = async (): Promise<DataRobotInfoResponse> => {
    const response = await apiClient.get<DataRobotInfoResponse>('/v1/user/datarobot-account');

    if (!response.data.datarobot_api_skoped_token && !response.data.datarobot_api_token) {
        try {
            await fetchAndStoreDataRobotToken();
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

export const fetchAndStoreDataRobotToken = async (): Promise<void> => {
    try {
        const apiKeysResponse = await drClient.get('/account/apiKeys/?isScoped=false');
        const apiKeysData = apiKeysResponse.data;

        let apiToken = null;
        try {
            if (apiKeysData && apiKeysData.data && Array.isArray(apiKeysData.data)) {
                const nonExpiringKeys = apiKeysData.data.filter(
                    (key: { expireAt: string | null; key: string }) => key.expireAt === null
                );
                if (nonExpiringKeys.length > 0) {
                    apiToken = nonExpiringKeys[0].key;
                }
            }
        } catch (apiKeyError) {
            console.warn('Could not process API keys response:', apiKeyError);
        }

        await apiClient.post('/v1/user/datarobot-account', {
            api_token: apiToken,
        });
    } catch (error) {
        console.error('Error fetching or storing DataRobot info:', error);
        throw error;
    }
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
