// global.d.ts

export {};

declare global {
    interface Window {
        ENV: {
            APP_BASE_URL?: string;
            API_PORT?: string;
            DATAROBOT_ENDPOINT?: string;
            IS_STATIC_FRONTEND?: boolean;
        };
    }
}
