export interface PlotlyData {
    [key: string]: unknown;
}

export const parsePlotData = (
    jsonString?: string
): {
    data: PlotlyData[];
    layout: {
        paper_bgcolor: string;
        [key: string]: unknown;
    };
} | null => {
    if (!jsonString) {
        return null;
    }

    try {
        return {
            paper_bgcolor: 'rgba(255,255,255, 0)',
            ...JSON.parse(jsonString),
        };
    } catch (error) {
        console.error('Failed to parse plot data:', error);
        return null;
    }
};

export const formatMessageDate = (timestamp?: string): string => {
    if (!timestamp) {
        return '';
    }

    try {
        const date = new Date(timestamp);

        return date.toLocaleString('en-US', {
            month: 'long',
            day: 'numeric',
            year: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
            hour12: true,
        });
    } catch (error) {
        console.error('Error formatting date:', error);
        return '';
    }
};
