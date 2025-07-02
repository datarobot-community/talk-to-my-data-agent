import { AxiosProgressEvent } from 'axios';
import apiClient from '../apiClient';

export type Dataset = {
    id: string;
    name: string;
    created: string;
    size: string;
    file_size?: number;
};

export const getDatasets = async ({
    limit,
    signal,
}: {
    limit: number;
    signal?: AbortSignal;
}): Promise<Dataset[]> => {
    const { data } = await apiClient.get<Dataset[]>(`/v1/registry/datasets?limit=${limit}`, {
        signal,
    });
    return data;
};

export async function uploadDataset({
    files,
    onUploadProgress,
    catalogIds,
    signal,
}: {
    files?: File[];
    catalogIds?: string[];
    onUploadProgress?: (progressEvent: AxiosProgressEvent) => void;
    signal?: AbortSignal;
}) {
    const formData = new FormData();

    if (files && files.length > 0) {
        files.forEach(file => formData.append('files', file));
    }

    formData.append('registry_ids', JSON.stringify(catalogIds || []));

    const response = await apiClient.post('/v1/datasets/upload', formData, {
        headers: {
            'content-type': 'multipart/form-data',
        },
        onUploadProgress,
        signal,
    });

    const { data } = response;

    return data;
}

export const deleteAllDatasets = async (): Promise<unknown> => {
    const { data } = await apiClient.delete(`/v1/datasets`);

    return data;
};
