import { AxiosProgressEvent } from 'axios';
import apiClient from '../apiClient';
import { DatasetResponse, Dataset } from './types';

/**
 * Fetch a Data Registry dataset's size (bytes) via the DataRobot reverse proxy.
 * The v2 datasets API returns it as `datasetSize`. Returns null when unknown.
 */
export async function getRegistryDatasetSize(datasetId: string): Promise<number | null> {
  const { data } = await apiClient.get<{ datasetSize?: number }>(
    `/v1/proxy/datarobot/api/v2/datasets/${datasetId}/`
  );
  return typeof data?.datasetSize === 'number' ? data.datasetSize : null;
}

/**
 * Fetch both Data Registry listings in a single request. The backend returns
 * them together because the platform permits only one AI Catalog search per
 * user at a time — two concurrent requests raced and one lost with a 409.
 */
export const getDatasets = async ({
  limit,
  signal,
}: {
  limit: number;
  signal?: AbortSignal;
}): Promise<{ local: Dataset[]; remote: Dataset[] }> => {
  const { data } = await apiClient.get<{ local: Dataset[]; remote: Dataset[] }>(
    `/v1/registry/datasets?limit=${limit}`,
    {
      signal,
    }
  );
  return data;
};

export const getDatasetById = async ({
  datasetId,
  skip = 0,
  limit = 1000,
  signal,
}: {
  datasetId: string;
  skip?: number;
  limit?: number;
  signal?: AbortSignal;
}): Promise<DatasetResponse> => {
  const { data } = await apiClient.get<DatasetResponse>(
    `/v1/datasets/${datasetId}?skip=${skip}&limit=${limit}`,
    {
      signal,
    }
  );
  return data;
};

export async function uploadDataset({
  files,
  onUploadProgress,
  catalogIds,
  dataSource,
  signal,
}: {
  files?: File[];
  catalogIds?: string[];
  dataSource?: string;
  onUploadProgress?: (progressEvent: AxiosProgressEvent) => void;
  signal?: AbortSignal;
}) {
  const formData = new FormData();

  dataSource ??= 'catalog';

  if (files && files.length > 0) {
    files.forEach(file => formData.append('files', file));
  }

  formData.append('registry_ids', JSON.stringify(catalogIds || []));

  const response = await apiClient.post(`/v1/datasets/upload?data_source=${dataSource}`, formData, {
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

export async function getSupportedDataSourceTypes(): Promise<string[]> {
  const response = await apiClient.get('/v1/supported-data-source-types');

  const { data } = response;

  return data.supported_types;
}

export const downloadDataset = async ({
  datasetId,
  signal,
  includeBom,
}: {
  datasetId: string;
  signal?: AbortSignal;
  includeBom?: boolean;
}): Promise<void> => {
  try {
    const response = await apiClient.get(`/v1/datasets/${datasetId}/download`, {
      params: { bom: includeBom },
      responseType: 'blob',
      signal,
    });

    const blob = response.data;
    const contentDisposition = response.headers['content-disposition'];
    const filename = contentDisposition?.match(/filename="?([^"]+)"?/)?.[1] || 'dataset.csv';

    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();

    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (error) {
    console.error('DEBUG Error downloading dataset:', error);
    throw error;
  }
};
