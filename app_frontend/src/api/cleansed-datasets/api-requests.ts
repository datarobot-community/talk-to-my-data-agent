import apiClient from '../apiClient';

type CleansedColumnReport = {
  new_column_name: string;
  original_column_name: string | null;
  errors: string[];
  warnings: string[];
  original_dtype: string | null;
  new_dtype: string | null;
  conversion_type: string | null;
};

type CleansedDataset = {
  dataset: {
    name: string;
    data_records: Record<string, never>[];
  };
  cleaning_report: CleansedColumnReport[];
  name: string;
};

export type DatasetMetadata = {
  name: string;
  dataset_type: string;
  original_name: string;
  created_at: string;
  columns: string[];
  row_count: number;
  data_source: string;
  file_size: number;
};

export const getCleansedDataset = async ({
  name,
  skip = 0,
  limit = 100,
  search,
  signal,
}: {
  name: string;
  skip?: number;
  limit?: number;
  search?: string;
  signal?: AbortSignal;
}): Promise<CleansedDataset> => {
  const encodedName = encodeURIComponent(name);
  const searchParam = search ? `&search=${encodeURIComponent(search)}` : '';
  const { data } = await apiClient.get<CleansedDataset>(
    `/v1/datasets/${encodedName}/cleansed?skip=${skip}&limit=${limit}${searchParam}`,
    { signal }
  );
  return data;
};

export const getDatasetMetadata = async ({
  name,
  signal,
}: {
  name: string;
  signal?: AbortSignal;
}): Promise<DatasetMetadata> => {
  const encodedName = encodeURIComponent(name);
  const { data } = await apiClient.get<DatasetMetadata>(`/v1/datasets/${encodedName}/metadata`, {
    signal,
  });
  return data;
};
